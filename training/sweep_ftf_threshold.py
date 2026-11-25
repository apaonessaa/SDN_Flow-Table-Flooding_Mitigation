"""
Sweep della soglia di decisione del TempBERT-FTF (senza riaddestrare).

Cosa fa:
  1. ricarica lo split (stesso di training) e il modello salvato da train_tempBERT
     (models/ftf_detector/<subdir>/temp_bert/{model.pt, meta.joblib});
  2. calcola P(attacco) su validation e test;
  3. fa variare la soglia tau, scegliendo i punti di funzionamento SOLO sulla validation
     e riportando le metriche corrispondenti sul test;
  4. salva probabilità, curve, tabella dei punti scelti, grafici e latenza di inferenza.

Prerequisito: load_data_ftf deve poter restituire anche i metadati (scenario_id, label, round)
delle righe di validation e test. Aggiungi in utils.py (senza rompere le chiamate esistenti):

    def load_data_ftf(dataset_path: str, return_meta: bool = False):
        ...  # codice esistente invariato, fino al return
        if return_meta:
            meta = {
                "val": pd.DataFrame({
                    "scenario_id": scenarios_val,
                    "label": y_val,
                    "round": df_trainval.loc[val_mask, ROUND_COL].reset_index(drop=True),
                }),
                "test": pd.DataFrame({
                    "scenario_id": scenarios_test,
                    "label": y_test,
                    "round": df.loc[test_mask, ROUND_COL].reset_index(drop=True),
                }),
            }
            return (X_train, X_val, X_test, y_train, y_val, y_test, profiles_train,
                    scenarios_train, scenarios_val, scenarios_test, meta)
        return (X_train, X_val, X_test, y_train, y_val, y_test, profiles_train,
                scenarios_train, scenarios_val, scenarios_test)

Uso (dalla cartella ftf_detector):

    python sweep_ftf_threshold.py --subdir 2_100 --expected-acc 0.87
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Module loader (stessa convenzione degli altri script)
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import load_data_ftf  # noqa: E402
from mitigation.tempbert_arch import DEVICE, TempBert, build_windows  # noqa: E402


# ----------------------------------------------------------------------------
# Inferenza
# ----------------------------------------------------------------------------
def load_model(model_dir):
    meta = joblib.load(os.path.join(model_dir, "meta.joblib"))
    scaler, params, features = meta["scaler"], meta["params"], meta["features"]
    model = TempBert(len(features), params["window"], params).to(DEVICE)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, params, features


def make_windows(X, groups, features, scaler, window):
    """Stessa pre-elaborazione del training: QuantileTransformer fittato sul train + finestre per scenario."""
    X = X[features]  # stesso ordine di colonne usato in training
    Xs = scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
    return build_windows(Xs, np.asarray(groups), window)


@torch.no_grad()
def predict_proba(model, seq, valid, batch_size=512):
    """P(attacco) = softmax(logits)[:, 1], nello stesso ordine delle righe."""
    model.eval()
    ds = TensorDataset(torch.from_numpy(seq), torch.from_numpy(valid))
    out = []
    for xb, vb in DataLoader(ds, batch_size=batch_size, shuffle=False):
        logits = model(xb.to(DEVICE), vb.to(DEVICE))
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def measure_latency(model, seq, valid, n=200, warmup=20, seed=0):
    """Latenza di inferenza FTF per una singola finestra (batch 1), in ms."""
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seq), n + warmup)
    times = []
    for k, i in enumerate(idx):
        x = torch.from_numpy(seq[i:i + 1]).to(DEVICE)
        v = torch.from_numpy(valid[i:i + 1]).to(DEVICE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x, v)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)),
            "max_ms": float(t.max()), "device": DEVICE, "n": int(n)}


# ----------------------------------------------------------------------------
# Metriche e sweep
# ----------------------------------------------------------------------------
def metrics_at(y, p, tau):
    y = np.asarray(y).astype(int)
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "tau": float(tau),
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def make_tau_grid(p_val, n_lin=201, n_quant=101):
    """Griglia lineare + quantili delle probabilità (utile se le probabilità sono sature vicino a 0/1)."""
    lin = np.linspace(0.0, 1.0, n_lin)
    quant = np.quantile(p_val, np.linspace(0.0, 1.0, n_quant))
    return np.unique(np.concatenate([lin, quant, [0.5]]))


def sweep(y, p, taus):
    return pd.DataFrame([metrics_at(y, p, t) for t in taus])


def pick_points(dv, recalls=(0.5, 0.7, 0.8, 0.9, 0.95), fprs=(0.01, 0.05, 0.10)):
    """Sceglie i tau sulla VALIDATION. recall e fpr sono non crescenti in tau."""
    pts = {
        "tau_0.5": 0.5,
        "max_accuracy": float(dv.loc[dv.accuracy.idxmax(), "tau"]),
        "max_f1": float(dv.loc[dv.f1.idxmax(), "tau"]),
    }
    for r in recalls:  # tau più alto che garantisce recall >= r
        ok = dv[dv.recall >= r]
        pts[f"recall>={r}"] = float(ok.tau.max()) if len(ok) else None
    for f in fprs:  # tau più basso che garantisce fpr <= f
        ok = dv[dv.fpr <= f]
        pts[f"fpr<={f}"] = float(ok.tau.min()) if len(ok) else None
    return pts


def pick_points_by_accuracy(dsel, targets=(0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55), tol=0.02):
    """
    Punti per livello di accuracy. L'accuracy NON è monotona in tau: è massima in tau* e scende
    verso la prevalenza di classe (~0.5) agli estremi, quindi ogni livello si raggiunge due volte:
      - ramo "fn" (tau > tau*): FTF conservativo, pochi falsi positivi e molti falsi negativi;
      - ramo "fp" (tau < tau*): FTF sensibile, molti falsi positivi e pochi falsi negativi.
    Gli estremi sono i casi limite: tau=0 (FTF sempre positivo, MFR sempre attivo) e
    tau>1 (FTF sempre negativo, MFR mai attivo: nessuna difesa).
    Un livello non raggiungibile (accuratezza più vicina oltre `tol` dal target) viene saltato.
    """
    d = dsel.sort_values("tau").reset_index(drop=True)
    i_star = int(d.accuracy.values.argmax())
    tau_star = float(d.tau.iloc[i_star])
    pts = {"always_attack(tau=0)": 0.0, "opt_accuracy": tau_star, "never_attack(tau>1)": 1.01}
    seen = set(pts.values())

    for branch, part in (("fn", d.iloc[i_star:]), ("fp", d.iloc[: i_star + 1])):
        for t in targets:
            j = (part.accuracy - t).abs().idxmin()
            if abs(float(d.accuracy.iloc[j]) - t) > tol:
                continue
            tau = float(d.tau.iloc[j])
            if tau in seen:
                continue
            seen.add(tau)
            pts[f"acc~{t:.2f}_{branch}"] = tau
    return pts


def plot_curves(dv, dt, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponibile: grafici saltati")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, d, name in zip(axes, (dv, dt), ("Validation", "Test")):
        for col in ("accuracy", "precision", "recall", "f1", "fpr"):
            ax.plot(d.tau, d[col], label=col)
        ax.set_title(name)
        ax.set_xlabel("soglia τ su P(attacco)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_vs_tau.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(dt.fpr, dt.recall, label="test")
    ax.plot(dv.fpr, dv.recall, label="validation", alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("Recall (TPR)")
    ax.set_title("ROC")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="2_100", help="K_Wpktin del modello, es. 2_100")
    ap.add_argument("--dataset", default=None, help="CSV del dataset FTF (default: dataset_ftf_<subdir>.csv)")
    ap.add_argument("--models-dir", default=os.path.join(root, "models", "ftf_detector"))
    ap.add_argument("--expected-acc", type=float, default=None,
                    help="accuracy test attesa (da tabella dei risultati) per il controllo di riproducibilità")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--points", choices=["accuracy", "legacy"], default="accuracy",
                    help="accuracy: scala di livelli di accuracy su due rami + estremi; "
                         "legacy: soglie per recall/FPR target")
    ap.add_argument("--select-on", choices=["val", "test"], default="val",
                    help="split su cui scegliere tau (val: le metriche su test restano indipendenti)")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
                    help="livelli di accuracy per --points accuracy")
    args = ap.parse_args()

    dataset = args.dataset or os.path.join(
        root, "feature_collector", "datasets", f"dataset_ftf_{args.subdir}.csv")
    model_dir = os.path.join(args.models_dir, args.subdir, "temp_bert")
    out_dir = os.path.join(args.models_dir, args.subdir, "threshold_sweep")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset : {dataset}")
    print(f"Modello : {model_dir}")
    print(f"Device  : {DEVICE}")

    # 1) split identico a quello di training
    (X_train, X_val, X_test, y_train, y_val, y_test,
     _profiles, _g_train, g_val, g_test, meta) = load_data_ftf(dataset, return_meta=True)

    # 2) modello + pre-elaborazione
    model, scaler, params, features = load_model(model_dir)
    w = params["window"]
    print(f"Finestra W_round = {w}, n_feature = {len(features)}")

    seq_va, valid_va = make_windows(X_val, g_val, features, scaler, w)
    seq_te, valid_te = make_windows(X_test, g_test, features, scaler, w)

    p_val = predict_proba(model, seq_va, valid_va, args.batch_size)
    p_te = predict_proba(model, seq_te, valid_te, args.batch_size)
    yv, yt = y_val.values.astype(int), y_test.values.astype(int)

    # 3) controllo di riproducibilità
    acc_05 = float(((p_te >= 0.5).astype(int) == yt).mean())
    print(f"\nScenari di test: {sorted(meta['test'].scenario_id.unique())}")
    print(f"Accuracy test a tau=0.5: {acc_05:.4f}")
    if args.expected_acc is not None and abs(acc_05 - args.expected_acc) > 0.005:
        print(f"ATTENZIONE: attesa {args.expected_acc:.4f}. Split, scaler o colonne potrebbero non coincidere "
              f"con quelli del training: i risultati dello sweep non sono affidabili.")

    # 4) salvataggio probabilità (servono anche per il replay della pipeline)
    meta["val"].assign(p_attack=p_val).to_csv(os.path.join(out_dir, "probs_val.csv"), index=False)
    meta["test"].assign(p_attack=p_te).to_csv(os.path.join(out_dir, "probs_test.csv"), index=False)

    # 5) misure indipendenti dalla soglia
    summary = {
        "subdir": args.subdir, "window": int(w),
        "roc_auc_val": float(roc_auc_score(yv, p_val)), "pr_auc_val": float(average_precision_score(yv, p_val)),
        "roc_auc_test": float(roc_auc_score(yt, p_te)), "pr_auc_test": float(average_precision_score(yt, p_te)),
        "acc_test_tau_0.5": acc_05,
        "frac_attack_val": float(yv.mean()), "frac_attack_test": float(yt.mean()),
    }
    print("\nMisure indipendenti dalla soglia:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 6) sweep
    taus = make_tau_grid(p_val)
    dv = sweep(yv, p_val, taus)
    dt = sweep(yt, p_te, taus)
    dv.to_csv(os.path.join(out_dir, "sweep_val.csv"), index=False)
    dt.to_csv(os.path.join(out_dir, "sweep_test.csv"), index=False)

    # 7) punti di funzionamento: tau scelti sulla validation, metriche sul test
    dsel = dv if args.select_on == "val" else dt
    if args.points == "accuracy":
        pts = pick_points_by_accuracy(dsel, targets=tuple(args.targets))
    else:
        pts = pick_points(dsel)
    rows = []
    for name, tau in pts.items():
        if tau is None:
            rows.append({"point": name, "tau": None})
            continue
        mv, mt = metrics_at(yv, p_val, tau), metrics_at(yt, p_te, tau)
        rows.append({"point": name, "tau": tau,
                     **{f"val_{k}": v for k, v in mv.items() if k != "tau"},
                     **{f"test_{k}": v for k, v in mt.items() if k != "tau"}})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "operating_points.csv"), index=False)
    with open(os.path.join(out_dir, "selected_tau.json"), "w") as f:
        json.dump(pts, f, indent=2)

    cols = ["point", "tau", "val_accuracy", "test_accuracy", "test_precision", "test_recall",
            "test_fpr", "test_fnr", "test_f1"]
    print("\nPunti di funzionamento (tau scelto su validation, metriche su test):")
    print(table[[c for c in cols if c in table.columns]].to_string(index=False, float_format="%.3f"))

    # 8) latenza di inferenza di FTF (singola finestra)
    lat = measure_latency(model, seq_te, valid_te)
    summary["ftf_latency"] = lat
    print(f"\nLatenza FTF (batch 1, {lat['device']}): media {lat['mean_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 9) grafici
    plot_curves(dv, dt, out_dir)
    print(f"\nRisultati salvati in: {out_dir}")


if __name__ == "__main__":
    main()