"""
Replay offline della pipeline FTF+MFR per ogni punto di funzionamento di FTF.

Per ogni round di un'esecuzione d'attacco e per ogni regola presente in quel round:
    regola classificata "malevola" dalla pipeline  <=>  FTF positivo nel round  AND  MFR positivo sulla regola
(MFR vede sempre la finestra completa della regola: il controller aggiorna le finestre a ogni round di
polling e invoca MFR solo se FTF è positivo).

Input:
  - probs_test.csv e selected_tau.json / operating_points.csv di sweep_ftf_threshold.py
  - dataset_MFR e modello MFR (TempBERT) salvato da train_tempBERT (model.pt + meta.joblib)

Uso:
    python replay_pipeline.py [--ftf-sweep-dir models/ftf_detector/2_100/threshold_sweep] [--mfr-expected-acc 0.98]

Limiti (da dichiarare nella relazione):
  - non modella il gate sull'occupazione della tabella (soglia dell'80%) né la lease FTF_TTL;
  - i falsi positivi di regola sono un limite inferiore del danno collaterale online: l'eviction per
    in_port con wildcard può rimuovere altre regole legittime sulla stessa porta (MAX_COLLATERAL_PORT);
  - il tempo di inferenza è misurato su questa macchina, con batch di regole, non nel controller.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Module loader (stessa convenzione degli altri script)
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
sys.path.append(current)

from training.utils import load_data_mfr
from accuracy.sweep_ftf_threshold import load_model, make_windows, predict_proba
from utils.measure_inference_time import latency_curve, INFERENCE_CSV


def confusion(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
        "accuracy": (tp + tn) / max(len(y), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfr-subdir", default="2", help="K del modello, es. 2")
    ap.add_argument("--ftf-subdir", default="2_100", help="K_W-pktin del modello, es. 2_100")
    ap.add_argument("--mfr-dataset", default=None, help="CSV del dataset MFR (default: dataset_mfr_<subdir>.csv)")
    ap.add_argument("--mfr-model-dir", default=os.path.join(root, "models", "mfr_detector"))
    ap.add_argument("--ftf-sweep-dir",
                    default=os.path.join(root, "models", "ftf_detector", "2_100", "threshold_sweep"))
    ap.add_argument("--tau-mfr", type=float, default=0.5)
    ap.add_argument("--mfr-expected-acc", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.ftf_sweep_dir
    mfr_subdir = args.mfr_subdir
    ftf_subdir = args.ftf_subdir
    os.makedirs(out_dir, exist_ok=True)
    dataset = args.mfr_dataset or os.path.join(
        root, "feature_collector", "datasets", f"dataset_mfr_{mfr_subdir}.csv")
    model_dir = os.path.join(args.mfr_model_dir, mfr_subdir, "temp_bert")

    # ---------------- MFR: inferenza sul test ----------------
    (X_train, X_val, X_test, y_train, y_val, y_test,
     _s, _p, k_train, k_val, k_test, meta_te) = load_data_mfr(dataset, return_meta=True)

    model, scaler, params, features = load_model(model_dir)
    seq, valid = make_windows(X_test, k_test, features, scaler, params["window"])
    p_mfr = predict_proba(model, seq, valid)
    y = y_test.values.astype(int)

    acc_mfr = float(((p_mfr >= args.tau_mfr).astype(int) == y).mean())
    print(f"\nMFR: W_round = {params['window']}, accuracy test a tau={args.tau_mfr}: {acc_mfr:.4f}")
    if args.mfr_expected_acc is not None and abs(acc_mfr - args.mfr_expected_acc) > 0.005:
        print(f"ATTENZIONE: attesa {args.mfr_expected_acc:.4f}: split/scaler/colonne MFR potrebbero non coincidere.")

    rules = meta_te.copy()
    rules["y"] = y
    rules["p_mfr"] = p_mfr

    # ---------------- FTF: probabilità per round ----------------
    probs = pd.read_csv(os.path.join(args.ftf_sweep_dir, "probs_test.csv"))
    ftf_scen, mfr_scen = set(probs.scenario_id.unique()), set(rules.scenario_id.unique())
    common = sorted(ftf_scen & mfr_scen)
    print(f"Scenari di test FTF: {sorted(ftf_scen)} | MFR: {sorted(mfr_scen)} | in comune: {common}")
    if ftf_scen != mfr_scen:
        print("ATTENZIONE: i test set di FTF e MFR non coincidono. Uso solo gli scenari in comune, "
              "gli unici che nessuno dei due modelli ha visto in training.")
    if not common:
        raise SystemExit("Nessuno scenario di test in comune tra FTF e MFR: il replay non è eseguibile.")

    ftf_att = probs[(probs.label == 1) & probs.scenario_id.isin(common)][["scenario_id", "round", "p_attack"]]
    ftf_norm = probs[(probs.label == 0) & probs.scenario_id.isin(common)]
    assert not ftf_att.duplicated(["scenario_id", "round"]).any(), "round duplicati nella serie d'attacco di FTF"

    rules = rules[rules.scenario_id.isin(common)].merge(ftf_att, on=["scenario_id", "round"], how="left")
    cov = float(rules.p_attack.notna().mean())
    print(f"Copertura del join per (scenario_id, round): {cov:.1%} delle regole")
    rules = rules.dropna(subset=["p_attack"]).reset_index(drop=True)
    if rules.empty:
        raise SystemExit("Join vuoto: controlla che 'round' abbia lo stesso significato nei due dataset.")

    rounds = (rules.groupby(["scenario_id", "round"])
              .agg(n_rules=("y", "size"), n_mal=("y", "sum"), p_attack=("p_attack", "first"))
              .reset_index())

    # ---------------- costo di inferenza ----------------
    xs_f, ys_f = latency_curve(INFERENCE_CSV, "ftf", ftf_subdir, "TempBERT")
    ftf_ms = float(np.interp(1, xs_f, ys_f))
    xs, ys = latency_curve(INFERENCE_CSV, "mfr", mfr_subdir, "TempBERT")
    print(f"t_FTF = {ftf_ms:.2f} ms; t_MFR(n regole) [ms]: " +
          ", ".join(f"{int(a)}->{b:.1f}" for a, b in zip(xs, ys)))

    # ---------------- punti di FTF ----------------
    with open(os.path.join(args.ftf_sweep_dir, "selected_tau.json")) as f:
        pts = {k: v for k, v in json.load(f).items() if v is not None}
    op = pd.read_csv(os.path.join(args.ftf_sweep_dir, "operating_points.csv")).set_index("point")

    rows = []
    for name, tau in pts.items():
        ftf_pos = rules.p_attack.values >= tau
        pipe = ftf_pos & (rules.p_mfr.values >= args.tau_mfr)
        c = confusion(rules.y.values, pipe)

        r = rounds.copy()
        r["ftf_pos"] = r.p_attack >= tau
        r["t_ms"] = ftf_ms + r.ftf_pos * np.interp(r.n_rules, xs, ys)
        with_mal = r[r.n_mal > 0]

        delays = []
        for _, g in rules.assign(pipe_pos=pipe).groupby("scenario_id"):
            start = g.loc[g["y"] == 1, "round"].min()
            hit = g.loc[(g["y"] == 1) & g["pipe_pos"], "round"]
            delays.append(hit.min() - start if len(hit) else np.nan)
        delays = np.array(delays, dtype=float)

        rows.append({
            "point": name, "tau_ftf": tau,
            "ftf_test_accuracy": float(op.loc[name, "test_accuracy"]) if name in op.index else np.nan,
            "ftf_round_recall": float(with_mal.ftf_pos.mean()),
            "ftf_normal_alarm_rate": float((ftf_norm.p_attack >= tau).mean()),
            **{f"pipe_{k}": v for k, v in c.items()},
            "detection_delay_rounds_mean": float(np.nanmean(delays)) if np.isfinite(delays).any() else np.nan,
            "scenarios_never_detected": int(np.isnan(delays).sum()),
            "ftf_positive_rounds_pct": float(r.ftf_pos.mean() * 100),
            "t_round_mean_ms": float(r.t_ms.mean()),
            "t_round_p95_ms": float(np.percentile(r.t_ms, 95)),
        })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "replay_pipeline.csv"), index=False)

    show = ["point", "tau_ftf", "ftf_test_accuracy", "ftf_round_recall", "ftf_normal_alarm_rate",
            "pipe_precision", "pipe_recall", "pipe_fpr", "pipe_fnr", "pipe_f1",
            "detection_delay_rounds_mean", "ftf_positive_rounds_pct", "t_round_mean_ms"]
    print("\nPipeline FTF+MFR (livello di regola, scenari di test in comune):")
    print(res[show].to_string(index=False, float_format="%.3f"))
    print(f"\nSalvato in: {os.path.join(out_dir, 'replay_pipeline.csv')}")


if __name__ == "__main__":
    main()