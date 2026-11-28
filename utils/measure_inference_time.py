"""
Misura del tempo di inferenza di tutti i modelli (Decision Tree, Random Forest, XGBoost, CatBoost,
FeatBERT, TempBERT), per FTF e per MFR, con LO STESSO metodo.

Cosa viene cronometrato (stessa definizione per tutti):
  - input: righe/finestre REALI del test set, già pre-elaborate (scaler, finestre) fuori dal cronometro;
  - alberi e boosting: model.predict(X) su un batch di n righe;
  - FeatBERT / TempBERT: forward pass del modello + argmax su un batch di n righe/finestre;
  - batch di dimensioni diverse (n = 1, 8, ..., 1024), `WARMUP` esecuzioni scartate e `REPS` cronometrate.
Non sono inclusi: lettura dei dati, QuantileTransformer, costruzione delle finestre, estrazione delle feature,
messaggi OpenFlow. Quindi è il costo del solo modello, non dell'intero percorso di decisione nel controller.

Output (in models/):
  - inference_times.csv          una riga per (task, subdir, modello, batch): media, std, p50, p95, ms per campione
  - inference_times_summary.csv  una riga per (task, subdir, modello): da usare nelle tabelle della relazione

Altri script possono leggere inference_times.csv con latency_curve() (vedi in fondo).

Uso:
    python measure_inference_time.py --only-task mfr --only-model TempBERT
"""
import argparse
import os
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

# Module loader (stessa convenzione degli altri script)
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
sys.path.append(current)

from training.utils import load_data_ftf, load_data_mfr
from mitigation.tempbert_arch import DEVICE, TempBert
from accuracy.sweep_ftf_threshold import make_windows
from training.feat_bert import FeatBert

# ----------------------------------------------------------------------------
# Configurazione: modifica solo qui
# ----------------------------------------------------------------------------
TASKS = [("ftf", "2_100"), ("mfr", "2")]  # (task, sottocartella del modello/dataset)
MODELS = ["Decision Tree", "Random Forest", "XGBoost", "CatBoost", "FeatBERT", "TempBERT"]
BATCH_SIZES = [1, 8, 32, 128, 512, 1024]
REPS = 30        # esecuzioni cronometrate per ogni batch
WARMUP = 5       # esecuzioni iniziali scartate (allocazioni, cache fredde)
THREADS = 1      # stessi thread per tutti i modelli (None = default della libreria); riportato nel CSV
SEED = 0         # seed del campionamento delle righe del batch (non influenza il tempo del modello)

INFERENCE_CSV = os.path.join(root, "models", "inference_times.csv")
SUMMARY_CSV = os.path.join(root, "models", "inference_times_summary.csv")

# Percorsi dei modelli, relativi a models/<task>_detector/<subdir>/.
MODEL_FILES = {
    "ftf": {
        "Decision Tree": "decision_tree_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "XGBoost": "xgboost_model.json",
        "CatBoost": "catboost_model.cbm",
        "FeatBERT": os.path.join("feat_bert", "model.pt"),
        "TempBERT": os.path.join("temp_bert", "model.pt"),
    },
    "mfr": {
        "Decision Tree": "decision_tree_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "XGBoost": "xgboost_model.json",
        "CatBoost": "catboost_model.cbm",
        "FeatBERT": os.path.join("feat_bert", "model.pt"),
        "TempBERT": os.path.join("temp_bert", "model.pt"),
    },
}

def models_dir(task, sub):
    return os.path.join(root, "models", f"{task}_detector", sub)


def dataset_path(task, sub):
    return os.path.join(root, "feature_collector", "datasets", f"dataset_{task}_{sub}.csv")


# ----------------------------------------------------------------------------
# Cronometro
# ----------------------------------------------------------------------------
def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_call(fn, reps=REPS, warmup=WARMUP):
    """Tempi (ms) di `reps` chiamate di fn(), dopo `warmup` chiamate scartate."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(ts)

@torch.no_grad()
def nn_predict_all(model, x_np, valid_np=None, bs=512):
    out = []
    for i in range(0, len(x_np), bs):
        x = torch.from_numpy(x_np[i:i + bs]).to(DEVICE)
        if valid_np is None:
            logits = model(x)
        else:
            logits = model(x, torch.from_numpy(valid_np[i:i + bs]).to(DEVICE))
        out.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Ricerca e caricamento dei modelli
# ----------------------------------------------------------------------------
def resolve_file(task, sub, model_name):
    path = os.path.join(models_dir(task, sub), MODEL_FILES[task][model_name])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{model_name}: file non trovato: {path}")
    return path


def load_tabular(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".joblib", ".pkl", ".pickle"):
        obj = joblib.load(path)
        if isinstance(obj, dict):
            for k in ("model", "clf", "estimator"):
                if k in obj:
                    return obj[k]
            raise ValueError(f"{path}: dizionario senza chiave 'model'/'clf'/'estimator' ({list(obj)})")
        return obj
    if ext == ".cbm":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(path)
        return m
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(path)
    return m

def load_nn(model_name, pt_path):
    d = os.path.dirname(pt_path)
    meta = joblib.load(os.path.join(d, "meta.joblib"))
    scaler, params, features = meta["scaler"], meta["params"], meta["features"]
    if model_name == "TempBERT":
        model = TempBert(len(features), params["window"], params)
    else:
        model = FeatBert(len(features), params)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model, scaler, params, features


# ----------------------------------------------------------------------------
# Dati reali del test set
# ----------------------------------------------------------------------------
def load_pool(task, sub):
    if task == "ftf":
        out = load_data_ftf(dataset_path(task, sub))
        return out[2], out[9], out[5]   # X_test, scenarios_test, y_test
    out = load_data_mfr(dataset_path(task, sub))
    return out[2], out[10], out[5]      # X_test, keys_test, y_test


# ----------------------------------------------------------------------------
# Misura di un modello
# ----------------------------------------------------------------------------
def measure_model(task, sub, model_name, X_pool, groups, y_pool, batch_sizes, threads):
    path = resolve_file(task, sub, model_name)
    rng = np.random.default_rng(SEED)
    n_params, runners = np.nan, {}
    acc = np.nan

    if model_name in ("FeatBERT", "TempBERT"):
        model, scaler, params, features = load_nn(model_name, path)
        n_params = int(sum(p.numel() for p in model.parameters()))
        if model_name == "TempBERT":
            seq, valid = make_windows(X_pool, groups, features, scaler, params["window"])
            pool_n = len(seq)
            acc = float((nn_predict_all(model, seq, valid) == y_pool.values).mean())
        else:
            Xs = scaler.transform(np.asarray(X_pool[features], dtype=np.float64)).astype(np.float32)
            pool_n = len(Xs)
            acc = float((nn_predict_all(model, Xs) == y_pool.values).mean())

        for n in batch_sizes:
            idx = rng.choice(pool_n, n, replace=pool_n < n)
            if model_name == "TempBERT":
                x = torch.from_numpy(seq[idx]).to(DEVICE)
                v = torch.from_numpy(valid[idx]).to(DEVICE)

                @torch.no_grad()
                def run(x=x, v=v):
                    return model(x, v).argmax(-1)
            else:
                x = torch.from_numpy(Xs[idx]).to(DEVICE)

                @torch.no_grad()
                def run(x=x):
                    return model(x).argmax(-1)
            runners[n] = run
    else:
        model = load_tabular(path)
        acc = float((np.asarray(model.predict(X_pool)).ravel() == y_pool.values).mean())
        print(f"[{model_name}] best_iteration: {getattr(model, 'best_iteration', 'n/d')}")
        is_cat = "CatBoost" in type(model).__name__
        if threads and not is_cat and hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=threads)
            except Exception:
                pass
        for n in batch_sizes:
            idx = rng.choice(len(X_pool), n, replace=len(X_pool) < n)
            X = X_pool.iloc[idx]
            if is_cat:
                runners[n] = lambda X=X: model.predict(X, thread_count=threads or -1)
            else:
                runners[n] = lambda X=X: model.predict(X)

    rows = []
    for n, run in runners.items():
        ts = time_call(run)
        rows.append({
            "task": task, "subdir": sub, "model": model_name, "batch": n,
            "mean_ms": float(ts.mean()), "std_ms": float(ts.std()),
            "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "per_sample_ms": float(ts.mean() / n),
            "reps": REPS, "warmup": WARMUP, "threads": threads if threads else -1,
            "device": DEVICE, "cpu": platform.processor() or platform.machine(),
            "n_params": n_params, "model_file_kb": round(os.path.getsize(path) / 1024, 1),
            "test_accuracy_check": acc,
        })
    return rows


# ----------------------------------------------------------------------------
# Tabelle riassuntive
# ----------------------------------------------------------------------------
def build_summary(df):
    out = []
    for (task, sub, model), d in df.groupby(["task", "subdir", "model"]):
        d = d.sort_values("batch")
        one = d[d.batch == d.batch.min()].iloc[0]
        big = d.iloc[-1]
        out.append({
            "task": task, "subdir": sub, "model": model,
            f"t_batch{int(one.batch)}_mean_ms": one.mean_ms, f"t_batch{int(one.batch)}_p95_ms": one.p95_ms,
            f"per_sample_ms_batch{int(big.batch)}": big.per_sample_ms,
            "n_params": one.n_params, "model_file_kb": one.model_file_kb,
            "threads": one.threads, "device": one.device,
        })
    return pd.DataFrame(out)


def latency_curve(csv_path, task, subdir, model="TempBERT"):
    """Per altri script: (batch, mean_ms) di un modello, da inference_times.csv."""
    df = pd.read_csv(csv_path, dtype={"subdir": str})
    d = df[(df.task == task) & (df.subdir == subdir) & (df.model == model)].sort_values("batch")
    if d.empty:
        raise SystemExit(f"Nessuna misura per {task}/{subdir}/{model} in {csv_path}: esegui measure_inference_time.py")
    return d.batch.values.astype(float), d.mean_ms.values.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-task", choices=["ftf", "mfr"], default=None)
    ap.add_argument("--only-model", choices=MODELS, default=None)
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    if args.threads and DEVICE == "cpu":
        torch.set_num_threads(args.threads)
    print(f"Device: {DEVICE} | thread torch: {torch.get_num_threads()} | CPU: {platform.processor() or platform.machine()}")

    rows = []
    for task, sub in TASKS:
        if args.only_task and task != args.only_task:
            continue
        print(f"\n##### {task.upper()} ({sub}) #####")
        X_pool, groups, y_pool = load_pool(task, sub)
        for model_name in MODELS:
            if args.only_model and model_name != args.only_model:
                continue
            try:
                r = measure_model(task, sub, model_name, X_pool, groups, y_pool, BATCH_SIZES, args.threads)
            except Exception as e:  # un modello che non si carica non blocca gli altri
                print(f"[{model_name}] saltato: {e}")
                continue
            rows += r
            one = r[0]
            print(f"[{model_name}] batch {one['batch']}: {one['mean_ms']:.3f} ms (p95 {one['p95_ms']:.3f})")

    if not rows:
        raise SystemExit("Nessuna misura prodotta.")
    new = pd.DataFrame(rows)

    # unisce alle misure già presenti (si può rilanciare solo un modello o un task)
    if os.path.exists(INFERENCE_CSV):
        old = pd.read_csv(INFERENCE_CSV, dtype={"subdir": str})
        keys = set(zip(new.task, new.subdir, new.model))
        old = old[[k not in keys for k in zip(old.task, old.subdir, old.model)]]
        new = pd.concat([old, new], ignore_index=True)
    os.makedirs(os.path.dirname(INFERENCE_CSV), exist_ok=True)
    new.to_csv(INFERENCE_CSV, index=False)

    summary = build_summary(new)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("\nRiepilogo:")
    print(summary.to_string(index=False, float_format="%.3f"))
    print(f"\nSalvato in:\n  {INFERENCE_CSV}\n  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()