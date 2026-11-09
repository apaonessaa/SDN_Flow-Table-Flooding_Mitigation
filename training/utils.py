import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Optuna Visualization
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from optuna.visualization import matplotlib as optuna_plt

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FTF_FEATURES = [
    "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
    "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
    "c_used", "paf", "ft_psi", "durations_entropy",
    "pkts_entropy", "bytes_entropy", "mean_duration", "cvd",
    "mean_pkts", "cvp", "mean_bytes", "cvb",
]
MFR_FEATURES = [
    "duration", "packet_count", "byte_count",
    "packets_per_seconds", "bytes_per_seconds", "psi",
]
TARGET = "label"
SCENARIO_COL = "scenario_id"
ROUND_COL = "round"
ATTACK_PROFILE_COL = "attack_profile"
BENIGN_PROFILE_LABEL = "benigni"

# MFR dataset
RULE_KEY_COL = "match_hash"
FLOW_KEY_COL = "flow_key"

# Numpy SEED
np.random.seed(RANDOM_SEED)

def load_data_ftf(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(f"Numero di entry del dataset: {len(df)}")

    """
    Order by scenario_id, label, round
    """
    df = df.sort_values([SCENARIO_COL, TARGET, ROUND_COL]).reset_index(drop=True)

    counts = df.groupby([SCENARIO_COL, TARGET]).size().unstack(fill_value=0)
    print(counts)
    print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")
    
    X = df[FTF_FEATURES]
    y = df[TARGET]
    scenarios = df[SCENARIO_COL]

    """
    (1) Split for train and test dataset
    """
    trainval_scenarios, test_scenarios = stratified_group_split(
        df, SCENARIO_COL, ATTACK_PROFILE_COL, 
        test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    trainval_mask = scenarios.isin(trainval_scenarios).to_numpy()
    test_mask = scenarios.isin(test_scenarios).to_numpy()

    df_trainval = df[trainval_mask].reset_index(drop=True)
    X_trainval = X[trainval_mask].reset_index(drop=True)
    y_trainval = y[trainval_mask].reset_index(drop=True)
    scenarios_trainval = scenarios[trainval_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    """
    (2) Split for train and validation dataset
    """
    train_scenarios, val_scenarios = stratified_group_split(
        df_trainval, SCENARIO_COL, ATTACK_PROFILE_COL, 
        test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    train_mask = scenarios_trainval.isin(train_scenarios).to_numpy()
    val_mask = scenarios_trainval.isin(val_scenarios).to_numpy()

    X_train = X_trainval[train_mask].reset_index(drop=True)
    y_train = y_trainval[train_mask].reset_index(drop=True)
    scenarios_train = scenarios_trainval[train_mask].reset_index(drop=True)
    df_train = df_trainval[train_mask].reset_index(drop=True)

    X_val = X_trainval[val_mask].reset_index(drop=True)
    y_val = y_trainval[val_mask].reset_index(drop=True)

    profiles_train = df_train[ATTACK_PROFILE_COL].reset_index(drop=True)
    
    scenarios_val = scenarios_trainval[val_mask].reset_index(drop=True)
    scenarios_test = scenarios[test_mask].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, profiles_train, scenarios_train, scenarios_val, scenarios_test

def load_data_mfr(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(f"Numero di entry del dataset: {len(df)}")

    """
    Group and Order by scenario_id+match_hash
    """
    df[FLOW_KEY_COL] = df[SCENARIO_COL].astype(str) + "_" + df[RULE_KEY_COL].astype(str)
    df = df.sort_values([SCENARIO_COL, FLOW_KEY_COL, ROUND_COL]).reset_index(drop=True)

    counts = df.groupby([SCENARIO_COL, TARGET]).size().unstack(fill_value=0)
    print(counts)
    print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")
    
    X = df[MFR_FEATURES]
    y = df[TARGET]
    scenarios = df[SCENARIO_COL]
    keys = df[FLOW_KEY_COL]

    """
    (1) Split for train and test dataset
    """
    trainval_scenarios, test_scenarios = stratified_group_split(
        df, SCENARIO_COL, ATTACK_PROFILE_COL, 
        test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    trainval_mask = scenarios.isin(trainval_scenarios).to_numpy()
    test_mask = scenarios.isin(test_scenarios).to_numpy()

    df_trainval = df[trainval_mask].reset_index(drop=True)
    X_trainval = X[trainval_mask].reset_index(drop=True)
    y_trainval = y[trainval_mask].reset_index(drop=True)
    scenarios_trainval = scenarios[trainval_mask].reset_index(drop=True)
    keys_trainval = keys[trainval_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    keys_test = keys[test_mask].reset_index(drop=True)

    """
    (2) Split for train and validation dataset
    """
    train_scenarios, val_scenarios = stratified_group_split(
        df_trainval, SCENARIO_COL, ATTACK_PROFILE_COL, 
        test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    train_mask = scenarios_trainval.isin(train_scenarios).to_numpy()
    val_mask = scenarios_trainval.isin(val_scenarios).to_numpy()

    X_train = X_trainval[train_mask].reset_index(drop=True)
    y_train = y_trainval[train_mask].reset_index(drop=True)
    scenarios_train = scenarios_trainval[train_mask].reset_index(drop=True)
    keys_train = keys_trainval[train_mask].reset_index(drop=True)
    df_train = df_trainval[train_mask].reset_index(drop=True)

    X_val = X_trainval[val_mask].reset_index(drop=True)
    y_val = y_trainval[val_mask].reset_index(drop=True)
    keys_val = keys_trainval[val_mask].reset_index(drop=True)

    profiles_train = df_train[ATTACK_PROFILE_COL].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, scenarios_train, profiles_train, keys_train, keys_val, keys_test

"""Stratified utility"""
def stratified_group_split(df, scenario_col, profile_col, test_size, random_state=RANDOM_SEED):
    def _group_profiles(df, group_col, profile_col):
        """
        Get the ATTACK_PROFILE for scenario, 
        if there is not ATT_PROFILE return the BENIGN profile
        """
        def pick(s):
            non_benign = s[s != BENIGN_PROFILE_LABEL]
            if len(non_benign) > 0:
                return non_benign.mode().iat[0]
            return BENIGN_PROFILE_LABEL
        return df.groupby(group_col)[profile_col].agg(pick)

    scenario_profiles = _group_profiles(df, scenario_col, profile_col)
    scenarios_unique = scenario_profiles.index.to_numpy()
    profiles_unique = scenario_profiles.to_numpy()

    try:
        left, right = train_test_split(
            scenarios_unique,
            test_size=test_size,
            stratify=profiles_unique,
            random_state=RANDOM_SEED,
        )
    except ValueError as e:
        print(f"[!] Stratificazione per '{profile_col}' non applicabile ({e}); "
              f"eseguo uno split non stratificato.")
        left, right = train_test_split(
            scenarios_unique, 
            test_size=test_size, 
            random_state=random_state,
        )
    return left, right

"""KFold split"""
def make_group_kfold_splits(X, scenarios, profiles):
    n_scenarios = scenarios.nunique() # scenarios_id
    effective_splits = min(N_SPLITS, n_scenarios)
    if effective_splits < 2:
        raise ValueError(
            f"Servono almeno 2 scenari distinti per la StratifiedGroupKFold, trovati {n_scenarios}."
        )
    if effective_splits < N_SPLITS:
        print(f"[!] Solo {n_scenarios} gruppi disponibili: uso {effective_splits}-fold invece di {N_SPLITS}-fold.")
    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_SEED)
    return list(sgkf.split(X, profiles, groups=scenarios))

"""Evaluate the model"""
def evaluate(name, y_true, y_pred, results, elapsed=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    print(f"\n=== {name} ===")
    if elapsed is not None:
        print(f"Tempo di training: {elapsed:.1f}s")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))

    row = {"model": name, "accuracy": acc, "precision": prec,
           "recall": rec, "f1": f1, "train_time_s": elapsed}

    results.append(row)

# --------------------------------------------------------------------------- #
# BERT
# --------------------------------------------------------------------------- #
# 1. Preprocessing
# --------------------------------------------------------------------------- #
def scale_features(X_fit, *X_others):
    """
    Data pre-processing: Gorishniy et al. 2021
    """
    pipeline = Pipeline([
        ("quantile", QuantileTransformer(output_distribution="normal", random_state=RANDOM_SEED))
    ])
    pipeline.fit(np.asarray(X_fit, dtype=np.float64))

    def _tf(X):
        return pipeline.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    return (_tf(X_fit), *[_tf(X) for X in X_others], pipeline)

# --------------------------------------------------------------------------- #
# 2. Initialization
# --------------------------------------------------------------------------- #
def init_uniform_(*tensors, fan_in):
    """
    Uniform rule for all input parameters in FeatBERT and TempBERT (feature tokens, 
    round projection, [CLS]):
        
        U(-1/sqrt(fan_in), +1/sqrt(fan_in)).

    This corresponds to `kaiming_uniform_(a=sqrt(5))` - the default for `nn.Linear` and 
    the scheme used in the FT-Transformer codebase.    
    
        `fan_in = d` for token vectors (weights, biases, and the `[CLS]` token, following 
            the `(n, d)` layout from the original code). 
        `fan_in = F` for the `Linear(F, d)` layer in TempBERT (which aligns with the 
        PyTorch default). 
    
    The BERT encoder retains the HuggingFace initialization.
    """
    bound = float(fan_in) ** -0.5
    for t in tensors:
        nn.init.uniform_(t, -bound, bound)

# --------------------------------------------------------------------------- #
# 3. Training
# --------------------------------------------------------------------------- #
def class_weights(y):
    counts = np.bincount(np.asarray(y), minlength=2)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)

def _loader(inputs, y, batch_size, shuffle):
    ds = TensorDataset(*[torch.from_numpy(a) for a in inputs],
                       torch.as_tensor(np.asarray(y), dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def predict_model(model, inputs, batch_size=512):
    model.eval()
    preds = []
    for *xs, _ in _loader(inputs, np.zeros(len(inputs[0])), batch_size, False):
        preds.append(model(*[x.to(DEVICE) for x in xs]).argmax(-1).cpu().numpy())
    return np.concatenate(preds)

# Parameters excluded from weight decay (in addition to biases and LayerNorm - i.e., 
# tensors with ndim < 2).
# As in FT-Transformer (Gorishniy et al., 2021, App. E.3): the tokenizer/input projection 
# and [CLS] do not undergo weight decay.
NO_DECAY_PREFIXES = ("tokenizer.", "proj.", "cls")

def fit_model(model_fn, train_inputs, y_tr, val_inputs, y_va, params, epochs, patience):
    """
    AdamW with constant learning rate (no scheduler, Gorishniy et al. 2021, §4.3), 
    class-weighted loss, gradient clip at 1.0, early stopping on F1 validation 
    and restore the best model.
    """
    torch.manual_seed(RANDOM_SEED)
    model = model_fn().to(DEVICE)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.startswith(NO_DECAY_PREFIXES) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": params["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=params["learning_rate"],
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr).to(DEVICE))
    loader = _loader(train_inputs, y_tr, params["batch_size"], shuffle=True)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for *xs, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(*[x.to(DEVICE) for x in xs]), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        f1 = f1_score(y_va, predict_model(model, val_inputs), zero_division=0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)

    del opt, loader, loss_fn, best_state
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return model, best_f1
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# FEATURE IMPORTANCE
# --------------------------------------------------------------------------- #

def permutation_importance(predict_fn, X, y, features, n_repeats=5):
    """
    Permutation Importance based on F1-score.
    X can be a pd.DataFrame (CatBoost, XGBoost) or a np.ndarray (FeatBERT).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    y = np.asarray(y)
    base = f1_score(y, predict_fn(X), zero_division=0)

    is_df = isinstance(X, pd.DataFrame)
    X_p = X.copy()
    drops = np.zeros((len(features), n_repeats))

    for j, feat in enumerate(features):
        orig = X_p[feat].copy() if is_df else X_p[:, j].copy()
        for r in range(n_repeats):
            perm_idx = rng.permutation(len(X_p))
            if is_df:
                X_p[feat] = orig.to_numpy()[perm_idx]
            else:
                X_p[:, j] = orig[perm_idx]
            f1 = f1_score(y, predict_fn(X_p), zero_division=0)
            drops[j, r] = base - f1
        if is_df:
            X_p[feat] = orig
        else:
            X_p[:, j] = orig

    return pd.Series(drops.mean(axis=1), index=list(features))

def permutation_importance_seq(predict_fn, seq, valid, y, features, n_repeats=5):
    """
    Permutation Importance based on F1-score for TempBERT.
    The input is (round, valid).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    y = np.asarray(y)
    base = f1_score(y, predict_fn(seq, valid), zero_division=0)
    seq_p = seq.copy()
    drops = np.zeros((len(features), n_repeats))
    for j in range(len(features)):
        orig = seq[:, :, j]
        for r in range(n_repeats):
            perm = rng.permutation(len(seq))
            col = orig[perm].copy()
            col[~valid] = 0.0
            seq_p[:, :, j] = col
            f1 = f1_score(y, predict_fn(seq_p, valid), zero_division=0)
            drops[j, r] = base - f1
        seq_p[:, :, j] = orig
    return pd.Series(drops.mean(axis=1), index=list(features))

def to_percent(importances: pd.Series) -> pd.Series:
    clipped = importances.clip(lower=0)
    total = clipped.sum()
    if total == 0:
        return clipped
    return clipped / total * 100
# --------------------------------------------------------------------------- #
# OPTUNA VISUALIZATION
# --------------------------------------------------------------------------- #
def _save_ax(ax, path):
    """Optuna restituisce un Axes oppure un array di Axes: prendo la figura."""
    fig = np.ravel(ax)[0].figure
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_study_plots(study, name, out_dir, top_k_contour=4):
    """Salva dati grezzi + plot di uno studio Optuna in out_dir/name/."""
    out_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # --- dati grezzi (per tabella in appendice e per rigenerare i plot senza rifare il tuning)
    study.trials_dataframe().to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    joblib.dump(study, os.path.join(out_dir, "study.joblib"))

    n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
    n_pruned = sum(t.state == TrialState.PRUNED for t in study.trials)
    print(f"[{name}] trial completati: {n_complete}, potati: {n_pruned}, "
          f"best trial: #{study.best_trial.number} (F1={study.best_value:.4f})")

    # --- parametri più importanti (per il contour plot)
    try:
        imp = optuna.importance.get_param_importances(study)
        top_params = list(imp.keys())[:top_k_contour]
    except Exception:
        top_params = None

    has_intermediate = any(t.intermediate_values for t in study.trials)

    plots = {
        "optimization_history": lambda: optuna_plt.plot_optimization_history(study),
        "param_importances":    lambda: optuna_plt.plot_param_importances(study),
        "slice":                lambda: optuna_plt.plot_slice(study),
        "parallel_coordinate":  lambda: optuna_plt.plot_parallel_coordinate(study),
        "edf":                  lambda: optuna_plt.plot_edf(study),
        "timeline":             lambda: optuna_plt.plot_timeline(study),
    }
    if top_params and len(top_params) >= 2:
        plots["contour"] = lambda: optuna_plt.plot_contour(study, params=top_params)
    if has_intermediate:   # solo studi con trial.report (FeatBERT / TempBERT)
        plots["intermediate_values"] = lambda: optuna_plt.plot_intermediate_values(study)

    for plot_name, fn in plots.items():
        try:
            _save_ax(fn(), os.path.join(out_dir, f"{plot_name}.png"))
        except Exception as e:     # es. pochi trial, parametri costanti, versione optuna vecchia
            plt.close("all")
            print(f"[{name}] plot '{plot_name}' saltato: {e}")

def plot_models_convergence(plots_dir, names, out_path):
    """Best-so-far del F1 medio in CV, tutti i modelli nello stesso grafico."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        study = joblib.load(os.path.join(plots_dir, name, "study.joblib"))
        done = [t for t in study.trials if t.state == TrialState.COMPLETE]
        ax.plot([t.number for t in done],
                np.maximum.accumulate([t.value for t in done]),
                label=name, linewidth=2)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Miglior F1 medio in CV (best-so-far)")
    ax.set_title("Convergenza del tuning Optuna")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)