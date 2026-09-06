import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

TRAINING_SET = '/tmp/dataset.csv' # current + '/logs/dataset_1788515678.csv'
MODELS_DIR = parent + '/models'

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
    "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
    "c_used", "paf", "ft_psi", "durations_entropy",
    "pkts_entropy", "bytes_entropy", "mean_duration", "cvd",
    "mean_pkts", "cvp", "mean_bytes", "cvb",
]
TARGET = "label"
GROUP_COL = "group_id"
RANDOM_SEED = 42

# Numero di trial Optuna e fold per la CV interna (modifica in base al tempo disponibile)
N_TRIALS = 50
N_SPLITS = 5
CV_EARLY_STOPPING_ROUNDS = 50
FINAL_EARLY_STOPPING_ROUNDS = 50

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------
# Caricamento dati e split (condiviso da tutti i modelli, stesso group_id)
# --------------------------------------------------------------------------
def load_data():
    df_raw = pd.read_csv(TRAINING_SET)
    df = df_raw[(df_raw["c_used"] > 0.0)].copy()

    print(f"Numero di entry del dataset: {len(df) - 1}\n"
          f"Entry eliminate: {len(df_raw) - len(df)}")

    counts = df.groupby([GROUP_COL, "label"]).size().unstack(fill_value=0)
    print(counts)
    print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")

    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COL]

    # Split esterno: 80% scenari per train+tuning, 20% scenari per test finale (mai visto durante il tuning)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    n_groups_train = groups_train.nunique()
    print(f"\nScenari (group_id) nel train+tuning: {n_groups_train} -> {sorted(groups_train.unique())}")
    print(f"Scenari (group_id) nel test finale: {groups.iloc[test_idx].nunique()} -> {sorted(groups.iloc[test_idx].unique())}")

    return X_train, X_test, y_train, y_test, groups_train


def evaluate(name, y_true, y_pred, results, elapsed=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"\n=== {name} ===")
    if elapsed is not None:
        print(f"Tempo di training: {elapsed:.1f}s")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "train_time_s": elapsed,
    })


def make_group_kfold_splits(X, y, groups, n_splits):
    """Ritorna gli split, riducendo n_splits se ci sono meno gruppi disponibili."""
    n_groups = groups.nunique()
    effective_splits = min(n_splits, n_groups)
    if effective_splits < 2:
        raise ValueError(
            f"Servono almeno 2 gruppi distinti per la GroupKFold, trovati {n_groups}."
        )
    if effective_splits < n_splits:
        print(f"[!] Solo {n_groups} gruppi disponibili: uso {effective_splits}-fold invece di {n_splits}-fold.")
    gkf = GroupKFold(n_splits=effective_splits)
    return list(gkf.split(X, y, groups=groups))


# --------------------------------------------------------------------------
# 1. CatBoost - tuning Optuna con GroupKFold + training finale
# --------------------------------------------------------------------------
def tune_catboost(X_train, y_train, groups_train, n_trials=N_TRIALS, n_splits=N_SPLITS):
    splits = make_group_kfold_splits(X_train, y_train, groups_train, n_splits)

    def objective(trial):
        params = {
            'silent': True,
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
            'rsm': trial.suggest_float('rsm', 0.5, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
            'iterations': 1000,
            'loss_function': 'Logloss',
            'eval_metric': 'F1',
            'random_seed': RANDOM_SEED,
            'auto_class_weights': 'Balanced',
        }

        fold_scores = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            train_pool = Pool(data=X_tr, label=y_tr)
            val_pool = Pool(data=X_val, label=y_val)

            model = CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False,
                early_stopping_rounds=CV_EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )
            best_score = model.get_best_score()['validation']['F1']
            fold_scores.append(best_score)

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction='maximize', study_name='catboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[CatBoost] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[CatBoost] Migliori parametri: {study.best_params}")

    return study.best_params


def train_catboost(X_train, X_test, y_train, y_test, groups_train, results, best_params):
    params = {
        'silent': True,
        'iterations': 1000,
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'random_seed': RANDOM_SEED,
        'auto_class_weights': 'Balanced',
        **best_params,
    }

    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_test, label=y_test)

    model = CatBoostClassifier(**params)

    t0 = time.time()
    model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=50,
        early_stopping_rounds=FINAL_EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(test_pool)
    evaluate("CatBoost", y_test, y_pred, results, elapsed)

    print(model.get_feature_importance(prettified=True))

    out_path = os.path.join(MODELS_DIR, "catboost_model.cbm")
    model.save_model(out_path)
    print(f"Model path: {out_path}")

    return model


# --------------------------------------------------------------------------
# 2. XGBoost - tuning Optuna con GroupKFold + training finale
# --------------------------------------------------------------------------
def tune_xgboost(X_train, y_train, groups_train, n_trials=N_TRIALS, n_splits=N_SPLITS):
    splits = make_group_kfold_splits(X_train, y_train, groups_train, n_splits)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'n_estimators': 1000,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': RANDOM_SEED,
            'early_stopping_rounds': CV_EARLY_STOPPING_ROUNDS,
            'n_jobs': -1,
        }

        fold_scores = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            n_pos = (y_tr == 1).sum()
            n_neg = (y_tr == 0).sum()
            scale_pos_weight = n_neg / max(n_pos, 1)

            model = XGBClassifier(scale_pos_weight=scale_pos_weight, **params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='binary', zero_division=0))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[XGBoost] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[XGBoost] Migliori parametri: {study.best_params}")

    return study.best_params


def train_xgboost(X_train, X_test, y_train, y_test, results, best_params):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_SEED,
        'scale_pos_weight': scale_pos_weight,
        'early_stopping_rounds': FINAL_EARLY_STOPPING_ROUNDS,
        'n_jobs': -1,
        **best_params,
    }

    model = XGBClassifier(**params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    evaluate("XGBoost", y_test, y_pred, results, elapsed)

    importances = pd.Series(model.feature_importances_, index=FEATURES)
    print(importances.sort_values(ascending=False))

    out_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    model.save_model(out_path)
    print(f"Model path: {out_path}")

    return model


# --------------------------------------------------------------------------
# Main: tuning + training di tutti i modelli sullo stesso split + confronto finale
# --------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, groups_train = load_data()

    results = []

    print("\n\n=== Tuning CatBoost (Optuna + GroupKFold) ===")
    best_cb_params = tune_catboost(X_train, y_train, groups_train)

    print("\n\n=== Tuning XGBoost (Optuna + GroupKFold) ===")
    best_xgb_params = tune_xgboost(X_train, y_train, groups_train)

    print("\n\n=== Training finale CatBoost ===")
    train_catboost(X_train, X_test, y_train, y_test, groups_train, results, best_cb_params)

    print("\n\n=== Training finale XGBoost ===")
    train_xgboost(X_train, X_test, y_train, y_test, results, best_xgb_params)

    comparison = pd.DataFrame(results).set_index("model")
    print("\n\n=== Confronto finale ===")
    print(comparison.to_string(float_format="%.4f"))

    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"\nTabella di confronto salvata in: {comparison_path}")

    # Salva anche i migliori iperparametri trovati, utile per riproducibilità
    best_params_path = os.path.join(MODELS_DIR, "best_params.joblib")
    joblib.dump({"catboost": best_cb_params, "xgboost": best_xgb_params}, best_params_path)
    print(f"Migliori iperparametri salvati in: {best_params_path}")