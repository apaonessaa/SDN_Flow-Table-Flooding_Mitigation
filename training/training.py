import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
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

TRAINING_SET = current + '/logs/dataset_1788515678.csv'
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
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------
# Caricamento dati e split (condiviso da tutti i modelli, stesso group_id)
# --------------------------------------------------------------------------
def load_data():
    df_raw = pd.read_csv(TRAINING_SET)
    df = df_raw[(df_raw["c_used"] > 0.0)].copy()

    print(f"Numero di entry del dataset: {len(df) - 1}\n"
          f"Entry eliminate: {len(df_raw) - len(df)}")

    counts = df.groupby(["group_id", "label"]).size().unstack(fill_value=0)
    print(counts)
    print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")

    X = df[FEATURES]
    y = df[TARGET]
    groups = df["group_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


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


# --------------------------------------------------------------------------
# 1. CatBoost
# --------------------------------------------------------------------------
def train_catboost(X_train, X_test, y_train, y_test, results):
    params = {
        'silent': True,
        'depth': 8,                     # Depth of the trees
        'learning_rate': 0.1,           # Learning rate of the model
        'l2_leaf_reg': 3.0,             # L2 regularization term on weights
        'rsm': 0.8,                     # Random Selection Rate
        'iterations': 50,              # Number of boosting iterations
        'loss_function': 'Logloss',     # Loss function
        'eval_metric': 'F1',            # Evaluation metric
        'random_seed': 42,              # Random seed for reproducibility
        'auto_class_weights': 'Balanced',
        'bagging_temperature': 1.0,
        'random_strength': 2.0,
    }

    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_test, label=y_test)

    model = CatBoostClassifier(**params)

    t0 = time.time()
    model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=50,
        early_stopping_rounds=20,
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
# 2. XGBoost
# --------------------------------------------------------------------------
def train_xgboost(X_train, X_test, y_train, y_test, results):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "max_depth": 8,
        "learning_rate": 0.1,
        "reg_lambda": 3.0, 
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "n_estimators": 100,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 20,
        "n_jobs": -1,
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
# Main: training di tutti i modelli sullo stesso split + confronto finale
# --------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    results = []
    train_catboost(X_train, X_test, y_train, y_test, results)
    train_xgboost(X_train, X_test, y_train, y_test, results)

    comparison = pd.DataFrame(results).set_index("model")
    print("\n\n=== Confronto finale ===")
    print(comparison.to_string(float_format="%.4f"))

    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"\nTabella di confronto salvata in: {comparison_path}")