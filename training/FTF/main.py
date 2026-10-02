import pandas as pd
import joblib

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
root = os.path.dirname(parent)
sys.path.append(root)

from ftf_utils import *
from ftf_catboost import *
from ftf_xgboost import *

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, groups_train, profiles_train = load_data()

    results = []

    best_cb_params = None
    best_xgb_params = None
    best_bert_params = None
    best_seq_params = None

    print("\n\n=== Tuning CatBoost (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    best_cb_params = tune_catboost(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning XGBoost (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    best_xgb_params = tune_xgboost(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning HF-BERT (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    #best_bert_params = tune_hf_bert(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning SeqBERT ===")
    #best_seq_params = tune_seq(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Training finale CatBoost ===")
    train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_cb_params)

    print("\n\n=== Training finale XGBoost ===")
    train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_xgb_params)

    print("\n\n=== Training finale HF-BERT ===")
    #train_hf_bert(X_train, X_val, X_test, y_train, y_val, y_test, results, best_bert_params, meta_test)

    print("\n\n=== Training finale SeqBERT ===")
    #train_seq(X_train, X_val, X_test, y_train, y_val, y_test,
    #        groups_train, groups_val, groups_test, results, best_seq_params, meta_test)

    comparison = pd.DataFrame(results).set_index("model")
    print("\n\n=== Confronto finale ===")
    print(comparison.to_string(float_format="%.4f"))

    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"\nTabella di confronto salvata in: {comparison_path}")

    best_params_path = os.path.join(MODELS_DIR, "best_params.joblib")
    joblib.dump(
        {
            "catboost": best_cb_params, 
            "xgboost": best_xgb_params, 
            "hf_bert": best_bert_params, 
            "seq": best_seq_params
        },
        best_params_path,
    )
    print(f"Migliori iperparametri salvati in: {best_params_path}")