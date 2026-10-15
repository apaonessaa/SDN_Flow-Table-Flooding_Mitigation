import pandas as pd
import joblib

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import *
from m_catboost import *
from m_xgboost import *
from m_seqbert import *
from feat_bert import *

MODELS_DIR = root + '/models/ftf_detector'
os.makedirs(MODELS_DIR, exist_ok=True)

TRAINING_SET = root + '/feature_collector/datasets/dataset_ftf.csv'

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, profiles_train, groups_train, groups_val, groups_test = load_data_ftf(TRAINING_SET)

    #d = X_train.describe(percentiles=[.01, .5, .99]).T
    #d["skew"] = X_train.skew()
    #print(d[["std","1%","50%","99%","min","max","skew"]])

    results = []

    best_cb_params = None
    best_xgb_params = None
    best_seq_params = None
    best_bert_params = None

    print("\n\n=== Tuning CatBoost (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    #best_cb_params = tune_catboost(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning XGBoost (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    best_xgb_params = tune_xgboost(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning SeqBERT ===")
    #best_seq_params = tune_seq(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning BERT ===")
    best_bert_params = tune_bert(X_train, y_train, groups_train, profiles_train, model_dir=MODELS_DIR)

    print("\n\n=== Training finale CatBoost ===")
    #train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_cb_params, model_dir=MODELS_DIR)

    print("\n\n=== Training finale XGBoost ===")
    train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_xgb_params, features=FTF_FEATURES, model_dir=MODELS_DIR)

    print("\n\n=== Training finale SeqBERT ===")
    #train_seq(X_train, X_val, X_test, y_train, y_val, y_test,
    #        groups_train, groups_val, groups_test, results, best_seq_params, features=FTF_FEATURES, model_dir=MODELS_DIR)

    print("\n\n=== Training finale BERT ===")
    train_bert(X_train, X_val, X_test, y_train, y_val, y_test, results, best_bert_params, model_dir=MODELS_DIR)

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
            "seq": best_seq_params
        },
        best_params_path,
    )
    print(f"Migliori iperparametri salvati in: {best_params_path}")