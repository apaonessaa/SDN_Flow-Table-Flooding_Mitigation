from utils import *
from m_catboost import *
from m_xgboost import *
from feat_bert import *
from temp_bert import *

# FeatBERT & TempBERT Ablation
from feat_bert_ablation import run_featbert_ablation
from temp_bert_ablation import run_tempbert_ablation

MODELS_DIR = root + '/models/ftf_detector'
PLOTS_DIR = os.path.join(MODELS_DIR, 'tuning_plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAINING_SET = root + '/feature_collector/datasets/dataset_ftf.csv'

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, profiles_train, groups_train, groups_val, groups_test = load_data_ftf(TRAINING_SET)

    results = []

    best_cb_params = None
    best_xgb_params = None
    best_featBERT_params = None
    best_tempBERT_params = None

    print("\n\n=== Tuning XGBoost===")
    best_xgb_params = tune_xgboost(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

    print("\n\n=== Tuning CatBoost ===")
    best_cb_params = tune_catboost(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

    print("\n\n=== Tuning FeatBERT ===")
    best_featBERT_params = tune_featBERT(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

    print("\n\n=== Ablation FeatBERT (M0-MLP vs M1-Tokenizer+MeanPool vs M2-FeatBERT) ===")
    ablation_df = run_featbert_ablation(
        X_train, X_val, X_test, y_train, y_val, y_test,
        best_featBERT_params, features=FTF_FEATURES, out_dir=PLOTS_DIR,
    )

    print("\n\n=== Tuning TempBERT ===")
    best_tempBERT_params = tune_tempBERT(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

    print("\n\n=== Ablation TempBERT (M0-MLP vs M1-RoundToken+MeanPool vs M2-TempBERT) ===")
    tempbert_ablation_df = run_tempbert_ablation(
        X_train, X_val, X_test, y_train, y_val, y_test,
        groups_train, groups_val, groups_test,
        best_tempBERT_params, features=FTF_FEATURES, out_dir=PLOTS_DIR,
    )

    plot_models_convergence(
        PLOTS_DIR, ["XGBoost", "CatBoost", "FeatBERT", "TempBERT"],
        os.path.join(PLOTS_DIR, "convergence_all_models.png"),
    )

    print("\n\n=== Training finale XGBoost ===")
    train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_xgb_params, features=FTF_FEATURES, model_dir=MODELS_DIR)

    print("\n\n=== Training finale CatBoost ===")
    train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_cb_params, model_dir=MODELS_DIR)

    print("\n\n=== Training finale FeatBERT ===")
    train_featBERT(X_train, X_val, X_test, y_train, y_val, y_test, results, best_featBERT_params, features=FTF_FEATURES, model_dir=MODELS_DIR)

    print("\n\n=== Training finale TempBERT ===")
    train_tempBERT(X_train, X_val, X_test, y_train, y_val, y_test,
            groups_train, groups_val, groups_test, results, best_tempBERT_params, features=FTF_FEATURES, model_dir=MODELS_DIR)

    comparison = pd.DataFrame(results).set_index("model")
    print("\n\n=== Confronto finale ===")
    print(comparison.to_string(float_format="%.4f"))

    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"\nTabella di confronto salvata in: {comparison_path}")

    best_params_path = os.path.join(MODELS_DIR, "best_params.joblib")
    joblib.dump(
        {
            "xgboost": best_xgb_params,
            "catboost": best_cb_params,
            "featBERT": best_featBERT_params,
            "tempBERT": best_tempBERT_params,
        },
        best_params_path,
    )
    print(f"Migliori iperparametri salvati in: {best_params_path}")