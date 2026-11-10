from utils import *
from m_catboost import *
from m_xgboost import *
from m_decision_tree import *
from m_random_forest import *
from feat_bert import *
from mfr_temp_bert import *
from ablation import run_featbert_ablation, run_tempbert_ablation

MODELS_DIR = root + '/models/mfr_detector'
os.makedirs(MODELS_DIR, exist_ok=True)

SUBDIR = [
    '2_10', '3_10', '4_10',
]

TRAINING_SETS = [
    root + '/feature_collector/datasets/dataset_mfr_1793987224_2_10.csv',
    root + '/feature_collector/datasets/dataset_mfr_1793987224_3_10.csv',
    root + '/feature_collector/datasets/dataset_mfr_1793987224_4_10.csv',
]

def save_best_params_csv(params: dict, path: str, dataset: str = None):
    row = {"dataset": dataset, **params} if dataset is not None else dict(params)
    pd.DataFrame([row]).to_csv(path, index=False)

if __name__ == "__main__":
    xgb_param_rows = []
    cb_param_rows = []
    dt_param_rows = []
    rf_param_rows = []
    featbert_param_rows = []
    tempbert_param_rows = []

    for TRAINING_SET, subdir in zip(TRAINING_SETS, SUBDIR):
        print(f"\n\n########## Dataset: {subdir} ##########")

        dataset_model_dir = os.path.join(MODELS_DIR, subdir)
        os.makedirs(dataset_model_dir, exist_ok=True)

        PLOTS_DIR = os.path.join(dataset_model_dir, 'tuning_plots')
        os.makedirs(PLOTS_DIR, exist_ok=True)

        X_train, X_val, X_test, y_train, y_val, y_test, groups_train, profiles_train, keys_train, keys_val, keys_test = load_data_mfr(TRAINING_SET)

        results = []

        best_cb_params = None
        best_xgb_params = None
        best_dt_params = None
        best_rf_params = None
        best_featBERT_params = None
        best_tempBERT_params = None

        print("\n\n=== Tuning Decision Tree ===")
        best_dt_params = tune_decision_tree(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)
 
        print("\n\n=== Tuning Random Forest ===")
        best_rf_params = tune_random_forest(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)
        
        print("\n\n=== Tuning XGBoost ===")
        best_xgb_params = tune_xgboost(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

        print("\n\n=== Tuning CatBoost ===")
        best_cb_params = tune_catboost(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

        print("\n\n=== Tuning FeatBERT ===")
        best_featBERT_params = tune_featBERT(X_train, y_train, groups_train, profiles_train, plots_dir=PLOTS_DIR)

        print("\n\n=== Tuning TempBERT ===")
        best_tempBERT_params = tune_tempBERT(X_train, y_train, groups_train, profiles_train, keys_train, plots_dir=PLOTS_DIR)

        save_best_params_csv(best_dt_params, os.path.join(dataset_model_dir, "best_params_decision_tree.csv"))
        save_best_params_csv(best_rf_params, os.path.join(dataset_model_dir, "best_params_random_forest.csv"))
        save_best_params_csv(best_xgb_params, os.path.join(dataset_model_dir, "best_params_xgboost.csv"))
        save_best_params_csv(best_cb_params, os.path.join(dataset_model_dir, "best_params_catboost.csv"))
        save_best_params_csv(best_featBERT_params, os.path.join(dataset_model_dir, "best_params_featbert.csv"))
        save_best_params_csv(best_tempBERT_params, os.path.join(dataset_model_dir, "best_params_tempbert.csv"))

        dt_param_rows.append({"dataset": subdir, **best_dt_params})
        rf_param_rows.append({"dataset": subdir, **best_rf_params})
        xgb_param_rows.append({"dataset": subdir, **best_xgb_params})
        cb_param_rows.append({"dataset": subdir, **best_cb_params})
        featbert_param_rows.append({"dataset": subdir, **best_featBERT_params})
        tempbert_param_rows.append({"dataset": subdir, **best_tempBERT_params})

        print("\n\n=== Training Decision Tree ===")
        train_decision_tree(X_train, X_test, y_train, y_test, results, best_dt_params, features=MFR_FEATURES, model_dir=dataset_model_dir)
 
        print("\n\n=== Training Random Forest ===")
        train_random_forest(X_train, X_test, y_train, y_test, results, best_rf_params, features=MFR_FEATURES, model_dir=dataset_model_dir)
        
        print("\n\n=== Training XGBoost ===")
        train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_xgb_params, features=MFR_FEATURES, model_dir=dataset_model_dir)

        print("\n\n=== Training CatBoost ===")
        train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_cb_params, model_dir=dataset_model_dir)

        print("\n\n=== Training FeatBERT ===")
        train_featBERT(X_train, X_val, X_test, y_train, y_val, y_test, results, best_featBERT_params, features=MFR_FEATURES, model_dir=dataset_model_dir)

        print("\n\n=== Training TempBERT ===")
        train_tempBERT(X_train, X_val, X_test, y_train, y_val, y_test,
                keys_train, keys_val, keys_test, results, best_tempBERT_params, features=MFR_FEATURES, model_dir=dataset_model_dir)
        
        print("\n\n=== Ablation FeatBERT ===")
        run_featbert_ablation(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            best_featBERT_params, out_dir=PLOTS_DIR, csv_name="featbert_ablation.csv")

        print("\n\n=== Ablation TempBERT ===")
        run_tempbert_ablation(
            MFRTempBert, prepare_seq, 
            X_train, X_val, X_test, y_train, y_val, y_test, 
            keys_train, keys_val, keys_test, 
            best_tempBERT_params, out_dir=PLOTS_DIR, csv_name="mfr_tempbert_ablation.csv")

        plot_models_convergence(
            PLOTS_DIR, ["DecisionTree", "RandomForest", "XGBoost", "CatBoost", "FeatBERT", "TempBERT"],
            os.path.join(PLOTS_DIR, "convergence_all_models.png"),
        )

        comparison = pd.DataFrame(results).set_index("model")
        print(f"\n\n=== Confronto finale ({subdir}) ===")
        print(comparison.to_string(float_format="%.4f"))

        comparison_path = os.path.join(dataset_model_dir, "model_comparison.csv")
        comparison.to_csv(comparison_path)
        print(f"\nTabella di confronto salvata in: {comparison_path}")

        best_params_path = os.path.join(dataset_model_dir, "best_params.joblib")
        joblib.dump(
            {
                "decision_tree": best_dt_params,
                "random_forest": best_rf_params,
                "xgboost": best_xgb_params,
                "catboost": best_cb_params,
                "featBERT": best_featBERT_params,
                "tempBERT": best_tempBERT_params,
            },
            best_params_path,
        )
        print(f"Migliori iperparametri salvati in: {best_params_path}")

    pd.DataFrame(dt_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_decision_tree_all_datasets.csv"))
    pd.DataFrame(rf_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_random_forest_all_datasets.csv"))
    pd.DataFrame(xgb_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_xgboost_all_datasets.csv"))
    pd.DataFrame(cb_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_catboost_all_datasets.csv"))
    pd.DataFrame(featbert_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_featbert_all_datasets.csv"))
    pd.DataFrame(tempbert_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_tempbert_all_datasets.csv"))

    print(f"\n\nCSV aggregati dei best params (tutti i dataset) salvati in: {MODELS_DIR}")
