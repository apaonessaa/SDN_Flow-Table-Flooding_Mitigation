from utils import *
from m_catboost import *
from m_xgboost import *
from m_decision_tree import *
from m_random_forest import *
from feat_bert import *
from temp_bert import *

from mitigation.tempbert_arch import TempBert
from ablation import run_featbert_ablation, run_tempbert_ablation

MODELS_DIR = root + '/models/ftf_detector/TempBERT_pi'
os.makedirs(MODELS_DIR, exist_ok=True)

SUBDIR = [
    '2_100_top4', '2_100_top8', '2_100_top12',
]

TRAINING_SETS = [
    '/tmp/dataset_ftf_2_100_top4.csv',
    '/tmp/dataset_ftf_2_100_top8.csv',
    '/tmp/dataset_ftf_2_100_top12.csv'
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

    BEST_FEATURES_TEMPBERT = [
        'src_ip_entropy','src_mac_entropy','durations_entropy',
        'ft_psi','mean_bytes','cvp','payload_entropy_cv',
        'payload_entropy_mean','pkts_entropy','cvb',
        'payload_size_entropy','dst_mac_entropy'
    ]
    N_TOP_FEATURES = [4, 8, len(BEST_FEATURES_TEMPBERT)]
    
    for TRAINING_SET, subdir, n_top_features in zip(TRAINING_SETS, SUBDIR, N_TOP_FEATURES):
        print(f"\n\n########## Dataset: {subdir} ##########")

        dataset_model_dir = os.path.join(MODELS_DIR, subdir)
        os.makedirs(dataset_model_dir, exist_ok=True)

        PLOTS_DIR = os.path.join(dataset_model_dir, 'tuning_plots')
        os.makedirs(PLOTS_DIR, exist_ok=True)

        print(BEST_FEATURES_TEMPBERT[:n_top_features])

        X_train, X_val, X_test, y_train, y_val, y_test, \
            profiles_train, groups_train, groups_val, groups_test = \
                load_data_ftf(TRAINING_SET, ftf_features=BEST_FEATURES_TEMPBERT[:n_top_features])
        
        results = []

        best_tempBERT_params = None

        print("\n\n=== Tuning TempBERT ===")
        best_tempBERT_params = tune_tempBERT(
            X_train, y_train, groups_train, profiles_train, 
            plots_dir=PLOTS_DIR, trials_csv=os.path.join(dataset_model_dir, "trials_tempbert.csv"))

        save_best_params_csv(best_tempBERT_params, os.path.join(dataset_model_dir, "best_params_tempbert.csv"))

        tempbert_param_rows.append({"dataset": subdir, **best_tempBERT_params})
        
        print("\n\n=== Training TempBERT ===")
        train_tempBERT(X_train, X_val, X_test, y_train, y_val, y_test,
                groups_train, groups_val, groups_test, results, best_tempBERT_params, 
                features=BEST_FEATURES_TEMPBERT[:n_top_features], model_dir=dataset_model_dir, label=subdir)

        comparison = pd.DataFrame(results).set_index("model")
        print(f"\n\n=== Confronto finale ({subdir}) ===")
        print(comparison.to_string(float_format="%.4f"))

        comparison_path = os.path.join(dataset_model_dir, "model_comparison.csv")
        comparison.to_csv(comparison_path)
        print(f"\nTabella di confronto salvata in: {comparison_path}")

        best_params_path = os.path.join(dataset_model_dir, "best_params.joblib")
        joblib.dump(
            {
                "tempBERT": best_tempBERT_params,
            },
            best_params_path,
        )
        print(f"Migliori iperparametri salvati in: {best_params_path}")
    pd.DataFrame(tempbert_param_rows).set_index("dataset").to_csv(
        os.path.join(MODELS_DIR, "best_params_tempbert_all_datasets.csv"))

    print(f"\n\nCSV aggregati dei best params (tutti i dataset) salvati in: {MODELS_DIR}")

