import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
root = os.path.dirname(parent)
sys.path.append(root)

FEATURES = [
    "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
    "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
    "c_used", "paf", "ft_psi", "durations_entropy",
    "pkts_entropy", "bytes_entropy", "mean_duration", "cvd",
    "mean_pkts", "cvp", "mean_bytes", "cvb",
]
TARGET = "label"
SCENARIO_COL = "scenario_id"
GROUP_COL = "round"
ATTACK_PROFILE_COL = "attack_profile"
BENIGN_PROFILE_LABEL = "benigni"

# Tuning + Training
N_TRIALS=10
N_SPLITS=2
RANDOM_SEED=42
TEST_SIZE=0.3
VAL_SIZE=0.2
CV_EARLY_STOPPING_ROUNDS=50
FINAL_EARLY_STOPPING_ROUNDS=50

def load_data(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(f"Numero di entry del dataset: {len(df)}")

    counts = df.groupby([SCENARIO_COL, TARGET]).size().unstack(fill_value=0)
    print(counts)
    print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")
    
    X = df[FEATURES]
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
    """
    Others
    
    print(f"\nScenari nel train (tuning+fit): {groups_train.nunique()} -> {sorted(groups_train.unique())}")
    print(f"Scenari in validation (early stopping): {groups_trainval[val_mask].nunique()} -> {sorted(groups_trainval[val_mask].unique())}")
    print(f"Scenari nel test finale: {groups[test_mask].nunique()} -> {sorted(groups[test_mask].unique())}")

    print_profile_distribution("train", df_train, SCENARIO_COL, ATTACK_PROFILE_COL)
    print_profile_distribution("validation", df_trainval[val_mask], SCENARIO_COL, ATTACK_PROFILE_COL)
    print_profile_distribution("test", df[test_mask], SCENARIO_COL, ATTACK_PROFILE_COL)

    # Profilo per riga, allineato a X_train/y_train: serve per stratificare
    # anche i fold di cross-validation nel tuning (vedi make_group_kfold_splits).
    """
    """
    groups_val = groups_trainval[val_mask].reset_index(drop=True)
    groups_test = groups[test_mask].reset_index(drop=True)

    meta_test = df[test_mask][[SCENARIO_COL, GROUP_COL, ATTACK_PROFILE_COL]].reset_index(drop=True)
    """
    return X_train, X_val, X_test, y_train, y_val, y_test, scenarios_train, profiles_train #, groups_val, groups_test, meta_test

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
        print(f"[!] Solo {n_scenarios} gruppi disponibili: uso {effective_splits}-fold invece di {n_splits}-fold.")
    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_SEED)
    return list(sgkf.split(X, profiles, groups=scenarios))

"""Evaluate the model"""
def evaluate(name, y_true, y_pred, results, elapsed=None):#, meta=None):
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
    """
    if meta is not None:
        row.update(report_by_profile(y_true, y_pred, meta))
        row.update(detection_latency(y_true, y_pred, meta))
    """
    results.append(row)
    