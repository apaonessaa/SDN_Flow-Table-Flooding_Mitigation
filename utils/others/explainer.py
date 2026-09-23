import numpy as np
import pandas as pd
from collections import defaultdict

from catboost import CatBoostClassifier, Pool

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
root = os.path.dirname(parent)
sys.path.append(root)

### FTF

FTF_TRAINING_SET = root + '/training/logs/dataset_ftf_1790077572.csv'
FTF_OUTPUT_MODEL = root + '/models/ftf_detector/catboost_model.cbm'

feature_cols = [
    "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
    "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
    "c_used", "paf", "ft_psi", "durations_entropy", "pkts_entropy", "bytes_entropy",
    "mean_duration", "cvd", "mean_pkts", "cvp", "mean_bytes", "cvb", 
    #"group_id", "scenario_id", "label"
]

ftf_model = CatBoostClassifier()
ftf_model.load_model(FTF_OUTPUT_MODEL)

df_train = pd.read_csv(FTF_TRAINING_SET)
X_train = df_train[feature_cols]
train_pool = Pool(X_train, cat_features=None)

feature_importances = ftf_model.get_feature_importance(
    type='PredictionValuesChange',
    data=train_pool,
    verbose=True
)

print("\nFTF Detector:\n")
for name, score in sorted(zip(feature_cols, feature_importances), key=lambda x: -x[1]):
    print(f"{name}: {score:.4f}")
print()

### MFR

MFR_TRAINING_SET = root + '/training/logs/dataset_mfr_1790077572.csv'
MFR_OUTPUT_MODEL = root + '/models/malicious_fr_detector/catboost_model.cbm'

feature_cols = [
    "duration", "packet_count", "byte_count",
    "packets_per_seconds", "bytes_per_seconds", "psi",
    #"group_id", "scenario_id", "label"
]

mfr_model = CatBoostClassifier()
mfr_model.load_model(MFR_OUTPUT_MODEL)

df_train = pd.read_csv(MFR_TRAINING_SET)
X_train = df_train[feature_cols]
train_pool = Pool(X_train, cat_features=None)

feature_importances = mfr_model.get_feature_importance(
    type='PredictionValuesChange',
    data=train_pool,
    verbose=True
)

print("\nMFR Detector:\n")
for name, score in sorted(zip(feature_cols, feature_importances), key=lambda x: -x[1]):
    print(f"{name}: {score:.4f}")

