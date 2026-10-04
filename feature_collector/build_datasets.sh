#!/bin/bash
set -euo pipefail

DIR="$(pwd)"
NETWORK_SETTINGS="${DIR}/datasets/network_settings.csv"
FTF_SETTINGS="${DIR}/datasets/ftf_settings.csv"
DATASET_FTF="${DIR}/datasets/dataset_ftf.csv"
DATASET_MFR="${DIR}/datasets/dataset_mfr.csv"

# Network Settings
echo "Network settings @ ${NETWORK_SETTINGS}"
cp "${DIR}/logs/network_settings_015.csv" "${NETWORK_SETTINGS}"
#tail -n +2 "${DIR}/logs/network_settings_1015.csv" >> "${NETWORK_SETTINGS}"

# FTF attacks settings
echo "FTF settings @ ${FTF_SETTINGS}"
cp "${DIR}/logs/ftf_settings_015.csv" "${FTF_SETTINGS}"
#tail -n +2 "${DIR}/logs/ftf_1015.csv" >> "${FTF_SETTINGS}"

## Scenarios 0-15
cp "${DIR}/logs/dataset_ftf_015.csv" "${DATASET_FTF}"
cp "${DIR}/logs/dataset_mfr_015.csv" "${DATASET_MFR}"
#tail -n +2 "${DIR}/logs/dataset_ftf_1015.csv" >> "${DATASET_FTF}"
#tail -n +2 "${DIR}/logs/dataset_mfr_1015.csv" >> "${DATASET_MFR}"

# Merge dataset + attack profile
python3 add_attack_profile_column.py

## Analysis
cat <<'EOF' | python3 -
import pandas as pd
# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)

MFR_DATASET=f"{current}/datasets/dataset_mfr.csv"
WINDOWS=[3,5,8,12,16]

RULE_KEY_COL = "match_hash"
FLOW_KEY_COL = "flow_key"
SCENARIO_COL = "scenario_id"
ROUND_COL = "round"

df = pd.read_csv(MFR_DATASET)
print(f"Numero di entry del dataset: {len(df)}")

"""
Order by scenario_id+match_hash
"""
df[FLOW_KEY_COL] = df[SCENARIO_COL].astype(str) + "_" + df[RULE_KEY_COL].astype(str)
df = df.sort_values([SCENARIO_COL, FLOW_KEY_COL, ROUND_COL]).reset_index(drop=True)

obs = df.groupby(FLOW_KEY_COL).size()
print("Osservazioni per regola:")
print(obs.describe())
for w in WINDOWS:
    print(f"  regole con >= {w} osservazioni: {(obs >= w).mean():.1%}")
EOF
