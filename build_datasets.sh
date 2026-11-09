#!/bin/bash
set -euo pipefail

TIMESTAMP="1793987224"

ROOT="$(pwd)"
SRC="${ROOT}/feature_collector/datasets/backup"
OUT="${ROOT}/feature_collector/datasets"

cp -t "${OUT}" "${SRC}/network_settings_${TIMESTAMP}.csv" "${SRC}/ftf_${TIMESTAMP}.csv"

NETWORK_SETTINGS="${OUT}/network_settings_${TIMESTAMP}.csv"

# Add the attack profile
cd "${SRC}"
python3 "${ROOT}/utils/add_attack_profile_column.py" --output-dir "${OUT}" --network-settings "${NETWORK_SETTINGS}" --no-backup dataset_ftf_*_*.csv
python3 "${ROOT}/utils/add_attack_profile_column.py" --output-dir "${OUT}" --network-settings "${NETWORK_SETTINGS}" --no-backup dataset_mfr_*_*.csv
cd -

## Analysis
TIMESTAMP="${TIMESTAMP}" MFR_DATASET_DIR="${OUT}" python3 - <<'EOF'
import os
import pandas as pd

mfr_dir = os.environ["MFR_DATASET_DIR"]
timestamp = os.environ["TIMESTAMP"]

MFR_DATASETS = [
    f"{mfr_dir}/dataset_mfr_{timestamp}_2_10.csv",
    f"{mfr_dir}/dataset_mfr_{timestamp}_3_10.csv",
    f"{mfr_dir}/dataset_mfr_{timestamp}_4_10.csv",
]

WINDOWS = [4, 8, 12, 16, 20, 24, 28, 32]

RULE_KEY_COL = "match_hash"
FLOW_KEY_COL = "flow_key"
SCENARIO_COL = "scenario_id"
ROUND_COL = "round"

for mfr_dataset in MFR_DATASETS:
    df = pd.read_csv(mfr_dataset)
    print(f"\n=== {os.path.basename(mfr_dataset)} ===")
    print(f"Numero di entry del dataset: {len(df)}")

    """
    Order by scenario_id+match_hash
    """
    df[FLOW_KEY_COL] = df[SCENARIO_COL].astype(str) + "_" + df[RULE_KEY_COL].astype(str)
    df = df.sort_values([SCENARIO_COL, FLOW_KEY_COL, ROUND_COL]).reset_index(drop=True)

    obs = df.groupby(FLOW_KEY_COL).size()
    n_rules = len(obs)
    print(f"Numero di regole: {n_rules}")
    print("Osservazioni per regola:")
    print(obs.describe())

    for w in WINDOWS:
        short = (obs <= w - 1).sum()   # < W: la finestra non si riempie (padding)
        print(f"  W={w:>2}: regole con <= {w-1:>2} oss. (finestra incompleta): "
              f"{short / n_rules:.1%} ({short})"
              f" | => almeno {w} oss.: {1 - short / n_rules:.1%}")
EOF