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
# I path vengono passati come variabili d'ambiente invece di essere
# ricostruiti con __file__: con `python3 -` (codice letto da stdin, non da
# un file su disco) __file__ non esiste e solleverebbe NameError.
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

# Numero di round storici richiesti dalla finestra di TempBERT (W),
# da non confondere con la dimensione W_pktin della finestra di Packet-In
# usata in cattura feature.
WINDOWS = [4, 8, 12, 16, 20]

RULE_KEY_COL = "match_hash"
FLOW_KEY_COL = "flow_key"
SCENARIO_COL = "scenario_id"
ROUND_COL = "round"

for mfr_dataset in MFR_DATASETS:
    df = pd.read_csv(mfr_dataset)
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