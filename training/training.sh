#!/bin/bash
set -euo pipefail

DIR="$(pwd)"
NETWORK_SETTINGS="${DIR}/data/network_settings.csv"
FTF_SETTINGS="${DIR}/data/ftf_settings.csv"
DATASET_FTF="${DIR}/data/dataset_ftf.csv"
DATASET_MFR="${DIR}/data/dataset_mfr.csv"

# Network Settings
echo "Network settings @ ${NETWORK_SETTINGS}"
cp "${DIR}/logs/network_settings_015.csv" "${NETWORK_SETTINGS}"
#tail -n +2 "${DIR}/logs/network_settings_1015.csv" >> "${NETWORK_SETTINGS}"

# FTF attacks settings
echo "FTF settings @ ${FTF_SETTINGS}"
cp "${DIR}/logs/ftf_settings_015.csv" "${FTF_SETTINGS}"
#tail -n +2 "${DIR}/logs/ftf_1015.csv" >> "${FTF_SETTINGS}"

## Scenarios 0-15
cp "${DIR}/logs/dataset_ftf_015_sorted.csv" "${DATASET_FTF}"
cp "${DIR}/logs/dataset_mfr_015.csv" "${DATASET_MFR}"
#tail -n +2 "${DIR}/logs/dataset_ftf_1015.csv" >> "${DATASET_FTF}"
#tail -n +2 "${DIR}/logs/dataset_mfr_1015.csv" >> "${DATASET_MFR}"

# Merge dataset + attack profile
python3 add_attack_profile_column.py

# Training 
python3 FTF/main.py
#python3 training_mfr.py

