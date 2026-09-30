#!/bin/bash
set -euo pipefail

DIR="$(pwd)"
NETWORK_SETTINGS="${DIR}/data/network_settings.csv"
FTF_SETTINGS="${DIR}/data/ftf_settings.csv"
DATASET_FTF="${DIR}/data/dataset_ftf.csv"
DATASET_MFR="${DIR}/data/dataset_mfr.csv"

# Network Settings
echo "Network settings @ ${NETWORK_SETTINGS}"
cp "${DIR}/logs/05/network_settings_1790077572.csv" "${NETWORK_SETTINGS}"
tail -n +2 "${DIR}/logs/69/network_settings_1790182132.csv" >> "${NETWORK_SETTINGS}"
tail -n +2 "${DIR}/logs/1015/network_settings_1790208259.csv" >> "${NETWORK_SETTINGS}"
tail -n +2 "${DIR}/logs/1623/network_settings_1790383866.csv" >> "${NETWORK_SETTINGS}"
tail -n +2 "${DIR}/logs/2431/network_settings_1790466105.csv" >> "${NETWORK_SETTINGS}"

# FTF attacks settings
echo "FTF settings @ ${FTF_SETTINGS}"
cp "${DIR}/logs/05/ftf_1790077572.csv" "${FTF_SETTINGS}"
tail -n +2 "${DIR}/logs/69/ftf_1790182132.csv" >> "${FTF_SETTINGS}"
tail -n +2 "${DIR}/logs/1015/ftf_1790208259.csv" >> "${FTF_SETTINGS}"
tail -n +2 "${DIR}/logs/1623/ftf_1790383866.csv" >> "${FTF_SETTINGS}"
tail -n +2 "${DIR}/logs/2431/ftf_1790466105.csv" >> "${FTF_SETTINGS}"

## Scenarios 0-15
cp "${DIR}/logs/05/dataset_ftf_1790077572.csv" "${DATASET_FTF}"
cp "${DIR}/logs/05/dataset_mfr_1790077572.csv" "${DATASET_MFR}"
tail -n +2 "${DIR}/logs/69/dataset_ftf.csv" >> "${DATASET_FTF}"
tail -n +2 "${DIR}/logs/69/dataset_mfr.csv" >> "${DATASET_MFR}"
tail -n +2 "${DIR}/logs/1015/dataset_ftf.csv" >> "${DATASET_FTF}"
tail -n +2 "${DIR}/logs/1015/dataset_mfr.csv" >> "${DATASET_MFR}"

## Scenarios 16-31
#tail -n +2 "${DIR}/logs/1623/dataset_ftf.csv" >> "${DATASET_FTF}"
#tail -n +2 "${DIR}/logs/1623/dataset_mfr.csv" >> "${DATASET_MFR}"
#tail -n +2 "${DIR}/logs/2431/dataset_ftf.csv" >> "${DATASET_FTF}"
#tail -n +2 "${DIR}/logs/2431/dataset_mfr.csv" >> "${DATASET_MFR}"

# Merge dataset + attack profile
python3 add_attack_profile_column.py

# Training 
#python3 training_ftf.py
python3 training_mfr.py

