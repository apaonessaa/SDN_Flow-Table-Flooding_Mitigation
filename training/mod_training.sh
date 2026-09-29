#!/bin/bash
set -euo pipefail

DIR="$(pwd)"

# Datasets
cp "${DIR}/logs/2433/dataset_ftf.csv" /tmp/dataset_ftf.csv
cp "${DIR}/logs/2433/dataset_mfr.csv" /tmp/dataset_mfr.csv

tail -n 3348 "${DIR}/logs/05/dataset_ftf_1790077572.csv" >> /tmp/dataset_ftf.csv
tail -n 198491 "${DIR}/logs/05/dataset_mfr_1790077572.csv" >> /tmp/dataset_mfr.csv
#tail -n 2575 "${DIR}/logs/69/dataset_ftf.csv" >> /tmp/dataset_ftf.csv
#tail -n 215790 "${DIR}/logs/69/dataset_mfr.csv" >> /tmp/dataset_mfr.csv
#tail -n 3692 "${DIR}/logs/1015/dataset_ftf.csv" >> /tmp/dataset_ftf.csv
#tail -n 301987 "${DIR}/logs/1015/dataset_mfr.csv" >> /tmp/dataset_mfr.csv
#tail -n 6334 "${DIR}/logs/1623/dataset_ftf.csv" >> /tmp/dataset_ftf.csv
#tail -n 458381 "${DIR}/logs/1623/dataset_mfr.csv" >> /tmp/dataset_mfr.csv

# Network Settings
cp "${DIR}/logs/05/network_settings_1790077572.csv" /tmp/network_settings.csv
tail -n +2 "${DIR}/logs/69/network_settings_1790182132.csv" >> /tmp/network_settings.csv
tail -n +2 "${DIR}/logs/1015/network_settings_1790208259.csv" >> /tmp/network_settings.csv
tail -n +2 "${DIR}/logs/1623/network_settings_1790383866.csv" >> /tmp/network_settings.csv
tail -n +2 "${DIR}/logs/2433/network_settings_1790466105.csv" >> /tmp/network_settings.csv

# Set datasets + profile
python3 add_attack_profile_column.py

# Training 
python3 training_ftf.py
#python3 training_mfr.py

