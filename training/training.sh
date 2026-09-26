#/bin/bash
DIR="$(pwd)"

cp "${DIR}/logs/dataset_ftf.csv" /tmp/dataset_ftf.csv
cp "${DIR}/logs/dataset_mfr.csv" /tmp/dataset_mfr.csv

tail -n 3348 "${DIR}/logs/05/dataset_ftf_1790077572.csv" >> /tmp/dataset_ftf.csv
tail -n 198491 "${DIR}/logs/05/dataset_mfr_1790077572.csv" >> /tmp/dataset_mfr.csv

tail -n 2575 "${DIR}/logs/69/dataset_ftf.csv" >> /tmp/dataset_ftf.csv
tail -n 215790 "${DIR}/logs/69/dataset_mfr.csv" >> /tmp/dataset_mfr.csv

tail -n 3692 "${DIR}/logs/1015/dataset_ftf.csv" >> /tmp/dataset_ftf.csv
tail -n 301987 "${DIR}/logs/1015/dataset_mfr.csv" >> /tmp/dataset_mfr.csv

python3 training_ftf.py

python3 training_mfr.py

echo "FTF"
cat "${DIR}/../models/ftf_detector/model_comparison.csv"

echo "MFR"
cat "${DIR}/../models/malicious_fr_detector/model_comparison.csv"
