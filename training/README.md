# README

Benign traffic PCAP:

- [CIC-IDS-2017-Dataset/Monday-WorkingHours.pcap](https://github.com/rokibulroni/CIC-IDS-2017-Dataset/tree/f2ec6ceca5be291ae607931423568d23e22b393c/PCAPs)

## Splitting by scenarios

```bash
cat <<'EOF' | bash
#!/bin/bash

PROJECT="$(pwd)" 

python3 "$PROJECT/utils/split_pcap_sessions.py" --input "$PROJECT/../Monday-WorkingHours.pcap" --outdir "$PROJECT/training/pcaps/scenarios" --segment-seconds 600
EOF
```

## Remapping & Splitting by hosts

```bash
cat <<'EOF' | bash
#!/bin/bash

PROJECT="$(pwd)"

for SCENARIO_DIR in "$PROJECT"/training/pcaps/scenarios/*.pcap; do
    SCENARIO=$(basename "$SCENARIO_DIR" .pcap)
    mkdir -p "$PROJECT/training/pcaps/traffics/$SCENARIO/"
done

for SCENARIO_DIR in "$PROJECT"/training/pcaps/traffics/*/; do
    SCENARIO=$(basename "$SCENARIO_DIR")
    python3 "$PROJECT/utils/remap_split_by_host_pcap.py" \
        "$PROJECT/training/pcaps/scenarios/$SCENARIO.pcap" \
        "$PROJECT/training/pcaps/traffics/$SCENARIO/traffic" \
        -n 14 -d 7
done
EOF
```

## Exclude group_id

```bash
$ cat dataset_ftf_1790077572.csv | awk -F',' '$21 != "6"' - > /tmp/dataset_ftf_1790077572.csv

$ cat dataset_mfr_1790077572.csv | awk -F',' '$8 != "6"' - > /tmp/dataset_ftf_1790077572.csv

$ cat dataset.csv | cut -d',' -f21 | sort -u

$ cat /tmp/dataset_mfr_1790077572.csv | cut -d',' -f8 | sort -u

$ cat dataset.csv | awk -F',' '$20 != "25"'
$ tail -n X dataset.csv 
```

## Feature importance

```bash
# to CSV format
$ cat <<EOF > /tmp/importance.txt
...
EOF

# Catboost
$ awk 'BEGIN{print "index,feature,importance"} NF==3{print $1","$2","$3}' /tmp/importance.txt

# XGBBoost
$ awk 'BEGIN{print "feature,importance"} NF==2{print $1","$2}' /tmp/importance.txt
```

---
