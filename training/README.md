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
#### TOMOD
$ cat datasets/dataset.csv > /tmp/dataset.csv && tail -n 8952 logs/dataset_1788536029.csv | awk -F',' '$20 != "6"' - >> /tmp/dataset.csv

$ cat dataset.csv | cut -d',' -f20 | sort -u
$ cat dataset.csv | awk -F',' '$20 != "25"'
$ tail -n X dataset.csv 
```

---
