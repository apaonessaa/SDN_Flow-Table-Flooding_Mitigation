# README

## Training

```bash
# Build the datasets
(feature_collector) $ ./build_datasets.sh

# Training the FTF models
(training) $ python3 ftf_main.py

# Training the MFR models
(training) $ python3 mfr_main.py
```

## Packet count

```bash
$ cat /tmp/scenarios_label.txt | awk 'BEGIN{print "index,benign,malicious"} NF==3{print $1","$2","$3}'
```
