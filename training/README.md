# README

Benign traffic PCAP:

- [CIC-IDS-2017-Dataset/Monday-WorkingHours.pcap](https://github.com/rokibulroni/CIC-IDS-2017-Dataset/tree/f2ec6ceca5be291ae607931423568d23e22b393c/PCAPs)

## Splitting by scenarios

```bash
$ python3 $PROJECT/utils/split_pcap_sessions.py --input Monday-WorkingHours.pcap --outdir ./pcaps/scenarios --segment-seconds 3600
```

## Remapping & Splitting by hosts

```bash
$ mdkir ./pcaps/traffics/scenario_007/traffic

$ python3 $PROJECT/utils/remap_split_pcap.py ./pcaps/scenarios/scenario_007.pcap ./pcaps/traffics/scenario_007/traffic -n 14 -d 7
```

## Exclude group_id

```bash
$ cat datasets/dataset.csv > /tmp/dataset.csv && tail -n 8952 logs/dataset_1788536029.csv | awk -F',' '$20 != "6"' - >> /tmp/dataset.csv
```

---