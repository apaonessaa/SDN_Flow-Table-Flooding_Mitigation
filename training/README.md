# README

```bash
$ cd pcaps

# Merge & Remove duplicate
$ mergecap -w merged_raw.pcap left.pcap right.pcap bottom.pcap
$ editcap -D 1000 merged_raw.pcap raw_traffic.pcap
1829987 packets seen, 955072 packets skipped with duplicate window of 1000 packets.

# Remapping
$ python3 ../../utils/remapping_pcap.py /home/ubuntu/Desktop/progetto/training/pcaps/raw_traffic.pcap /home/ubuntu/Desktop/progetto/training/pcaps/traffic.pcap

# Splitting
$ python3 ../../utils/splitting_pcap.py /home/ubuntu/Desktop/progetto/training/pcaps/traffic.pcap traffic 10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4 10.0.0.5 10.0.0.6
```

Dataset:

- https://github.com/westermo/network-traffic-dataset/tree/main/data/reduced

```bash
$ python3 ../../utils/remapping_pcap.py /home/ubuntu/Desktop/progetto/training/pcaps/raw_traffic.pcap /home/ubuntu/Desktop/progetto/training/pcaps/traffic.pcap && python3 ../../utils/splitting_pcap.py /home/ubuntu/Desktop/progetto/training/pcaps/traffic.pcap traffic 10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4 10.0.0.5 10.0.0.6
[*] Fase 1: analisi dei top talker...
[*] Top 4 talker individuati: ['198.18.134.11', '198.18.134.12', '198.18.134.15', '198.18.134.99']
[*] Coppie IP/MAC aggregatori: [('10.0.0.5', '00:00:00:00:00:05'), ('10.0.0.6', '00:00:00:00:00:06')]
[*] Fase 2: riscrittura del pcap (IP + MAC)...
[*] Completato: 874915 pacchetti processati, 442 scartati (src==dst)
[*] Indirizzi univoci mappati: 38
[*] Timestamp assoluto minimo globale: 1678440574.08554
    10.0.0.1: offset=0.001s, 275432 pacchetti
    10.0.0.2: offset=0.000s, 189730 pacchetti
    10.0.0.3: offset=0.331s, 257918 pacchetti
    10.0.0.4: offset=0.880s, 121346 pacchetti
    10.0.0.5: offset=0.882s, 15438 pacchetti
    10.0.0.6: offset=6.461s, 9080 pacchetti
```

## Balanced Dataset

```bash
$ wc -l *
   15316 benign.csv
    3298 malicious.csv

$ cp benign.csv raw_training_set.csv

$ cat malicious.csv | tail -n 7074 >> raw_training_set.csv

# Elimina pattern 0
$ cat raw_training_set.csv | grep -v -e '^0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,*' > training_set.csv

$ wc -l *training_set.csv 
   18613 raw_training_set.csv
   18541 training_set.csv
```

```bash
$ python3 training_model.py
```

## Data Leakage

- https://northhavenanalytics.com/definitive-guide-data-leakage-machine-learning-prevention/


---
