# README

Traffic samples:

- https://wiki.wireshark.org/samplecaptures

Normal traffics:

- [SkypeIRC.cap](https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/SkypeIRC.cap)

```bash
$ python3 "$ROOT"/utils/remap_split_by_host_pcap.py "$ROOT"/testing/pcaps/SkypeIRC.cap "$ROOT"/testing/pcaps/traffic -n 4 -d 3

[*] Host Mininet generati: 4
    h1 -> IP 10.0.0.1  MAC 00:00:00:00:00:01
    h2 -> IP 10.0.0.2  MAC 00:00:00:00:00:02
    h3 -> IP 10.0.0.3  MAC 00:00:00:00:00:03
    h4 -> IP 10.0.0.4  MAC 00:00:00:00:00:04
[*] Fase 1: analisi dei top talker...
[*] Top 3 talker individuati: ['192.168.1.2', '192.168.1.1', '212.204.214.114']
[*] Coppie IP/MAC aggregatori: [('10.0.0.4', '00:00:00:00:00:04')]
[*] Fase 2: riscrittura (IP + MAC) e smistamento per host...
[*] Completato: 2263 pacchetti processati, 0 scartati (src==dst)
[*] Indirizzi originali mappati: 184
[*] Durata cattura originale: 322.7s
    10.0.0.1: offset=0.000s, 1177 pacchetti
    10.0.0.2: offset=0.270s, 355 pacchetti
    10.0.0.3: offset=0.126s, 141 pacchetti
    10.0.0.4: offset=3.344s, 574 pacchetti
```

## Commands

```bash
$ sudo ovs-ofctl dump-flows -O OpenFlow14 s1 | wc -l

$ cat controller_1788434572.log | grep -i anomaly | wc -l
```

---
