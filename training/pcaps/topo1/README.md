#README

```bash
$ python3 ../../../utils/remapping_pcap.py -n 3 -d 1 ./SkypeIRC.cap ./new_SkypeIRC.cap
```

```bash
$ python3 ../../../utils/splitting_pcap.py ./new_SkypeIRC.cap traffic 10.0.0.1 10.0.0.2 10.0.0.3
```
