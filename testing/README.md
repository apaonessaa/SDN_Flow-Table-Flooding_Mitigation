# README

Traffic samples:

- https://wiki.wireshark.org/samplecaptures

Normal traffics:

- [SkypeIRC.cap](https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/SkypeIRC.cap)

## Commands

```bash
$ sudo ovs-ofctl dump-flows -O OpenFlow14 s1 | wc -l

$ cat controller_1788434572.log | grep -i anomaly | wc -l
```

---
