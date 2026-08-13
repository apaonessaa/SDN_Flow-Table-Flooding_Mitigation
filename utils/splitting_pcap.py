#!/usr/bin/env python3
"""
Divide un pcap per host sorgente, registrando anche l'offset temporale
di ciascun host rispetto al timestamp assoluto piu' antico del file,
necessario per riallineare correttamente i replay multipli.
"""

import sys
import json
from scapy.all import PcapReader, PcapWriter, IP


def split_pcap_with_timing(infile, output_prefix, host_ips):
    writers = {
        ip: PcapWriter(f"{output_prefix}_{ip.replace('.', '_')}.pcap", sync=True)
        for ip in host_ips
    }

    global_min_time = None
    first_packet_time = {ip: None for ip in host_ips}
    per_host_count = {ip: 0 for ip in host_ips}

    with PcapReader(infile) as reader:
        for pkt in reader:
            ts = float(pkt.time)   # timestamp assoluto (epoch) del pacchetto

            if global_min_time is None or ts < global_min_time:
                global_min_time = ts

            if pkt.haslayer(IP):
                src = pkt[IP].src
                if src in writers:
                    writers[src].write(pkt)
                    per_host_count[src] += 1
                    if first_packet_time[src] is None or ts < first_packet_time[src]:
                        first_packet_time[src] = ts

    for w in writers.values():
        w.close()

    # offset di ogni host rispetto all'inizio assoluto del pcap
    offsets = {
        ip: (first_packet_time[ip] - global_min_time) if first_packet_time[ip] is not None else 0.0
        for ip in host_ips
    }

    metadata = {"offsets": offsets, "packet_counts": per_host_count}
    with open(f"{output_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[*] Timestamp assoluto minimo globale: {global_min_time}")
    for ip in host_ips:
        print(f"    {ip}: offset={offsets[ip]:.3f}s, {per_host_count[ip]} pacchetti")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Uso: {sys.argv[0]} <input.pcap> <output_prefix> <ip1> [ip2] ...")
        sys.exit(1)

    infile = sys.argv[1]
    output_prefix = sys.argv[2]
    host_ips = sys.argv[3:]

    split_pcap_with_timing(infile, output_prefix, host_ips)