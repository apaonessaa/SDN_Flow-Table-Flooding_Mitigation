#!/usr/bin/env python3
"""
Mappa un pcap con molti host su una topologia Mininet ridotta,
riscrivendo sia IP che MAC in modo coerente con gli host reali.
"""

import sys
import ipaddress
from collections import Counter
from scapy.all import PcapReader, PcapWriter, IP, Ether

N_MININET_HOSTS = 3
N_DEDICATED_HOSTS = 2
MININET_SUBNET = ipaddress.ip_network("10.0.0.0/24")

MININET_HOST_MACS = [
    "00:00:00:00:00:01",  # h1 -> 10.0.0.1
    "00:00:00:00:00:02",  # h2 -> 10.0.0.2
    "00:00:00:00:00:03",  # h3 -> 10.0.0.3
    #"00:00:00:00:00:04",  # h4 -> 10.0.0.4
    #"00:00:00:00:00:05",  # h5 -> 10.0.0.5
    #"00:00:00:00:00:06",  # h6 -> 10.0.0.6
]

def analyze_top_talkers(pcap_path, top_n):
    counter = Counter()
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if pkt.haslayer(IP):
                counter[pkt[IP].src] += 1
                counter[pkt[IP].dst] += 1
    return [ip for ip, _ in counter.most_common(top_n)]


def build_ip_mac_mapping(top_talkers):
    """Costruisce la mappa: IP originale -> (IP mininet, MAC mininet).
    L'indice usato per l'IP e' lo stesso usato per il MAC, cosi'
    restano sempre coerenti tra loro (stesso host = stessa coppia)."""
    mapping = {}

    for i, ip in enumerate(top_talkers):
        new_ip = str(MININET_SUBNET.network_address + i + 1)
        new_mac = MININET_HOST_MACS[i]
        mapping[ip] = (new_ip, new_mac)

    dedicated_count = len(top_talkers)
    aggregator_indices = list(range(dedicated_count, N_MININET_HOSTS))
    aggregator_pairs = [
        (str(MININET_SUBNET.network_address + idx + 1), MININET_HOST_MACS[idx])
        for idx in aggregator_indices
    ]

    return mapping, aggregator_pairs


def remap(original_ip, mapping, aggregator_pairs):
    if original_ip in mapping:
        return mapping[original_ip]

    bucket = hash(original_ip) % len(aggregator_pairs)
    new_ip, new_mac = aggregator_pairs[bucket]
    mapping[original_ip] = (new_ip, new_mac)
    return new_ip, new_mac


def rewrite_pcap(infile, outfile):
    print("[*] Fase 1: analisi dei top talker...")
    top_talkers = analyze_top_talkers(infile, N_DEDICATED_HOSTS)
    print(f"[*] Top {N_DEDICATED_HOSTS} talker individuati: {top_talkers}")

    mapping, aggregator_pairs = build_ip_mac_mapping(top_talkers)
    print(f"[*] Coppie IP/MAC aggregatori: {aggregator_pairs}")

    print("[*] Fase 2: riscrittura del pcap (IP + MAC)...")
    packet_count = 0
    dropped_count = 0

    with PcapReader(infile) as reader, PcapWriter(outfile, sync=True) as writer:
        for pkt in reader:
            packet_count += 1

            if not (pkt.haslayer(IP) and pkt.haslayer(Ether)):
                continue

            new_src_ip, new_src_mac = remap(pkt[IP].src, mapping, aggregator_pairs)
            new_dst_ip, new_dst_mac = remap(pkt[IP].dst, mapping, aggregator_pairs)

            if new_src_ip == new_dst_ip:
                dropped_count += 1
                continue

            # --- riscrittura IP ---
            pkt[IP].src = new_src_ip
            pkt[IP].dst = new_dst_ip
            del pkt[IP].chksum
            if pkt.haslayer("TCP"):
                del pkt["TCP"].chksum
            elif pkt.haslayer("UDP"):
                del pkt["UDP"].chksum

            # --- riscrittura MAC (il pezzo che mancava) ---
            pkt[Ether].src = new_src_mac
            pkt[Ether].dst = new_dst_mac

            writer.write(pkt)

    print(f"[*] Completato: {packet_count} pacchetti processati, {dropped_count} scartati (src==dst)")
    print(f"[*] Indirizzi univoci mappati: {len(mapping)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <input.pcap> <output.pcap>")
        sys.exit(1)
    rewrite_pcap(sys.argv[1], sys.argv[2])