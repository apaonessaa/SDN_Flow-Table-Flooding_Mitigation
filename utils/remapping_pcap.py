#!/usr/bin/env python3
"""
Mappa un pcap con molti host su una topologia Mininet ridotta,
riscrivendo sia IP che MAC in modo coerente con gli host reali.
"""
import sys
import argparse
import ipaddress
from collections import Counter
from scapy.all import PcapReader, PcapWriter, IP, Ether

MININET_SUBNET = ipaddress.ip_network("10.0.0.0/24")

def generate_host_macs(n):
    """Genera n MAC sequenziali nella forma 00:00:00:00:00:XX (h1..hn -> 10.0.0.1..10.0.0.n)."""
    return [f"00:00:00:00:00:{i:02x}" for i in range(1, n + 1)]

def analyze_top_talkers(pcap_path, top_n):
    counter = Counter()
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if pkt.haslayer(IP):
                counter[pkt[IP].src] += 1
                counter[pkt[IP].dst] += 1
    return [ip for ip, _ in counter.most_common(top_n)]

def build_ip_mac_mapping(top_talkers, n_mininet_hosts, mininet_host_macs):
    """Costruisce la mappa: IP originale -> (IP mininet, MAC mininet).
    L'indice usato per l'IP e' lo stesso usato per il MAC, cosi'
    restano sempre coerenti tra loro (stesso host = stessa coppia)."""
    mapping = {}
    for i, ip in enumerate(top_talkers):
        new_ip = str(MININET_SUBNET.network_address + i + 1)
        new_mac = mininet_host_macs[i]
        mapping[ip] = (new_ip, new_mac)
    dedicated_count = len(top_talkers)
    aggregator_indices = list(range(dedicated_count, n_mininet_hosts))
    aggregator_pairs = [
        (str(MININET_SUBNET.network_address + idx + 1), mininet_host_macs[idx])
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

def rewrite_pcap(infile, outfile, n_mininet_hosts, n_dedicated_hosts, mininet_host_macs):
    print("[*] Fase 1: analisi dei top talker...")
    top_talkers = analyze_top_talkers(infile, n_dedicated_hosts)
    print(f"[*] Top {n_dedicated_hosts} talker individuati: {top_talkers}")

    mapping, aggregator_pairs = build_ip_mac_mapping(
        top_talkers, n_mininet_hosts, mininet_host_macs
    )
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rimappa IP e MAC di un pcap su una topologia Mininet ridotta."
    )
    parser.add_argument("infile", help="File pcap di input")
    parser.add_argument("outfile", help="File pcap di output")
    parser.add_argument(
        "-n", "--n-mininet-hosts",
        type=int,
        default=3,
        help="Numero totale di host Mininet nella topologia (default: 3)",
    )
    parser.add_argument(
        "-d", "--n-dedicated-hosts",
        type=int,
        default=2,
        help="Numero di top talker a cui riservare un host dedicato (default: 2)",
    )
    return parser.parse_args()

def validate_args(args):
    if args.n_mininet_hosts < 1:
        print("[!] Errore: --n-mininet-hosts deve essere >= 1.")
        sys.exit(1)
    if args.n_mininet_hosts > 254:
        print("[!] Errore: --n-mininet-hosts non puo' superare 254 (subnet /24).")
        sys.exit(1)
    if args.n_dedicated_hosts > args.n_mininet_hosts:
        print(
            "[!] Errore: --n-dedicated-hosts non puo' essere maggiore di "
            "--n-mininet-hosts."
        )
        sys.exit(1)

"""
python3 remap_pcap.py input.pcap output.pcap -n 3 -d 2
"""
if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    mininet_host_macs = generate_host_macs(args.n_mininet_hosts)
    print(f"[*] Host Mininet generati: {args.n_mininet_hosts}")
    for i, mac in enumerate(mininet_host_macs, start=1):
        print(f"    h{i} -> IP {MININET_SUBNET.network_address + i}  MAC {mac}")

    rewrite_pcap(
        args.infile,
        args.outfile,
        args.n_mininet_hosts,
        args.n_dedicated_hosts,
        mininet_host_macs,
    )