#!/usr/bin/env python3
import sys
import json
import socket
import struct
import argparse
import ipaddress
from collections import Counter

import dpkt

MININET_SUBNET = ipaddress.ip_network("10.0.0.0/24")

# --------------------------------------------------------------------------
# Utility indirizzi
# --------------------------------------------------------------------------

def generate_host_macs(n):
    """Genera n MAC nella forma 00:00:00:00:00:XX (h1..hn -> 10.0.0.1..10.0.0.n)."""
    return [f"00:00:00:00:00:{i:02x}" for i in range(1, n + 1)]


def mac_str_to_bytes(mac_str):
    return bytes.fromhex(mac_str.replace(":", ""))


def ip_str_to_bytes(ip_str):
    return socket.inet_aton(ip_str)


# --------------------------------------------------------------------------
# Fase 1: individuazione top talker (singola passata leggera con dpkt)
# --------------------------------------------------------------------------

def analyze_top_talkers(pcap_path, top_n):
    counter = Counter()
    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for _, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            counter[socket.inet_ntoa(ip.src)] += 1
            counter[socket.inet_ntoa(ip.dst)] += 1
    return [ip for ip, _ in counter.most_common(top_n)]


def build_ip_mac_mapping(top_talkers, n_mininet_hosts, mininet_host_macs):
    """IP originale -> (IP mininet, MAC mininet bytes). Stesso indice per
    IP e MAC cosi' restano sempre coerenti tra loro (stesso host = stessa coppia)."""
    mapping = {}
    for i, ip in enumerate(top_talkers):
        new_ip = str(MININET_SUBNET.network_address + i + 1)
        new_mac = mac_str_to_bytes(mininet_host_macs[i])
        mapping[ip] = (new_ip, new_mac)

    dedicated_count = len(top_talkers)
    aggregator_indices = list(range(dedicated_count, n_mininet_hosts))
    aggregator_pairs = [
        (str(MININET_SUBNET.network_address + idx + 1), mac_str_to_bytes(mininet_host_macs[idx]))
        for idx in aggregator_indices
    ]
    return mapping, aggregator_pairs


def remap(original_ip, mapping, aggregator_pairs):
    if original_ip in mapping:
        return mapping[original_ip]
    if not aggregator_pairs:
        # tutti gli host sono dedicati e non c'e' spazio per un aggregatore:
        # riusa deterministicamente uno degli host dedicati
        bucket = hash(original_ip) % len(mapping)
        new_ip, new_mac = list(mapping.values())[bucket]
    else:
        bucket = hash(original_ip) % len(aggregator_pairs)
        new_ip, new_mac = aggregator_pairs[bucket]
    mapping[original_ip] = (new_ip, new_mac)
    return new_ip, new_mac


# --------------------------------------------------------------------------
# Fase 2: riscrittura + smistamento in un'unica passata
# --------------------------------------------------------------------------

def rewrite_and_split(infile, output_prefix, n_mininet_hosts, n_dedicated_hosts, mininet_host_macs):
    print("[*] Fase 1: analisi dei top talker...")
    top_talkers = analyze_top_talkers(infile, n_dedicated_hosts)
    print(f"[*] Top {n_dedicated_hosts} talker individuati: {top_talkers}")

    mapping, aggregator_pairs = build_ip_mac_mapping(
        top_talkers, n_mininet_hosts, mininet_host_macs
    )
    print(f"[*] Coppie IP/MAC aggregatori: {[(ip, mac.hex(':')) for ip, mac in aggregator_pairs]}")

    # un writer dpkt per ciascun host mininet finale (10.0.0.1 .. 10.0.0.n)
    host_ips = [str(MININET_SUBNET.network_address + i) for i in range(1, n_mininet_hosts + 1)]
    out_files = {}
    writers = {}
    for ip in host_ips:
        fname = f"{output_prefix}_{ip.replace('.', '_')}.pcap"
        fh = open(fname, "wb")
        out_files[ip] = fh
        writers[ip] = dpkt.pcap.Writer(fh)

    global_min_time = None
    global_max_time = None
    first_packet_time = {ip: None for ip in host_ips}
    per_host_count = {ip: 0 for ip in host_ips}
    packet_count = 0
    dropped_count = 0

    print("[*] Fase 2: riscrittura (IP + MAC) e smistamento per host...")
    with open(infile, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            packet_count += 1

            if global_min_time is None or ts < global_min_time:
                global_min_time = ts
            if global_max_time is None or ts > global_max_time:
                global_max_time = ts

            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            ip = eth.data
            src_str = socket.inet_ntoa(ip.src)
            dst_str = socket.inet_ntoa(ip.dst)

            new_src_ip, new_src_mac = remap(src_str, mapping, aggregator_pairs)
            new_dst_ip, new_dst_mac = remap(dst_str, mapping, aggregator_pairs)

            if new_src_ip == new_dst_ip:
                dropped_count += 1
                continue

            # --- riscrittura IP ---
            ip.src = ip_str_to_bytes(new_src_ip)
            ip.dst = ip_str_to_bytes(new_dst_ip)
            ip.sum = 0  # forza dpkt a ricalcolare il checksum IP in pack()

            if isinstance(ip.data, dpkt.tcp.TCP):
                ip.data.sum = 0  # ricalcolo checksum TCP
            elif isinstance(ip.data, dpkt.udp.UDP):
                ip.data.sum = 0  # ricalcolo checksum UDP

            # --- riscrittura MAC ---
            eth.src = new_src_mac
            eth.dst = new_dst_mac

            raw = bytes(eth)  # qui dpkt ricalcola i checksum azzerati

            writer = writers.get(new_src_ip)
            if writer is not None:
                writer.writepkt(raw, ts=ts)
                per_host_count[new_src_ip] += 1
                if first_packet_time[new_src_ip] is None or ts < first_packet_time[new_src_ip]:
                    first_packet_time[new_src_ip] = ts

    for fh in out_files.values():
        fh.close()

    original_capture_duration = (global_max_time - global_min_time) if global_min_time is not None else 0.0
    offsets = {
        ip: (first_packet_time[ip] - global_min_time) if first_packet_time[ip] is not None else 0.0
        for ip in host_ips
    }

    metadata = {
        "ip_mac_mapping": {
            orig_ip: {"new_ip": new_ip, "new_mac": new_mac.hex(":")}
            for orig_ip, (new_ip, new_mac) in mapping.items()
        },
        "offsets": offsets,
        "packet_counts": per_host_count,
        "original_capture_duration": original_capture_duration,
        "packets_processed": packet_count,
        "packets_dropped_src_eq_dst": dropped_count,
    }
    with open(f"{output_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[*] Completato: {packet_count} pacchetti processati, {dropped_count} scartati (src==dst)")
    print(f"[*] Indirizzi originali mappati: {len(mapping)}")
    print(f"[*] Durata cattura originale: {original_capture_duration:.1f}s")
    for ip in host_ips:
        print(f"    {ip}: offset={offsets[ip]:.3f}s, {per_host_count[ip]} pacchetti")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rimappa IP/MAC di un pcap su una topologia Mininet ridotta "
                     "e smista i pacchetti in un pcap per host, con metadata di timing."
    )
    parser.add_argument("infile", help="File pcap di input")
    parser.add_argument("output_prefix", help="Prefisso per i file pcap/JSON di output")
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


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    mininet_host_macs = generate_host_macs(args.n_mininet_hosts)
    print(f"[*] Host Mininet generati: {args.n_mininet_hosts}")
    for i, mac in enumerate(mininet_host_macs, start=1):
        print(f"    h{i} -> IP {MININET_SUBNET.network_address + i}  MAC {mac}")

    rewrite_and_split(
        args.infile,
        args.output_prefix,
        args.n_mininet_hosts,
        args.n_dedicated_hosts,
        mininet_host_macs,
    )
