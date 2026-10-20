import os
import math
import random
import time
import re
import subprocess

from scapy.all import (
    Ether, IP, TCP, UDP, ICMP, Packet, RandIP, RandMAC
)

class FTFUtils:

    SUPPORTED_PROTOS = ["TCP", "UDP", "ICMP"]

    @staticmethod
    def random_mac() -> str:
        return str(RandMAC())

    @staticmethod
    def random_ip() -> str:
        return str(RandIP())

    @staticmethod
    def random_local_ip() -> str:
        return f'10.0.0.{random.randint(1,254)}'

    @staticmethod
    def random_port() -> int:
        return random.randint(1024, 65535)

    @staticmethod
    def random_proto() -> str:
        return random.choice(FTFUtils.SUPPORTED_PROTOS)

    @staticmethod
    def random_data() -> bytes:
        n = random.randint(100, 1400)
        return os.urandom(n)

    @staticmethod
    def build_packet(src_mac: str, src_ip: str, dst_mac: str, dst_ip: str) -> Packet:
        eth = Ether(src=src_mac, dst=dst_mac)
        ip  = IP(src=src_ip, dst=dst_ip)
        proto = FTFUtils.random_proto()
        if proto == "TCP":
            transport = TCP(
                sport=FTFUtils.random_port(),
                dport=FTFUtils.random_port(),
                flags="S"
            )
        elif proto == "UDP":
            transport = UDP(
                sport=FTFUtils.random_port(),
                dport=FTFUtils.random_port()
            )
        elif proto == "ICMP":
            transport = ICMP(
                type=random.choice([0, 8]),   # 0 = reply, 8 = echo request
                code=0
            )
        return eth / ip / transport / FTFUtils.random_data()

    @staticmethod
    def random_af(attack_frequency: int) -> int:
        start = math.ceil(attack_frequency * 0.2)
        return random.randint(start, attack_frequency - 1) if attack_frequency != 0 else 1

    @staticmethod
    def random_anp(added_new_packet: int) -> int:
        start = math.ceil(added_new_packet * 0.2)
        return random.randint(start, added_new_packet) if added_new_packet != 0 else 1

    @staticmethod
    def random_anm(added_new_mac: int) -> int:
        start = math.ceil(added_new_mac * 0.5)
        return random.randint(start, added_new_mac) if added_new_mac != 0 else 1

    @staticmethod
    def measure_rtt(dst_ip: str, timeout: int = 10) -> float:
        out = subprocess.run(
            ['ping', '-c', '1', '-W', str(timeout), dst_ip],
            capture_output=True, text=True
        ).stdout
        m = re.search(r'time=([\d.]+)\s*ms', out)
        if not m:
            return None
        rtt = float(m.group(1))
        t = time.time()
        print(f"t={t}, rtt={rtt}")
        return rtt

    @staticmethod
    def estimate_hard_timeout(dst_ip: str, n: int=10, t_max: int=30) -> int:
        threashold = 0.5
        for i in range(1, n+1):
            rtt0 = FTFUtils.measure_rtt(dst_ip)
            time.sleep(1)
            rtt1 = FTFUtils.measure_rtt(dst_ip)

            if rtt0 is None or rtt1 is None:
                print(f"[!] Not reachable host {dst_ip}")
                time.sleep(t_max)
                continue

            if abs(rtt1 - rtt0) / rtt1 <= threashold:
                print("[!] No Flow Rule installed")
                time.sleep(t_max)
                continue
            
            for t in range(2, i*t_max+1):
                time.sleep(1)
                rtti = FTFUtils.measure_rtt(dst_ip)

                if rtti is None:
                    print("[!] Failed Probe")
                    break
                
                if rtti > rtt0 or abs(rtti - rtt0) / rtti <= threashold:
                    return t

                print(f"[.] t={t} sec", flush=True)
            
            time.sleep(t_max)
        
        return t_max

    @staticmethod
    def estimate_idle_timeout(dst_ip: str, n: int=10, hard_timeout: int=30) -> int:
        threashold = 0.5
        for _ in range(n):
            rtt0 = FTFUtils.measure_rtt(dst_ip) 
            time.sleep(1)
            rtt1 = FTFUtils.measure_rtt(dst_ip)

            if rtt0 is None or rtt1 is None:
                print(f"[!] Not reachable host {dst_ip}")
                time.sleep(hard_timeout)
                continue

            if abs(rtt1 - rtt0) / rtt1 <= threashold:
                print("[!] No Flow Rule installed")
                time.sleep(hard_timeout)
                continue

            t_sum = 1
            for t in range(2, hard_timeout):

                time.sleep(t)
                t_sum += t

                if t_sum >= hard_timeout:
                    rtt0 = FTFUtils.measure_rtt(dst_ip)
                    print(f"[+] rtt0={rtt0}")
                    t_sum = t
                    time.sleep(t)

                rtti = FTFUtils.measure_rtt(dst_ip)

                if rtti is None:
                    print("[!] Failed Probe")
                    break

                if rtti > rtt0 or abs(rtti - rtt0) / rtti <= threashold:
                    return t
                    
                print(f"[.] t={t} sec", flush=True)
                            
            time.sleep(hard_timeout)
        
        return hard_timeout
