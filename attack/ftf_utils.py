import os
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
        start = int(attack_frequency * 0.2)
        return random.randint(start, attack_frequency - 1) if attack_frequency != 0 else 0

    @staticmethod
    def random_anp(added_new_packet: int) -> int:
        start = int(added_new_packet * 0.2)
        return random.randint(start, added_new_packet) if added_new_packet != 0 else 0

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
    def estimate_hard_timeout(dst_ip: str, max_int: int, max_interval: int) -> int:
        for i in range(1, max_int+1):
            rtt0 = FTFUtils.measure_rtt(dst_ip)
            time.sleep(1)
            rtt1 = FTFUtils.measure_rtt(dst_ip)

            if rtt0 is None or rtt1 is None:
                print(f"[!] Not reachable host {dst_ip}")
                time.sleep(10)
                continue

            if abs(rtt1 - rtt0) / rtt1 <= 0.5:
                print("[!] No Packet In")
                time.sleep(10)
                continue
            
            for interval in range(2, i*max_interval+1):
                time.sleep(1)
                rtti = FTFUtils.measure_rtt(dst_ip)

                if rtti is None:
                    print("[!] Failed Probe")
                    break
                
                if rtti > rtt0 or abs(rtti - rtt0) / rtti <= 0.5:
                    return interval

                print(f"[.] Interval={interval}", flush=True)
            
            time.sleep(max_interval)
        
        return max_interval

    @staticmethod
    def estimate_idle_timeout(dst_ip: str, max_int: int, hard_timeout: int) -> int:
        for _ in range(max_int):
            rtt0 = FTFUtils.measure_rtt(dst_ip) 
            time.sleep(1)
            rtt1 = FTFUtils.measure_rtt(dst_ip)

            if rtt0 is None or rtt1 is None:
                print(f"[!] Not reachable host {dst_ip}")
                time.sleep(10)
                continue

            if abs(rtt1 - rtt0) / rtt1 <= 0.5:
                print("[!] No Packet In")
                time.sleep(10)
                continue

            inter_sum = 1
            for interval in range(2, hard_timeout):

                time.sleep(interval)

                inter_sum += interval

                if inter_sum >= hard_timeout:
                    rtt0 = FTFUtils.measure_rtt(dst_ip)
                    print(f"[+] rtt0={rtt0}")
                    inter_sum = interval
                    time.sleep(interval)

                rtti = FTFUtils.measure_rtt(dst_ip)

                if rtti is None:
                    print("[!] Failed Probe")
                    break

                if rtti > rtt0 or abs(rtti - rtt0) / rtti <= 0.5:
                    return interval
                    
                print(f"[.] Interval={interval}", flush=True)
                            
            time.sleep(hard_timeout)
        
        return hard_timeout
