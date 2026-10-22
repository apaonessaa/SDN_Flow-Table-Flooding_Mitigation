import os
import subprocess
from mininet.net import Mininet
from threading import Thread, Event

class FTFAttack:

    WORKER_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/ftf_worker.py"
    
    def __init__(self, attacker: str, probe_dst: str, 
                npacket: int, added_new_packets: int, added_new_macs: int,
                n: int, t_max: int, 
                net: Mininet, 
                local_ip: bool=False, 
                out_file: str='/tmp/ftf.csv'):
        self.attacker = attacker
        self.probe_dst = probe_dst
        self.npacket = npacket
        self.added_new_packets = added_new_packets
        self.added_new_macs = added_new_macs
        self.n = n
        self.t_max = t_max
        self._stop_event: Event = Event()
        self._worker_proc = None
        self.local_ip = local_ip
        self.out_file = out_file
        self.net = net

    def run(self):
        attacker_host = self.net.get(self.attacker)
        dst_host = self.net.get(self.probe_dst)

        cmd = [
            'python3', self.WORKER_PATH,
            '--iface', attacker_host.defaultIntf().name,
            '--dst-ip', dst_host.IP(),
            '--npacket', str(self.npacket),
            '--anp', str(self.added_new_packets),
            '--anm', str(self.added_new_macs),
            '--n', str(self.n),
            '--t-max', str(self.t_max),
            '--local-ip' if self.local_ip else ''
            '--out-file', self.out_file,
        ]

        print(f"[*] FTF attack worker: {' '.join(cmd)}")
        self._worker_proc = attacker_host.popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT
        )

        for line in self._worker_proc.stdout:
            print(line.decode(errors='replace').rstrip())

        self._worker_proc.wait()

    def start(self) -> None:
        self._stop_event.clear()
        self._attack_thread = Thread(target=self.run, name="flood-attack", daemon=True)
        self._attack_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_proc and self._worker_proc.poll() is None:
            self._worker_proc.terminate()
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