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
        self._attack_thread = None     # altrimenti stop()/join()/is_running() danno AttributeError
        self.returncode = None

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
            '--out-file', self.out_file,
        ]
        if self.local_ip:
            cmd.append('--local-ip')

        self._worker_proc = attacker_host.popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        log_path = os.path.splitext(self.out_file)[0] + '.log'
        with open(log_path, 'w') as lf:                 # output del worker su file
            for raw in self._worker_proc.stdout:
                line = raw.decode(errors='replace').rstrip()
                print(line)
                lf.write(line + '\n'); lf.flush()

        self.returncode = self._worker_proc.wait()
        if self.returncode != 0:
            print(f"[!] Worker terminato con codice {self.returncode}")

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