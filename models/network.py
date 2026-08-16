#!/usr/bin/python
import json
import threading
import subprocess
import sys
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel, info

from attack.ftf_attack import FTFAttack

class CustomTopo(Topo):
    "Custom Topology Network"
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        self.addLink(h1, s1)
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        self.addLink(h2, s1)
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        self.addLink(h3, s1)
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        self.addLink(h4, s1)

PCAP_PREFIX = "/home/ubuntu/Desktop/progetto/models/pcaps/traffic"
ATTACK_NPACKET   = 20       # pacchetti per thread di attacco
ATTACK_ANP       = 5        # burst size
ATTACK_WAIT      = None     # secondi di durata attacco (None = aspetta fine naturale)

def load_metadata(prefix):
    with open(f"{prefix}_metadata.json") as f:
        return json.load(f)

def run_script(path: str) -> None:
    info(f"- {path}\n")
    result = subprocess.run(
        ['bash', path],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        info(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Error code: {result.returncode}")

def replay_worker(host, pcap_file, launch_time, offset):
    """Attende fino all'istante sincronizzato (launch_time + offset),
    poi avvia tcpreplay per questo host. --multiplier=1.0 preserva
    i delta temporali ORIGINALI tra i pacchetti dello stesso host,
    invece di forzare un rate fisso come --pps."""
    sleep_duration = (launch_time + offset) - time()
    if sleep_duration > 0:
        sleep(sleep_duration)

    iface = host.defaultIntf().name
    cmd = f"tcpreplay --intf1={iface} --multiplier=1.0 {pcap_file}"
    print(f"[*] {host.name}: avvio a t+{offset:.3f}s -> {cmd}")

    proc = host.popen(cmd, shell=True)
    proc.wait()
    print(f"[*] {host.name}: replay terminato")

def main() -> None:
    topo = CustomTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        switch=OVSSwitch,
        autoStaticArp=True,
    )
    info("*** Start Network\n")
    net.start()
    info("*** Scripts\n")
    run_script('../config/ft_size.sh')
    sleep(20)

    info("*** [TRAFFIC] Normal + Flow Table Flooding Attack\n")

    # --- Thread 1: traffic replay ---
    metadata = load_metadata(PCAP_PREFIX)
    offsets = metadata["offsets"]
    launch_time = time() + 1.0 
    threads = []
    for h in net.hosts:
        ip = h.IP()
        pcap_file = f"{PCAP_PREFIX}_{ip.replace('.', '_')}.pcap"
        offset = offsets.get(ip, 0.0)
        t = threading.Thread(
            target=replay_worker,
            args=(h, pcap_file, launch_time, offset)
        )
        threads.append(t)

    # --- Thread 1: FTF attack ---
    src = 'h3'; dst = 'h2'
    info(f"*** [ATTACK] Coppia: {src} -> {dst}\n")
    attack = FTFAttack(
        attacker=src,
        probe_dst=dst,
        npacket=ATTACK_NPACKET,
        added_new_packets=ATTACK_ANP,
        max_iter=100,
        max_interval=20,
        net=net
    )

    for t in threads:
        t.start()

    attack.start()

    for t in threads:
        t.join()

    if ATTACK_WAIT is not None:
        attack.join(timeout=ATTACK_WAIT)
        attack.stop()
    else:
        attack.join()

    info("*** [END]\n")
    #CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    main()
