#!/usr/bin/python
import json
import threading
import subprocess
import sys
from time import time, sleep
from random import randint
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
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        # Hosts
        for i in range(1, 3):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24', mac=f'00:00:00:00:00:0{i}')
            self.addLink(h, s1)
        for i in range(3, 5):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24', mac=f'00:00:00:00:00:0{i}')
            self.addLink(h, s3)

        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        self.addLink(h5, s2)
        attacker = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')
        self.addLink(attacker, s2)

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


def wait_for_network(net: Mininet, timeout: int = 60) -> None:
    """Aspetta che h1 raggiunga h3 prima di avviare il monitor."""
    h1, h3 = net.get('h1', 'h3')
    info("*** Verifica connettività h1->h3\n")
    for attempt in range(timeout):
        result = h1.cmd(f'ping -c 1 -W 1 {h3.IP()}')
        if '1 received' in result:
            info(f"*** Rete pronta (tentativo {attempt + 1})\n")
            return
        sleep(1)
    info("*** ATTENZIONE: rete non pronta dopo timeout, avvio monitor comunque\n")

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

    info("*** [TRAFFIC] Normal traffic\n")
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

    for t in threads:
        t.start()

    info("*** [TRAFFIC] Flow Table Flooding Attack\n")

    PAIRS = {
        'h1':'h2',
        'h3':'h4',
        'h5':'h6',
    }
    ATTACK_NPACKET   = [200, 80, 50]
    ATTACK_ANP       = [20, 10, 5]
    ATTACK_WAIT      = None
    PCAP_PREFIX = "/home/ubuntu/Desktop/progetto/training/pcaps/traffic"

    """
    h1-eth0<->s1-eth2 (OK OK) 
    h2-eth0<->s1-eth3 (OK OK) 
    h3-eth0<->s3-eth2 (OK OK) 
    h4-eth0<->s3-eth3 (OK OK) 
    h5-eth0<->s2-eth3 (OK OK) 
    h6-eth0<->s2-eth4 (OK OK) 
    s1-eth1<->s2-eth1 (OK OK) 
    s2-eth2<->s3-eth1 (OK OK)
    """
    i = 0
    for src, dst in PAIRS.items():
        info(f"*** [ATTACK] Coppia: {src} -> {dst}, NPkts={ATTACK_NPACKET[i]}, ANP={ATTACK_ANP[i]}\n")
        attack = FTFAttack(
            attacker=src,
            probe_dst=dst,
            npacket=ATTACK_NPACKET[i],
            added_new_packets=ATTACK_ANP[i],
            max_iter=100,
            max_interval=50,
            net=net
        )
        attack.start()
        if ATTACK_WAIT is not None:
            attack.join(timeout=ATTACK_WAIT)
            attack.stop()
        else:
            attack.join()

        sleep(randint(5, 20))

        info(f"*** [ATTACK] Coppia: {dst} -> {src}, NPkts={ATTACK_NPACKET[(i+1)%3]}, ANP={ATTACK_ANP[(i+1)%3]}\n")
        attack = FTFAttack(
            attacker=dst,
            probe_dst=src,
            npacket=ATTACK_NPACKET[(i+1)%3],
            added_new_packets=ATTACK_ANP[(i+1)%3],
            max_iter=100,
            max_interval=50,
            net=net
        )
        attack.start()
        if ATTACK_WAIT is not None:
            attack.join(timeout=ATTACK_WAIT)
            attack.stop()
        else:
            attack.join()

        i=(i+1)%3

        sleep(randint(5, 20))

    for t in threads:
        t.join()

    info("*** [END]\n")
    #CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    main()
