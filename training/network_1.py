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

PAIRS = {
    'h1':'h2',
    'h3':'h4',
    'h5':'h6',
}
ATTACK_NPACKET   = 100      # pacchetti per thread di attacco
ATTACK_ANP       = 10       # burst size
ATTACK_FREQUENCY = 10       # burst delay (s)
PRE_ATTACK_WAIT  = 10       # secondi di monitoraggio prima dell'attacco
ATTACK_WAIT      = None     # secondi di durata attacco (None = aspetta fine naturale)
POST_ATTACK_WAIT = 10       # secondi di monitoraggio dopo l'attacco
PCAP_PREFIX = "/home/ubuntu/Desktop/progetto/training/pcaps/traffic"

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
    #wait_for_network(net)
    sleep(20)

    info("*** [TRAFFIC] Flow Table Flooding Attack\n")

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
    for src, dst in PAIRS.items():
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

        attack.start()

        if ATTACK_WAIT is not None:
            attack.join(timeout=ATTACK_WAIT)
            attack.stop()
        else:
            attack.join()

        sleep(20)

        info(f"*** [ATTACK] Coppia: {dst} -> {src}\n")

        attack = FTFAttack(
            attacker=dst,
            probe_dst=src,
            npacket=ATTACK_NPACKET,
            added_new_packets=ATTACK_ANP,
            max_iter=100,
            max_interval=20,
            net=net
        )

        attack.start()

        if ATTACK_WAIT is not None:
            attack.join(timeout=ATTACK_WAIT)
            attack.stop()
        else:
            attack.join()
        
        sleep(20)


    info("*** [END]\n")
    #CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    main()
