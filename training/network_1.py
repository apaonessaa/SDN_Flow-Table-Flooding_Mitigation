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

def replay_worker(host, pcap_file, launch_time, offset, proc_holder, stop_event):
    sleep_duration = (launch_time + offset) - time()
    if sleep_duration > 0:
        sleep(sleep_duration)

    if stop_event.is_set():
        return

    iface = host.defaultIntf().name
    cmd = f"tcpreplay --intf1={iface} --multiplier=1.0 {pcap_file}"
    print(f"[*] {host.name}: avvio a t+{offset:.3f}s -> {cmd}")

    proc = host.popen(cmd, shell=True)
    proc_holder[host.name] = proc  # lo rendiamo visibile al thread principale
    proc.wait()
    print(f"[*] {host.name}: replay terminato")

def kill_replay_traffic(procs, stop_event):
    stop_event.set()
    for name, proc in procs.items():
        if proc.poll() is None:
            print(f"[*] Termino tcpreplay su {name} (pid={proc.pid})")
            proc.terminate()
    sleep(1)
    for name, proc in procs.items():
        if proc.poll() is None:
            proc.kill()

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
    PCAP_PREFIX = "/home/ubuntu/Desktop/progetto/training/pcaps/traffic"
    metadata = load_metadata(PCAP_PREFIX)
    offsets = metadata["offsets"]
    launch_time = time() + 1.0
    threads = []
    procs = {}                     # host.name -> Popen
    stop_event = threading.Event()

    for h in net.hosts:
        ip = h.IP()
        pcap_file = f"{PCAP_PREFIX}_{ip.replace('.', '_')}.pcap"
        offset = offsets.get(ip, 0.0)

        t = threading.Thread(
            target=replay_worker,
            args=(h, pcap_file, launch_time, offset, procs, stop_event)
        )
        threads.append(t)

    for t in threads:
        t.start()

    sleep(5)

    info("*** [TRAFFIC] Flow Table Flooding Attack\n")
    PAIRS = {
        'h1':'h2',
        'h3':'h4',
        'h5':'h6',
    }
    ATTACK_NPACKET   = [200, 80, 50]
    ATTACK_ANP       = [20, 10, 5]
    ATTACK_WAIT      = None

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

        sleep(5)

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

        sleep(5)

    kill_replay_traffic(procs, stop_event)
    for t in threads:
        t.join(timeout=5)

    info("*** [END]\n")
    #CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    main()
