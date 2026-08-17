#!/usr/bin/python
import json
import threading
import subprocess
import sys
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
#from mininet.cli import CLI
from mininet.log import setLogLevel, info

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
    wait_for_network(net)

    info("*** [TRAFFIC]\n")

    metadata = load_metadata(PCAP_PREFIX)
    offsets = metadata["offsets"]

    # istante di riferimento COMUNE per tutti i thread, calcolato
    # una sola volta, cosi' ogni offset e' relativo allo stesso punto zero
    launch_time = time() + 1.0  # 1s di margine per lo spawn dei thread

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
    for t in threads:
        t.join()

    info("*** [END]\n")
    sleep(60)
    #CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    main()
