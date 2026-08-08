#!/usr/bin/python
from __future__ import annotations
import subprocess
import sys
from time import sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.cli import CLI
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
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            self.addLink(h, s1)
        for i in range(3, 5):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            self.addLink(h, s3)

        h5 = self.addHost('h5', ip='10.0.0.5/24')
        self.addLink(h5, s2)
        attacker = self.addHost('h6', ip='10.0.0.6/24')
        self.addLink(attacker, s2)


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

def start_tcpreplay(iface: str, pcap: str, loop: int = 500) -> subprocess.Popen:
    """Avvia un tcpreplay in background e restituisce il Popen associato."""
    return subprocess.Popen(
        ["sudo", "tcpreplay", "--loop", str(loop), "-i", iface, pcap],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

def wait_for_tcpreplays(procs: list[tuple[str, subprocess.Popen]]) -> None:
    """Attende il termine di tutti i processi tcpreplay avviati.

    I processi girano già in parallelo (sono stati lanciati con Popen),
    quindi attenderli in sequenza con .wait() non li serializza: si limita
    a bloccare finché anche l'ultimo non ha terminato. Se uno dei processi
    termina con errore, l'eccezione viene sollevata solo dopo aver atteso
    (e loggato) tutti gli altri, per non lasciare repliche orfane in corso.
    """
    info(f"*** In attesa del termine di {len(procs)} tcpreplay...\n")
    failures = []
    for iface, proc in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            failures.append((iface, proc.returncode, stderr))
            info(f"*** [tcpreplay:{iface}] terminato con errore ({proc.returncode})\n")
        else:
            info(f"*** [tcpreplay:{iface}] completato\n")

    if failures:
        for iface, code, stderr in failures:
            print(f"[tcpreplay:{iface}] exit code {code}\n{stderr}", file=sys.stderr)
        raise RuntimeError(f"{len(failures)} tcpreplay terminati con errore")

    info("*** Tutti i tcpreplay sono terminati con successo\n")


def main() -> None:
    topo = CustomTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        switch=OVSSwitch,
        autoSetMacs=True,
        autoStaticArp=True,
    )
    info("*** Start Network\n")
    net.start()
    info("*** Scripts\n")
    run_script('../config/ft_size.sh')
    wait_for_network(net)

    info("*** [TRAFFIC]\n")
    interfaces = [
        's1-eth2', 's1-eth3',
        's2-eth3', 's2-eth4',
        's3-eth2', 's3-eth3',
    ]
    procs = [(iface, start_tcpreplay(iface, "normal.pcap")) for iface in interfaces]

    wait_for_tcpreplays(procs)

    info("*** [END]\n")
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    main()
