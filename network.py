#!/usr/bin/python
import subprocess
import sys
from time import sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel, info

# Custom modules
from monitor import NetworkMonitor
from flow_table_flood import FloodAttack

# ── Parametri configurabili ──────────────────────────────────────────────────
MONITOR_INTERVAL = 1      # secondi tra un campionamento e l'altro
ATTACK_NPACKET   = 10000   # pacchetti per thread di attacco
ATTACK_THREADS   = 4
ATTACK_BSIZE     = 100    # burst size
ATTACK_BDELAY    = 1      # burst delay (s)
PRE_ATTACK_WAIT  = 10     # secondi di monitoraggio prima dell'attacco
ATTACK_WAIT      = None     # secondi di durata attacco (None = aspetta fine naturale)
POST_ATTACK_WAIT = 10     # secondi di monitoraggio dopo l'attacco
# ────────────────────────────────────────────────────────────────────────────


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
        autoSetMacs=True,
        autoStaticArp=True,
    )

    info("*** Start Network\n")
    net.start()

    info("*** Scripts\n")
    run_script('./config/ft_size.sh')

    # Recupera s2 dal nodo della rete (non dalla topologia)
    s2 = net.get('s2')

    wait_for_network(net)
    
    
    # ── Fase 1: monitoraggio PRE-attacco ────────────────────────────────────
    info("*** [PRE-ATTACK] Avvio monitor\n")
    monitor = NetworkMonitor(
        net,
        interval=MONITOR_INTERVAL,
        label='monitor',
    )
    monitor.start()
    sleep(PRE_ATTACK_WAIT)

    # ── Fase 2: attacco (monitor continua a girare) ──────────────────────────
    info("*** [ATTACK] Avvio flood attack su s2\n")
    attack = FloodAttack(
        switch=s2,
        iface=s2.intfList()[2],
        npacket=ATTACK_NPACKET,
        threads=ATTACK_THREADS,
        bsize=ATTACK_BSIZE,
        bdelay=ATTACK_BDELAY,
    )

    attack.start()

    if ATTACK_WAIT is not None:
        # Aspetta al massimo ATTACK_WAIT secondi, poi forza lo stop
        attack.join(timeout=ATTACK_WAIT)
        attack.stop()
    else:
        # Aspetta la fine naturale dell'attacco
        attack.join()

    # ── Fase 3: monitoraggio POST-attacco ───────────────────────────────────
    info("*** [POST-ATTACK] Monitoraggio residuo\n")
    sleep(POST_ATTACK_WAIT)

    # Ferma il monitor solo ora: ha coperto tutte e tre le fasi
    monitor.stop()
    monitor.wait()
    info("*** Monitor fermato\n")
    

    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    main()
