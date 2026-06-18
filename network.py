#!/usr/bin/python
import subprocess
import sys
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel, info

# Custom modules
from mmmmonitor import start_monitor
from flow_table_flood import FloodAttack

import builtins
builtins.start_monitor = start_monitor
builtins.FloodAttack = FloodAttack

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
        #self.addLink(s1, s3)

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
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    main()