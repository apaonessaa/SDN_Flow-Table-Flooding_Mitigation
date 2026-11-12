from mininet.topo import Topo
from mininet.node import OVSSwitch
from mininet.log import info

class TestOnlineTopology(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h10')
    att_hosts = ('h15', 'h14')
    victim_dpid = 2
    ping_pairs = [
        ('h1', 'h1'), ('h1', 'h2'), ('h1', 'h3'), ('h1', 'h4'), ('h1', 'h5'), 
        ('h1', 'h6'), ('h1', 'h7'), ('h1', 'h8'), ('h1', 'h9'), ('h1', 'h10'),
        ('h1', 'h11'), ('h1', 'h12'), ('h2', 'h2'), ('h2', 'h3'), ('h2', 'h4'),
        ('h2', 'h5'), ('h2', 'h6'), ('h2', 'h7'), ('h2', 'h8'), ('h2', 'h9'), 
        ('h2', 'h10'), ('h2', 'h11'), ('h2', 'h12'), ('h3', 'h3'), ('h3', 'h4'),
        ('h3', 'h5'), ('h3', 'h6'), ('h3', 'h7'), ('h3', 'h8'), ('h3', 'h9'),
        ('h3', 'h10'), ('h3', 'h11'), ('h3', 'h12'), ('h4', 'h4'), ('h4', 'h5'),
        ('h4', 'h6'), ('h4', 'h7'), ('h4', 'h8'), ('h4', 'h9'), ('h4', 'h10'),
        ('h4', 'h11'), ('h4', 'h12'), ('h5', 'h5'), ('h5', 'h6'), ('h5', 'h7'),
        ('h5', 'h8'), ('h5', 'h9'), ('h5', 'h10'), ('h5', 'h11'), ('h5', 'h12'),
        ('h6', 'h6'), ('h6', 'h7'), ('h6', 'h8'), ('h6', 'h9'), ('h6', 'h10'),
        ('h6', 'h11'), ('h6', 'h12'), ('h7', 'h7'), ('h7', 'h8'), ('h7', 'h9'),
        ('h7', 'h10'), ('h7', 'h11'), ('h7', 'h12'), ('h8', 'h8'), ('h8', 'h9'),
        ('h8', 'h10'), ('h8', 'h11'), ('h8', 'h12'), ('h9', 'h9'), ('h9', 'h10'),
        ('h9', 'h11'), ('h9', 'h12'), ('h10', 'h10'), ('h10', 'h11'),
        ('h10', 'h12'), ('h11', 'h11'), ('h11', 'h12'), ('h12', 'h12')
    ]

    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')
        h7 = self.addHost('h7', ip='10.0.0.7/24', mac='00:00:00:00:00:07')
        h8 = self.addHost('h8', ip='10.0.0.8/24', mac='00:00:00:00:00:08')
        h9 = self.addHost('h9', ip='10.0.0.9/24', mac='00:00:00:00:00:09')
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:0a')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:0b')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:0c')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:0d')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:0e')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:0f')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s1)
        self.addLink(h6, s1)
        self.addLink(h7, s3)
        self.addLink(h8, s3)
        self.addLink(h9, s3)
        self.addLink(h10, s3)
        self.addLink(h11, s3)
        self.addLink(h12, s2)
        self.addLink(h13, s2)
        self.addLink(h14, s2)
        self.addLink(h15, s2)
