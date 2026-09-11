#!/usr/bin/python
import json
import threading
import subprocess
import socket
import math
import csv
import pandas as pd
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.topo import Topo
from mininet.log import setLogLevel, info

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
root = os.path.dirname(parent)
sys.path.append(parent)
sys.path.append(root)

from attack.ftf_attack import FTFAttack

import random
rng = random.Random(42)

SET_FT_SIZE_PATH = root + "/config/set_ft_size.sh"

class CustomTopo1(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h11')
    att_hosts = ('h15', 'h14')
    victim_dpid = 1
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s1)
        self.addLink(h6, s1)
        self.addLink(h7, s1)
        self.addLink(h8, s1)
        self.addLink(h9, s1)
        self.addLink(h10, s1)
        self.addLink(h11, s1)
        self.addLink(h12, s1)
        self.addLink(h13, s1)
        self.addLink(h14, s1)
        self.addLink(h15, s1)

class CustomTopo2(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h11')
    att_hosts = ('h15', 'h14')
    victim_dpid = 2
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s1)
        self.addLink(h6, s1)
        self.addLink(h7, s1)
        self.addLink(h8, s1)
        self.addLink(h9, s1)
        self.addLink(h10, s2)
        self.addLink(h11, s2)
        self.addLink(h12, s2)
        self.addLink(h13, s2)
        self.addLink(h14, s2)
        self.addLink(h15, s2)

class CustomTopo3(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h14')
    att_hosts = ('h15', 'h14')
    victim_dpid = 3
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s1)
        self.addLink(h6, s2)
        self.addLink(h7, s2)
        self.addLink(h8, s2)
        self.addLink(h9, s2)
        self.addLink(h10, s2)
        self.addLink(h11, s3)
        self.addLink(h12, s3)
        self.addLink(h13, s3)
        self.addLink(h14, s3)
        self.addLink(h15, s3)

class CustomTopo4(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h12')
    att_hosts = ('h15', 'h2')
    victim_dpid = 1
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow14')
        s4 = self.addSwitch('s4', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s2, s4)
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s2)
        self.addLink(h6, s2)
        self.addLink(h7, s2)
        self.addLink(h8, s3)
        self.addLink(h9, s3)
        self.addLink(h10, s3)
        self.addLink(h11, s3)
        self.addLink(h12, s4)
        self.addLink(h13, s4)
        self.addLink(h14, s4)
        self.addLink(h15, s1)

class CustomTopo5(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h13')
    att_hosts = ('h15', 'h10')
    victim_dpid = 3
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow14')
        s4 = self.addSwitch('s4', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s3, s4)
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s2)
        self.addLink(h6, s2)
        self.addLink(h7, s2)
        self.addLink(h8, s2)
        self.addLink(h9, s3)
        self.addLink(h10, s3)
        self.addLink(h11, s3)
        self.addLink(h12, s4)
        self.addLink(h13, s4)
        self.addLink(h14, s4)
        self.addLink(h15, s3)

class CustomTopo6(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h13')
    att_hosts = ('h15', 'h4')
    victim_dpid = 2
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow14')
        s4 = self.addSwitch('s4', cls=OVSSwitch, protocols='OpenFlow14')
        s5 = self.addSwitch('s5', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s2, s4)
        self.addLink(s2, s5)
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s2)
        self.addLink(h5, s2)
        self.addLink(h6, s2)
        self.addLink(h7, s3)
        self.addLink(h8, s3)
        self.addLink(h9, s3)
        self.addLink(h10, s4)
        self.addLink(h11, s4)
        self.addLink(h12, s4)
        self.addLink(h13, s5)
        self.addLink(h14, s5)
        self.addLink(h15, s2)

class CustomTopo7(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h13')
    att_hosts = ('h15', 'h4')
    victim_dpid = 2
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow14')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow14')
        s4 = self.addSwitch('s4', cls=OVSSwitch, protocols='OpenFlow14')
        s5 = self.addSwitch('s5', cls=OVSSwitch, protocols='OpenFlow14')
        s6 = self.addSwitch('s6', cls=OVSSwitch, protocols='OpenFlow14')
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s2, s4)
        self.addLink(s2, s5)
        self.addLink(s3, s6)
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
        h10 = self.addHost('h10', ip='10.0.0.10/24', mac='00:00:00:00:00:10')
        h11 = self.addHost('h11', ip='10.0.0.11/24', mac='00:00:00:00:00:11')
        h12 = self.addHost('h12', ip='10.0.0.12/24', mac='00:00:00:00:00:12')
        h13 = self.addHost('h13', ip='10.0.0.13/24', mac='00:00:00:00:00:13')
        h14 = self.addHost('h14', ip='10.0.0.14/24', mac='00:00:00:00:00:14')
        h15 = self.addHost('h15', ip='10.0.0.15/24', mac='00:00:00:00:00:15')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s6)
        self.addLink(h4, s6)
        self.addLink(h5, s2)
        self.addLink(h6, s2)
        self.addLink(h7, s3)
        self.addLink(h8, s3)
        self.addLink(h9, s3)
        self.addLink(h10, s4)
        self.addLink(h11, s4)
        self.addLink(h12, s4)
        self.addLink(h13, s5)
        self.addLink(h14, s5)
        self.addLink(h15, s2)

def load_metadata(prefix):
    with open(f"{prefix}traffic_metadata.json") as f:
        return json.load(f)

def start_controller(env_vars: dict, log_path: str = "controller.log") -> subprocess.Popen:
    log_file = open(log_path, "a")
    full_env = os.environ.copy()
    full_env.update(env_vars)
    proc = subprocess.Popen(
        ['ryu-manager','--verbose','--observe-links','controller.py'],
        env=full_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,)
    return proc

def wait_for_controller(host: str = "127.0.0.1", port: int = 6653, timeout: float = 15.0) -> bool:
    deadline = time() + timeout
    while time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            sleep(0.5)
    return False

def stop_controller(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def run_script(path: str, env_vars: dict) -> None:
    info(f"- {path}\n")
    full_env = os.environ.copy()
    full_env.update(env_vars)
    result = subprocess.run(
        ['bash', path],
        env=full_env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        info(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Error code: {result.returncode}")

def wait_for_network(src: Host, dst: Host, timeout: int = 60) -> None:
    info(f"*** Verifica connettività {src.IP()}->{dst.IP()}\n")
    for attempt in range(timeout):
        result = src.cmd(f'ping -c 1 -W 1 {dst.IP()}')
        if '1 received' in result:
            info(f"*** Rete pronta (tentativo {attempt + 1})\n")
            return
        sleep(1)
    info("*** ATTENZIONE: rete non pronta dopo timeout\n")

def replay_worker(host, pcap_file, launch_time, duration, offset, proc_holder, stop_event):
    sleep_duration = (launch_time + offset) - time()
    if sleep_duration > 0:
        sleep(sleep_duration)
    if stop_event.is_set():
        return
    iface = host.defaultIntf().name
    if duration > 0:
        cmd = f"tcpreplay --intf1={iface} --loop=0 --duration={duration} {pcap_file}"
    else:
        cmd = f"tcpreplay --intf1={iface} --loop=0 {pcap_file}"
    print(f"[*] {host.name}: avvio a t+{offset:.3f}s -> {cmd}")
    proc = host.popen(cmd, shell=True)
    proc_holder[host.name] = proc
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

def sample_attack_rate(npkts, flow_table_size, min_anp=5):
    stealth_profiles = {
        "aggressive": (5, 20),
        "moderate":   (60, 180),
        "stealth":    (300, 420),
    }
    profile = rng.choice(list(stealth_profiles.keys()))
    t_min, t_max = stealth_profiles[profile]
    fill_time = rng.uniform(t_min, t_max)
    anp = max(min_anp, math.ceil(flow_table_size / fill_time))
    return anp, profile, fill_time

def main() -> None:
    # Topologies
    topos = [
        CustomTopo1(),
        CustomTopo2(),
        CustomTopo3(),
        CustomTopo4(),
        CustomTopo5(),
        CustomTopo6(),
        CustomTopo7()
    ]

    pcap_prefixes = [
        current + "/pcaps/traffics/scenario_000/",
        current + "/pcaps/traffics/scenario_001/",
        current + "/pcaps/traffics/scenario_002/",
        current + "/pcaps/traffics/scenario_003/",
        current + "/pcaps/traffics/scenario_004/",
        current + "/pcaps/traffics/scenario_005/",
        current + "/pcaps/traffics/scenario_006/",
        current + "/pcaps/traffics/scenario_007/",
        current + "/pcaps/traffics/scenario_008/",
        current + "/pcaps/traffics/scenario_009/",
        current + "/pcaps/traffics/scenario_010/",
        current + "/pcaps/traffics/scenario_011/",
        current + "/pcaps/traffics/scenario_012/",
        current + "/pcaps/traffics/scenario_013/",
        current + "/pcaps/traffics/scenario_014/",
        current + "/pcaps/traffics/scenario_015/",
        current + "/pcaps/traffics/scenario_016/",
        current + "/pcaps/traffics/scenario_017/",
        current + "/pcaps/traffics/scenario_018/",
        current + "/pcaps/traffics/scenario_019/",
        current + "/pcaps/traffics/scenario_020/",
        current + "/pcaps/traffics/scenario_021/",
        current + "/pcaps/traffics/scenario_022/",
        current + "/pcaps/traffics/scenario_023/",
        current + "/pcaps/traffics/scenario_024/",
        current + "/pcaps/traffics/scenario_025/",
        current + "/pcaps/traffics/scenario_026/",
        current + "/pcaps/traffics/scenario_027/",
        current + "/pcaps/traffics/scenario_028/",
        current + "/pcaps/traffics/scenario_029/",
        current + "/pcaps/traffics/scenario_030/",
        current + "/pcaps/traffics/scenario_031/",
        current + "/pcaps/traffics/scenario_032/",
        current + "/pcaps/traffics/scenario_033/",
        current + "/pcaps/traffics/scenario_034/",
        current + "/pcaps/traffics/scenario_035/",
        current + "/pcaps/traffics/scenario_036/",
        current + "/pcaps/traffics/scenario_037/",
        current + "/pcaps/traffics/scenario_038/",
        current + "/pcaps/traffics/scenario_039/",
        current + "/pcaps/traffics/scenario_040/",
        current + "/pcaps/traffics/scenario_041/",
        current + "/pcaps/traffics/scenario_042/",
        current + "/pcaps/traffics/scenario_043/",
        current + "/pcaps/traffics/scenario_044/",
        current + "/pcaps/traffics/scenario_045/",
        current + "/pcaps/traffics/scenario_046/",
        current + "/pcaps/traffics/scenario_047/",
        current + "/pcaps/traffics/scenario_048/",
    ]
    
    att_wait = None
    k_factor = 3
    n_host = 15
    full_size = int(math.factorial(n_host) / math.factorial(n_host-2)) + 1
    half_size = int(full_size / 2)
    log_timestamp = int(time())

    #df = pd.read_csv(current + "/pcaps/scenarios/manifest.csv")
    csv_path = current + f"/logs/network_settings_{log_timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topo", "hard_timeout", "idle_timeout", "flow_table_size", "npkts", "profile", "anp", "att_duration", "fill_time"])

    for i in range(8):
        topo = topos[i%len(topos)]
        # hard_timeout: 5 < hard <= 30
        hard_timeout = rng.randint(6, 30)
        # idle_timeout: idle <= hard
        idle_timeout = rng.randint(2, hard_timeout)
        # flow_table_size
        flow_table_size = rng.randint(half_size, 30*n_host)
        # att_npkts: att_npkts >= flow_table_size
        margin = rng.randint(0, 10*n_host)
        npkts = flow_table_size + margin
        # att_anp: 1 min - ~10 min
        #df['start'] = pd.to_datetime(df['start'])
        #df['end'] = pd.to_datetime(df['end'])
        #T_min = 60
        #T_max = int((df['end'].iloc[i] - df['start'].iloc[i]).total_seconds() - 30)
        anp, profile, fill_time = sample_attack_rate(npkts, flow_table_size)
        
        # att_duration
        att_duration = -1

        info("*** CONTROLLER\n")
        controller_envs = {
            "GROUP_ID": f"{i}",
            "LABEL": "1",
            "DATASET_PATH": f"{current}/logs/dataset_{log_timestamp}.csv",
            "FLOW_TABLE_SIZE": f"{flow_table_size}",
            "IDLE_TIMEOUT": f"{idle_timeout}",
            "HARD_TIMEOUT": f"{hard_timeout}",
            "K_FACTOR": f"{k_factor}",
            "DPID": f"{topo.victim_dpid}"
        }
        controller = start_controller(controller_envs, log_path=f"{current}/logs/controller_{i}_{log_timestamp}.log")

        if controller.poll() is not None:
            raise RuntimeError(f"Il controller è terminato subito con codice {controller.returncode}, controlla controller.log")

        if not wait_for_controller(port=6653, timeout=20):
            controller.terminate()
            raise RuntimeError("Timeout: il controller non si è messo in ascolto sulla porta 6653")
        
        info("*** NETWORK\n")
        try:
            net = Mininet(
                topo=topo,
                controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
                switch=OVSSwitch,
                autoStaticArp=True
            )
            net.start()

            info("*** NETWORK CONFIG\n")
            ft_envs = {
                "SWITCHES": " ".join([sw.name for sw in net.switches]),
                "FLOW_TABLE_SIZE": f"{flow_table_size}",
            }
            run_script(SET_FT_SIZE_PATH, ft_envs)
                
            info("*** START TRAFFIC\n")

            metadata = load_metadata(pcap_prefixes[i])
            offsets = metadata["offsets"]
            original_capture_duration = metadata["original_capture_duration"]
            packet_counts = metadata["packet_counts"]

            launch_time = time() + 1.0
            procs = {}
            stop_event = threading.Event()
            threads = []

            (attacker, dst) = topo.att_hosts
            for h in net.hosts:
                if h.name != attacker:
                    ip = h.IP()
                    pcap_file = f"{pcap_prefixes[i]}traffic_{ip.replace('.', '_')}.pcap"
                    offset = offsets.get(ip, 0.0)
                    t = threading.Thread(
                        target=replay_worker,
                        args=(h, pcap_file, launch_time, att_duration, offset, procs, stop_event)
                    )
                    threads.append(t)

            info(f"*** [FTF ATTACK] ({attacker} -> {dst}), NPkts={npkts}, ANP={anp}\n")
            attack = FTFAttack(
                attacker=attacker,
                probe_dst=dst,
                npacket=npkts,
                added_new_packets=anp,
                max_iter=10,
                max_interval=30,
                net=net,
                out_file=f"{current}/logs/ftf_{log_timestamp}.csv",
            )

            (src, dst) = topo.conn_hosts
            h_src, h_dst = net.get(src, dst)
            wait_for_network(h_src, h_dst)

            sleep(hard_timeout)

            # Start threads
            t0 = time()

            attack.start()
            for t in threads:
                t.start()

            # Stop threads
            attack.join()
            kill_replay_traffic(procs, stop_event)
            for t in threads:
                t.join(timeout=5)

            att_duration = math.ceil(time() - t0)
            info(f"*** END TRAFFIC: Session duration {att_duration}\n")

        finally:
            sleep(60)
            net.stop()
            stop_controller(controller)
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)

        sleep(30)

        info("*** CONTROLLER\n")
        controller_envs = {
            "GROUP_ID": f"{i}",
            "LABEL": "0",
            "DATASET_PATH": f"{current}/logs/dataset_{log_timestamp}.csv",
            "FLOW_TABLE_SIZE": f"{flow_table_size}",
            "IDLE_TIMEOUT": f"{idle_timeout}",
            "HARD_TIMEOUT": f"{hard_timeout}",
            "K_FACTOR": f"{k_factor}",
            "DPID": f"{topo.victim_dpid}"
        }
        controller = start_controller(controller_envs, log_path=f"{current}/logs/controller_{i}_{log_timestamp}.log")

        if controller.poll() is not None:
            raise RuntimeError(f"Il controller è terminato subito con codice {controller.returncode}, controlla controller.log")

        if not wait_for_controller(port=6653, timeout=20):
            controller.terminate()
            raise RuntimeError("Timeout: il controller non si è messo in ascolto sulla porta 6653")
        
        info("*** NETWORK\n")
        try:
            net = Mininet(
                topo=topo,
                controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
                switch=OVSSwitch,
                autoStaticArp=True
            )
            net.start()

            info("*** NETWORK CONFIG\n")
            ft_envs = {
                "SWITCHES": " ".join([sw.name for sw in net.switches]),
                "FLOW_TABLE_SIZE": f"{flow_table_size}",
            }
            run_script(SET_FT_SIZE_PATH, ft_envs)
                
            info("*** START TRAFFIC\n")

            metadata = load_metadata(pcap_prefixes[i])
            offsets = metadata["offsets"]
            original_capture_duration = metadata["original_capture_duration"]
            packet_counts = metadata["packet_counts"]

            launch_time = time() + 1.0
            procs = {}
            stop_event = threading.Event()
            threads = []

            (attacker, dst) = topo.att_hosts
            for h in net.hosts:
                if h.name != attacker:
                    ip = h.IP()
                    offset = offsets.get(ip, 0.0)
                    pcap_file = f"{pcap_prefixes[i]}traffic_{ip.replace('.', '_')}.pcap"
                    if offset < att_duration:
                        t = threading.Thread(
                            target=replay_worker,
                            args=(h, pcap_file, launch_time, -1, offset, procs, stop_event)
                        )
                        threads.append(t)

            (src, dst) = topo.conn_hosts
            h_src, h_dst = net.get(src, dst)
            wait_for_network(h_src, h_dst)

            sleep(hard_timeout)

            # Start threads
            for t in threads:
                t.start()

            # Stop threads
            sleep(att_duration+5)
            kill_replay_traffic(procs, stop_event)
            for t in threads:
                t.join(timeout=5)
            
            info(f"*** END TRAFFIC\n")

        finally:
            sleep(60)
            net.stop()
            stop_controller(controller)
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i%len(topos), hard_timeout, idle_timeout, flow_table_size, npkts, profile, anp, att_duration, fill_time])

        sleep(30)

if __name__ == '__main__':
    setLogLevel('info')
    main()
