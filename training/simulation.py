#!/usr/bin/python
import json
import threading
import subprocess
import socket
import math
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
sys.path.append(parent)

from attack.ftf_attack import FTFAttack

SET_FT_SIZE_PATH = parent + "/config/set_ft_size.sh"

class CustomTopo1(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h11')
    att_hosts = ('h15', 'h14')
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

class CustomTopo2(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h12')
    att_hosts = ('h15', 'h2')
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

class CustomTopo3(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h13')
    att_hosts = ('h15', 'h10')
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

class CustomTopo4(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h13')
    att_hosts = ('h15', 'h4')
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

def load_metadata(prefix):
    with open(f"{prefix}_metadata.json") as f:
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

def main() -> None:
    # Topologies
    topos = [
        CustomTopo1(),
        CustomTopo2(),
        CustomTopo3(),
        CustomTopo4(),
    ]

    pcap_prefixes = [
        current + "/pcaps/traffics/scenario_001/traffic",
        current + "/pcaps/traffics/scenario_002/traffic",
        current + "/pcaps/traffics/scenario_003/traffic",
        current + "/pcaps/traffics/scenario_004/traffic",
        """
        current + "/pcaps/traffics/scenario_005/traffic",
        current + "/pcaps/traffics/scenario_006/traffic",
        current + "/pcaps/traffics/scenario_007/traffic",
        current + "/pcaps/traffics/scenario_008/traffic",
        """
    ]
    n_host = 15
    min_size = int(math.factorial(n_host) / math.factorial(n_host-2) + 1)
    flow_table_sizes = [min_size, 15*n_host, 25*n_host, 30*n_host]
    idle_timeouts = [5, 10, 15, 20]
    hard_timeouts = [10, 15, 20, 25]
    k_factor = 4
    att_npkts = [flow_table_sizes[i]+10*i for i in range(len(flow_table_sizes))]
    att_anps = [5, 10, 20, 30]
    att_wait = None
    log_timestamp = int(time())

    for i in range(len(pcap_prefixes)):
        topo = topos[i%len(topos)]
        flow_table_size = flow_table_sizes[i%len(flow_table_sizes)]
        idle_timeout = idle_timeouts[i%len(idle_timeouts)]
        hard_timeout = hard_timeouts[i%len(hard_timeouts)]
        npkts = att_npkts[i%len(att_npkts)]
        anp = att_anps[i%len(att_anps)]
        att_duration = -1

        info("*** CONTROLLER\n")
        controller_envs = {
            "GROUP_ID": f"{i}",
            "LABEL": "1",
            "DATASET_PATH": f"{current}/logs/dataset_{log_timestamp}.csv",
            "FLOW_TABLE_SIZE": f"{flow_table_size}",
            "IDLE_TIMEOUT": f"{idle_timeout}",
            "HARD_TIMEOUT": f"{hard_timeout}",
            "K_FACTOR": f"{k_factor}"
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
                autoStaticArp=True)
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
                    pcap_file = f"{pcap_prefixes[i]}_{ip.replace('.', '_')}.pcap"
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
                net=net
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

        sleep(20)

        info("*** CONTROLLER\n")
        controller_envs = {
            "GROUP_ID": f"{i}",
            "LABEL": "0",
            "DATASET_PATH": f"{current}/logs/dataset_{log_timestamp}.csv",
            "FLOW_TABLE_SIZE": f"{flow_table_size}",
            "IDLE_TIMEOUT": f"{idle_timeout}",
            "HARD_TIMEOUT": f"{hard_timeout}",
            "K_FACTOR": f"{k_factor}"
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
                autoStaticArp=True)
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
                    pcap_file = f"{pcap_prefixes[i]}_{ip.replace('.', '_')}.pcap"
                    offset = offsets.get(ip, 0.0)
                    t = threading.Thread(
                        target=replay_worker,
                        args=(h, pcap_file, launch_time, att_duration, offset, procs, stop_event)
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
            for t in threads:
                t.join()

            info(f"*** END TRAFFIC\n")

        finally:
            sleep(60)
            net.stop()
            stop_controller(controller)
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)
        
        sleep(20)

if __name__ == '__main__':
    setLogLevel('info')
    main()
