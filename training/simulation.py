#!/usr/bin/python
import json
import threading
import subprocess
import socket
import sys
import os
import math
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.topo import Topo
from mininet.log import setLogLevel, info

from attack.ftf_attack import FTFAttack

class CustomTopo1(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h3')
    att_hosts = ('h4', 'h3')
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
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)
        self.addLink(h4, s2)

class CustomTopo2(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h8')
    att_hosts = ('h9', 'h8')
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
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s3)
        self.addLink(h4, s3)
        self.addLink(h5, s4)
        self.addLink(h6, s4)
        self.addLink(h7, s4)
        self.addLink(h8, s5)
        self.addLink(h9, s5)

class CustomTopo3(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h5')
    att_hosts = ('h3', 'h4')
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
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s3)
        self.addLink(h4, s3)
        self.addLink(h5, s4)
        self.addLink(h6, s4)
        self.addLink(h7, s2)
        self.addLink(h8, s2)

class CustomTopo4(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h9')
    att_hosts = ('h7', 'h8')
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
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s2)
        self.addLink(h5, s2)
        self.addLink(h6, s2)
        self.addLink(h7, s3)
        self.addLink(h8, s3)
        self.addLink(h9, s3)

class CustomTopo5(Topo):
    "Custom Topology Network"
    conn_hosts = ('h4', 'h5')
    att_hosts = ('h2', 'h3')
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
        self.addLink(s1, s5)
        self.addLink(s3, s6)
        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        self.addLink(h1, s5)
        self.addLink(h2, s2)
        self.addLink(h3, s2)
        self.addLink(h4, s6)
        self.addLink(h5, s4)

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

def replay_worker(host, pcap_file, launch_time, pps, duration, offset, proc_holder, stop_event):
    sleep_duration = (launch_time + offset) - time()
    if sleep_duration > 0:
        sleep(sleep_duration)
    if stop_event.is_set():
        return
    iface = host.defaultIntf().name
    if duration > 0:
        cmd = f"tcpreplay --intf1={iface} --loop=0 --pps={pps} --duration={duration} {pcap_file}"
    else:
        cmd = f"tcpreplay --intf1={iface} --loop=0 --pps={pps} {pcap_file}"
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

def group_duration(packet_counts, pps):
    durations_per_host = {ip: count / pps for ip, count in packet_counts.items()}
    implied_duration = max(durations_per_host.values())
    return implied_duration, durations_per_host

def rescale_offset(raw_offset, original_capture_duration, group_duration):
    if original_capture_duration <= 0:
        return 0.0
    fraction = raw_offset / original_capture_duration
    return min(fraction, 1.0) * group_duration

def estimate_timeout_inference(idle_timeout: int, hard_timeout: int):
    res = 0
    s = 0
    for t in range(1,idle_timeout+1):
        s += t
        if s >= hard_timeout:
            res += (s+t)
            s = 0
    return res + hard_timeout

def main() -> None:
    # Topologies
    topos = [
        CustomTopo1(),
        CustomTopo2(),
        #CustomTopo3(),
        #CustomTopo4(),
        #CustomTopo5()
    ]

    pcap_prefixes = [
        "/home/ubuntu/Desktop/progetto/training/pcaps/topo1/traffic",
        "/home/ubuntu/Desktop/progetto/training/pcaps/topo2/traffic"
    ]
    flow_table_sizes = [60, 150]
    idle_timeouts = [10, 5]
    hard_timeouts = [20, 15]
    pps = [120, 300]

    att_npkts = [70, 170]
    att_anps = [10, 15]
    att_wait = None
    log_timestamp = int(time())

    n_topos = len(topos)
    for i in range(n_topos):
        att_duration = -1

        info("*** CONTROLLER\n")
        controller_envs = {
            "GROUP_ID": f"{i}",
            "LABEL": "1",
            "DATASET_PATH": f"/tmp/dataset_{log_timestamp}.csv",
            "FLOW_TABLE_SIZE": f"{flow_table_sizes[i]}",
            "IDLE_TIMEOUT": f"{idle_timeouts[i]}",
            "HARD_TIMEOUT": f"{hard_timeouts[i]}",
        }
        controller = start_controller(controller_envs, log_path=f"/tmp/controller_{i}_{log_timestamp}.log")

        if controller.poll() is not None:
            raise RuntimeError(f"Il controller è terminato subito con codice {controller.returncode}, controlla controller.log")

        if not wait_for_controller(port=6653, timeout=20):
            controller.terminate()
            raise RuntimeError("Timeout: il controller non si è messo in ascolto sulla porta 6653")

        info("*** NETWORK\n")
        try:
            net = Mininet(
                topo=topos[i],
                controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
                switch=OVSSwitch,
                autoStaticArp=True)
            net.start()

            info("*** NETWORK CONFIG\n")
            ft_envs = {
                "SWITCHES": " ".join([sw.name for sw in net.switches]),
                "FLOW_TABLE_SIZE": f"{flow_table_sizes[i]}",
            }
            run_script('./set_ft_size.sh', ft_envs)
                
            info("*** START TRAFFIC\n")

            metadata = load_metadata(pcap_prefixes[i])
            offsets = metadata["offsets"]
            original_capture_duration = metadata["original_capture_duration"]
            packet_counts = metadata["packet_counts"]
            implied_duration, durations_per_host = group_duration(packet_counts, pps[i])

            launch_time = time() + 1.0
            procs = {}
            stop_event = threading.Event()
            threads = []

            (attacker, dst) = topos[i].att_hosts
            for h in net.hosts:
                if h.name != attacker:
                    ip = h.IP()
                    pcap_file = f"{pcap_prefixes[i]}_{ip.replace('.', '_')}.pcap"
                    offset = rescale_offset(offsets.get(ip, 0.0), original_capture_duration, implied_duration)
                    t = threading.Thread(
                        target=replay_worker,
                        args=(h, pcap_file, launch_time, pps[i], -1, offset, procs, stop_event)
                    )
                    threads.append(t)

            info(f"*** [FTF ATTACK] ({attacker} -> {dst}), NPkts={att_npkts[i]}, ANP={att_anps[i]}\n")
            attack = FTFAttack(
                attacker=attacker,
                probe_dst=dst,
                npacket=att_npkts[i],
                added_new_packets=att_anps[i],
                max_iter=10,
                max_interval=30,
                net=net
            )

            (src, dst) = topos[i].conn_hosts
            h_src, h_dst = net.get(src, dst)
            wait_for_network(h_src, h_dst)

            sleep(hard_timeouts[i])

            # Start threads
            t0 = time()

            for t in threads:
                t.start()
            attack.start()

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

        sleep(10)

        info("*** CONTROLLER\n")
        controller_envs = {
            "GROUP_ID": f"{i}",
            "LABEL": "0",
            "DATASET_PATH": f"/tmp/dataset_{log_timestamp}.csv",
            "FLOW_TABLE_SIZE": f"{flow_table_sizes[i]}",
            "IDLE_TIMEOUT": f"{idle_timeouts[i]}",
            "HARD_TIMEOUT": f"{hard_timeouts[i]}",
        }
        controller = start_controller(controller_envs, log_path=f"/tmp/controller_{i}_{log_timestamp}.log")

        if controller.poll() is not None:
            raise RuntimeError(f"Il controller è terminato subito con codice {controller.returncode}, controlla controller.log")

        if not wait_for_controller(port=6653, timeout=20):
            controller.terminate()
            raise RuntimeError("Timeout: il controller non si è messo in ascolto sulla porta 6653")

        info("*** NETWORK\n")
        try:
            net = Mininet(
                topo=topos[i],
                controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
                switch=OVSSwitch,
                autoStaticArp=True)
            net.start()

            info("*** NETWORK CONFIG\n")
            ft_envs = {
                "SWITCHES": " ".join([sw.name for sw in net.switches]),
                "FLOW_TABLE_SIZE": f"{flow_table_sizes[i]}",
            }
            run_script('./set_ft_size.sh', ft_envs)
                
            info("*** START TRAFFIC\n")

            metadata = load_metadata(pcap_prefixes[i])
            offsets = metadata["offsets"]
            original_capture_duration = metadata["original_capture_duration"]
            packet_counts = metadata["packet_counts"]
            implied_duration, durations_per_host = group_duration(packet_counts, pps[i])

            launch_time = time() + 1.0
            procs = {}
            stop_event = threading.Event()
            threads = []

            (attacker, dst) = topos[i].att_hosts
            for h in net.hosts:
                if h.name != attacker:
                    ip = h.IP()
                    pcap_file = f"{pcap_prefixes[i]}_{ip.replace('.', '_')}.pcap"
                    offset = rescale_offset(offsets.get(ip, 0.0), original_capture_duration, implied_duration)
                    t = threading.Thread(
                        target=replay_worker,
                        args=(h, pcap_file, launch_time, pps[i], att_duration, offset, procs, stop_event)
                    )
                    threads.append(t)

            (src, dst) = topos[i].conn_hosts
            h_src, h_dst = net.get(src, dst)
            wait_for_network(h_src, h_dst)

            sleep(hard_timeouts[i])

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
        
        sleep(10)

if __name__ == '__main__':
    setLogLevel('info')
    main()
