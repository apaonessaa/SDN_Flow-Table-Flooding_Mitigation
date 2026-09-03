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

class CustomTopo(Topo):
    "Custom Topology Network"
    conn_hosts = ('h1', 'h2')
    att_hosts = ('h5', 'h4')
    def build(self):
        info("* Build Topology\n")
        # Switch
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow14')
        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        self.addLink(h1, s1)
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        self.addLink(h2, s1)
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        self.addLink(h3, s1)
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        self.addLink(h4, s1)
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        self.addLink(h5, s1)

def load_metadata(prefix):
    with open(f"{prefix}_metadata.json") as f:
        return json.load(f)

def start_controller(env_vars: dict, log_path: str = "controller.log") -> subprocess.Popen:
    log_file = open(log_path, "w+")
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
    pcap_prefix = current + "/pcaps/traffic"
    topo = CustomTopo()
    flow_table_size = 10
    idle_timeout = 10
    hard_timeout = 20
    k_factor = 3
    npkts = 20
    anp = 5
    att_wait = None
    att_duration = -1
    log_path = current + "/logs"
    log_timestamp = int(time())

    info("*** CONTROLLER\n")
    controller_envs = {
        "DATASET_PATH": f"{log_path}/dataset_{log_timestamp}.csv",
        "FLOW_TABLE_SIZE": f"{flow_table_size}",
        "IDLE_TIMEOUT": f"{idle_timeout}",
        "HARD_TIMEOUT": f"{hard_timeout}",
        "K_FACTOR": f"{k_factor}"
    }
    controller = start_controller(controller_envs, log_path=f"{log_path}/controller_{log_timestamp}.log")

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

        metadata = load_metadata(pcap_prefix)
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
                pcap_file = f"{pcap_prefix}_{ip.replace('.', '_')}.pcap"
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
            t.join(timeout=5) #t.join() #  #

        att_duration = math.ceil(time() - t0)
        info(f"*** END TRAFFIC: Session duration {att_duration}\n")

    finally:
        sleep(30)
        net.stop()
        stop_controller(controller)
        subprocess.run(['sudo', 'mn', '-c'], capture_output=True)

if __name__ == '__main__':
    setLogLevel('info')
    main()
