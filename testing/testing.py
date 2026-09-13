#!/usr/bin/python
import threading
import subprocess
import math
import random
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.topo import Topo
from mininet.log import setLogLevel, info

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils.sim_utils import *
from attack.ftf_attack import FTFAttack

SET_FT_SIZE_PATH = root + "/config/set_ft_size.sh"

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

def main() -> None:
    pcap_prefix = current + "/pcaps/"
    topo = CustomTopo()
    flow_table_size = 100
    idle_timeout = 5
    hard_timeout = 10
    k_factor = 4
    npkts = 120
    anp = 10
    log_path = current + "/logs"
    log_timestamp = int(time())

    info("*** CONTROLLER\n")
    controller_envs = {
        "REPORT_PATH": f"{log_path}/report_{log_timestamp}.txt",
        "FLOW_TABLE_SIZE": f"{flow_table_size}",
        "IDLE_TIMEOUT": f"{idle_timeout}",
        "HARD_TIMEOUT": f"{hard_timeout}",
        "K_FACTOR": f"{k_factor}",
        "MODEL": "CatBoost"
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

        info(f"Traffic packet count: {packet_counts}\n")

        launch_time = time() + 1.0
        procs = {}
        stop_event = threading.Event()
        threads = []

        (attacker, dst) = topo.att_hosts
        for h in net.hosts:
            if h.name != attacker:
                ip = h.IP()
                pcap_file = f"{pcap_prefix}traffic_{ip.replace('.', '_')}.pcap"
                offset = offsets.get(ip, 0.0)
                t = threading.Thread(
                    target=replay_worker,
                    args=(h, pcap_file, launch_time, -1, offset, procs, stop_event)
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

        #attack.start()

        #sleep(random.randint(30, 60))
        for t in threads:
            t.start()

        # Stop threads
        #attack.join()
        #sleep(random.randint(30, 60))
        #kill_replay_traffic(procs, stop_event)
        sleep(90)
        for t in threads:
            t.join(timeout=5)

        info(f"*** END TRAFFIC: Session duration {math.ceil(time() - t0)}\n")

    finally:
        sleep(30)
        net.stop()
        stop_controller(controller)
        subprocess.run(['sudo', 'mn', '-c'], capture_output=True)

if __name__ == '__main__':
    setLogLevel('info')
    main()
