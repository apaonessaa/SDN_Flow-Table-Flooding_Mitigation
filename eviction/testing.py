#!/usr/bin/python
import argparse
import json
import threading
import subprocess
import math
import random
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.topo import Topo
from mininet.log import setLogLevel, info
from mininet.cli import CLI

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils.sim_utils import *
from utils.resource_monitor import ResourceMonitor, find_ryu, find_by_name, pin_process
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


def parse_cpus(spec):
    """'0' -> {0}; '0,2' -> {0, 2}; '0-1' -> {0, 1}"""
    cpus = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


def parse_args():
    ap = argparse.ArgumentParser(description="Test online FTF detection/mitigation")
    ap.add_argument("--no-cli", action="store_true",
                    help="non aprire la CLI di Mininet a fine traffico (run automatiche)")
    ap.add_argument("--idle", type=float, default=10.0,
                    help="secondi di baseline senza traffico prima dei replay (min: hard_timeout)")
    ap.add_argument("--controller-cpus", default=None,
                    help="core su cui fissare il controller (es. '0'); il resto della "
                         "simulazione viene spostato sui core rimanenti")
    ap.add_argument("--log-level", default="INFO",
                    help="livello di log del controller (INFO, WARNING, DEBUG)")
    return ap.parse_args()


def main(args) -> None:
    pcap_prefix = current + "/pcaps/"
    topo = CustomTopo()
    flow_table_size = 100
    idle_timeout = 2
    hard_timeout = 5
    k_factor = 2
    npkts = 230
    anp = 10
    log_path = current + "/logs"
    os.makedirs(log_path, exist_ok=True)
    log_timestamp = int(time())

    info("*** CONTROLLER\n")
    controller_envs = {
        "REPORT_PATH": f"{log_path}/report_{log_timestamp}.txt",
        "FLOW_TABLE_SIZE": f"{flow_table_size}",
        "IDLE_TIMEOUT": f"{idle_timeout}",
        "HARD_TIMEOUT": f"{hard_timeout}",
        "K_FACTOR": f"{k_factor}",
        "MAX_COLLATERAL_PORT": "0",
        "LOG_LEVEL": args.log_level,
    }
    controller = start_controller(controller_envs, log_path=f"{log_path}/controller_{log_timestamp}.log")

    if controller.poll() is not None:
        raise RuntimeError(f"Il controller è terminato subito con codice {controller.returncode}, controlla controller.log")

    if not wait_for_controller(port=6653, timeout=20):
        controller.terminate()
        raise RuntimeError("Timeout: il controller non si è messo in ascolto sulla porta 6653")

    # Processo ryu-manager vero (controller.pid puo' essere quello di uno shell/sudo)
    ryu_proc = find_ryu(controller)

    # Pinning opzionale: controller su alcuni core, tutto il resto (host
    # Mininet, replay, attacco, questo script) sugli altri. Va fatto PRIMA di
    # creare la rete, perche' i processi figli ereditano l'affinita'.
    if args.controller_cpus:
        ctl_cpus = parse_cpus(args.controller_cpus)
        others = set(os.sched_getaffinity(0)) - ctl_cpus
        if others:
            pin_process(ryu_proc, ctl_cpus)
            os.sched_setaffinity(0, others)
            info(f"*** PINNING: controller su {sorted(ctl_cpus)}, resto su {sorted(others)}\n")
        else:
            info("*** PINNING ignorato: nessun core rimane libero per il resto della simulazione\n")

    # Metadati della run: rendono ogni set di CSV autoesplicativo
    with open(f"{log_path}/meta_{log_timestamp}.json", "w") as f:
        json.dump({
            "run_id": log_timestamp,
            "flow_table_size": flow_table_size, "idle_timeout": idle_timeout,
            "hard_timeout": hard_timeout, "k_factor": k_factor,
            "npkts": npkts, "anp": anp,
            "idle_baseline_s": max(hard_timeout, args.idle),
            "controller_cpus": args.controller_cpus,
            "controller_envs": controller_envs,
            "ryu_pid": ryu_proc.pid,
        }, f, indent=2)

    net = None
    mon = None
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
            net=net,
            local_ip=True
        )

        (src, dst) = topo.conn_hosts
        h_src, h_dst = net.get(src, dst)
        wait_for_network(h_src, h_dst)

        # Il monitor parte DOPO wait_for_network (che genera flow di test) e
        # dopo aver atteso la loro scadenza: la baseline e' un controller a riposo.
        # OVS si cerca qui perche' e' un servizio di sistema, non un figlio dello script.
        mon = ResourceMonitor(
            {"ryu": ryu_proc, "ovs": find_by_name("ovs-vswitchd")},
            f"{log_path}/resources_{log_timestamp}.csv",
        )
        mon.start()
        mon.mark("idle_start")
        sleep(max(hard_timeout, args.idle))

        # Start threads
        t0 = time()

        mon.mark("attack_start")
        attack.start()

        #sleep(random.randint(30, 60))
        mon.mark("benign_start")
        for t in threads:
            t.start()

        # Stop threads
        attack.join()
        mon.mark("attack_end")
        sleep(30)
        kill_replay_traffic(procs, stop_event)
        for t in threads:
            t.join(timeout=5)
        mon.mark("traffic_end")

        info(f"*** END TRAFFIC: Session duration {math.ceil(time() - t0)}\n")

        # Il monitor si ferma prima della CLI: il tempo passato al prompt
        # non fa parte della misura.
        mon.stop()

        if not args.no_cli:
            CLI(net)

    finally:
        if mon is not None:
            mon.stop()
        sleep(30)
        if net is not None:
            net.stop()
        stop_controller(controller)
        subprocess.run(['sudo', 'mn', '-c'], capture_output=True)

if __name__ == '__main__':
    setLogLevel('info')
    main(parse_args())
