#!/usr/bin/python
import threading
import subprocess
import math
import csv
from time import time, sleep
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.log import setLogLevel, info

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from training.topology import *
from utils.sim_utils import *
from attack.ftf_attack import FTFAttack

import random
rng = random.Random(67)

SET_FT_SIZE_PATH = root + "/config/set_ft_size.sh"

def sample_attack_rate(npkts, flow_table_size, min_anp={"aggressive": 10,"moderate": 7,"stealth": 2}):
    stealth_profiles = {
        "aggressive": (10, 60),
        "moderate":   (120, 300),
        "stealth":    (350, 550),
    }
    profile = rng.choice(list(stealth_profiles.keys()))
    t_min, t_max = stealth_profiles[profile]
    fill_time = rng.uniform(t_min, t_max)
    anp = max(min_anp[profile], math.ceil(flow_table_size / fill_time))
    return anp, profile, fill_time

def main() -> None:
    # Topologies
    # Topologies
    topos = [
        CustomTopo1(),
        CustomTopo2(),
        CustomTopo3(),
        CustomTopo4(),
        CustomTopo5(),
        CustomTopo6()
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
        #current + "/pcaps/traffics/scenario_048/",
    ]
    
    att_wait = None
    k_factor = 2
    n_host = 15
    full_size = int(math.factorial(n_host) / math.factorial(n_host-2)) + 1
    half_size = int(full_size / 2)
    log_timestamp = int(time())

    csv_path = current + f"/logs/network_settings_{log_timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topo", "hard_timeout", "idle_timeout", "flow_table_size", "npkts", "profile", "anp", "att_duration", "fill_time"])

    for i in range(0, 34):
        # hard_timeout: 5 < hard <= 30
        hard_timeout = rng.randint(6, 15)
        # idle_timeout: idle <= hard
        idle_timeout = rng.randint(2, hard_timeout)
        # flow_table_size
        flow_table_size = rng.randint(half_size, 2*full_size)
        # att_npkts: att_npkts >= flow_table_size
        npkts = flow_table_size + rng.randint(0, half_size)
        # att_anp: 1 min - ~10 min
        anp, profile, fill_time = sample_attack_rate(npkts, flow_table_size)

    for i in range(34, 44):
        topo = topos[i%len(topos)]
        # hard_timeout: 5 < hard <= 30
        hard_timeout = rng.randint(6, 30)
        # idle_timeout: idle <= hard
        idle_timeout = rng.randint(2, hard_timeout)
        # flow_table_size
        flow_table_size = rng.randint(half_size, 2*full_size)
        # att_npkts: att_npkts >= flow_table_size
        npkts = flow_table_size + rng.randint(0, half_size)
        # att_anp: 1 min - ~10 min
        anp, profile, fill_time = sample_attack_rate(npkts, flow_table_size)
        
        # att_duration
        att_duration = -1

        info("*** CONTROLLER\n")
        controller_envs = {
            "SCENARIO_ID": f"{i}",
            "FTF_LABEL": "1",
            "DATASET_FTF_PATH": f"{current}/logs/dataset_ftf.csv",
            "DATASET_MFR_PATH": f"{current}/logs/dataset_mfr.csv",
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
                local_ip=False if i%2 else True,
            )

            (src, dst) = topo.conn_hosts
            h_src, h_dst = net.get(src, dst)
            wait_for_network(h_src, h_dst)

            sleep(hard_timeout)

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
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)

        sleep(30)

        info("*** CONTROLLER\n")
        controller_envs = {
            "SCENARIO_ID": f"{i}",
            "FTF_LABEL": "0",
            "DATASET_FTF_PATH": f"{current}/logs/dataset_ftf.csv",
            "DATASET_MFR_PATH": f"{current}/logs/dataset_mfr.csv",
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
