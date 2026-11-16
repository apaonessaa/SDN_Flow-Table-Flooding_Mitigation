#!/usr/bin/python
"""
Test online FTF: confronto FTFM vs baseline.

Singola run:
    sudo python3 test_online.py --controller ftfm --no-cli --warmup 60 --isolate
Campagna di confronto (N coppie, ordine alternato ftfm/baseline, poi analisi):
    sudo python3 test_online.py --pairs 5 --warmup 60 --isolate

File scritti in logs/<controller>/ (prefisso <id> = timestamp della run):
    meta_<id>.json           parametri, ambiente, affinity reale, switch vittima, coppie di probe
    events_<id>.csv          marker di fase
    resources_<id>.csv       CPU%/RAM per processo (ResourceMonitor, 0.5 s)
    cpuacct_<id>.csv         CPU-secondi CUMULATIVI, context switch, steal della VM (1 s)  [NUOVO]
    portstats_<id>.csv       contatori di tutte le porte degli switch (carico offerto)    [NUOVO]
    setup_<id>.csv           latenza di setup di flow NUOVE (probe a freddo)              [NUOVO]
    network_<id>_*.csv       flow rule e ping (NetworkMonitor)
    ftf_<id>.csv, report_<id>.txt, controller_<id>.log
"""
import argparse
import csv
import hashlib
import json
import math
import threading
import subprocess
import random
import platform
import re
rng = random.Random(69)

from time import time, sleep
import psutil
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.topo import Topo
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from topology import *

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils.sim_utils import *
from resource_monitor import ResourceMonitor, find_ryu, find_by_name, pin_process
from network_monitor import NetworkMonitor
from test_attack.ftf_attack import FTFAttack
from topology import TestOnlineTopology

SET_FT_SIZE_PATH = root + "/config/set_ft_size.sh"

CONTROLLER_APPS = {
    "ftfm": "controller.py",
    "baseline": "baseline_controller.py",
}

_RTT_RE = re.compile(r'time=([\d.]+) ms')


# ---------------------------------------------------------------------------
# Utilita'
# ---------------------------------------------------------------------------
class EventLog:
    """Marker di fase, scritti subito su disco (resistono a un crash)."""
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["ts", "event"])

    def mark(self, label):
        with self._lock, open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([time(), label])
        info(f"[event] {label}\n")


def write_json(path, d):
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)


def file_sha1(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def env_info():
    d = {"cpu_count": os.cpu_count(), "platform": platform.platform(),
         "python": platform.python_version(), "main_affinity": sorted(os.sched_getaffinity(0))}
    try:
        d["ovs_version"] = subprocess.run(["ovs-vswitchd", "--version"], capture_output=True,
                                          text=True, timeout=5).stdout.splitlines()[0]
    except Exception:
        pass
    try:
        import ryu
        d["ryu_version"] = ryu.__version__
    except Exception:
        pass
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            d["cpu_governor"] = f.read().strip()
    except OSError:
        pass
    try:
        d["cpu_mhz"] = round(psutil.cpu_freq().current)
    except Exception:
        pass
    return d


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


def plan_cpus(args):
    """(core controller, core ovs). --isolate: controller su 0, ovs su 1, resto altrove."""
    ctl = parse_cpus(args.controller_cpus) if args.controller_cpus else None
    ovs = parse_cpus(args.ovs_cpus) if args.ovs_cpus else None
    if args.isolate:
        if (os.cpu_count() or 1) >= 4:
            ctl, ovs = ctl or {0}, ovs or {1}
        else:
            info("*** --isolate ignorato: servono almeno 4 core\n")
    return ctl, ovs


def affinity_masks(proc):
    """Maschere di affinity distinte dei thread del processo (verifica del pinning)."""
    masks = set()
    try:
        for t in proc.threads():
            try:
                masks.add(tuple(sorted(os.sched_getaffinity(t.id))))
            except OSError:
                pass
    except psutil.Error:
        pass
    return [list(m) for m in sorted(masks)]


def host_switch_map(net):
    """{host: switch a cui e' collegato}, ricavato dai link reali."""
    m = {}
    for h in net.hosts:
        link = h.defaultIntf().link
        if link is None:
            continue
        other = link.intf2 if link.intf1.node is h else link.intf1
        m[h.name] = other.node.name
    return m


def victim_switch_name(topo, sw_of, attacker):
    v = getattr(topo, "victim_dpid", None)
    if v is not None:
        s = str(v).lower()
        if s.startswith("s") and s[1:].isdigit():
            return s
        if s.isdigit():
            return f"s{int(s)}"
    info("*** ATTENZIONE: topo.victim_dpid non utilizzabile, uso lo switch dell'attaccante\n")
    return sw_of.get(attacker, "s2")


def build_probe_pairs(switch_names, sw_of, victim, excluded, n_per_group, prng):
    """Coppie (src, dst, gruppo) per i probe di setup. 'victim' = il percorso passa dallo
    switch vittima (topologia a linea, ordine dei nomi); 'control' = non ci passa."""
    order = {s: i for i, s in enumerate(sorted(switch_names, key=lambda x: (len(x), x)))}

    def touches(a, b):
        lo, hi = sorted((order[sw_of[a]], order[sw_of[b]]))
        return lo <= order[victim] <= hi

    hosts = [h for h in sw_of if h not in excluded]
    vic, ctl = [], []
    for a in hosts:
        for b in hosts:
            if a != b:
                (vic if touches(a, b) else ctl).append((a, b))
    vic = prng.sample(vic, min(n_per_group, len(vic)))
    ctl = prng.sample(ctl, min(n_per_group, len(ctl)))
    pairs = []
    for i in range(max(len(vic), len(ctl))):
        if i < len(vic):
            pairs.append((vic[i][0], vic[i][1], "victim"))
        if i < len(ctl):
            pairs.append((ctl[i][0], ctl[i][1], "control"))
    return pairs


# ---------------------------------------------------------------------------
# Strumenti di misura aggiuntivi
# ---------------------------------------------------------------------------
class CounterSampler(threading.Thread):
    """Ogni `interval` s: (1) contatori di TUTTE le porte degli switch (lato root
    namespace, quindi senza toccare le shell degli host): da qui carico offerto
    dall'attaccante, traffico benigno consegnato, drop; (2) CPU-secondi cumulativi
    di ryu/ovs, context switch e tempi di sistema incl. steal. I CPU-secondi
    cumulativi non dipendono dal campionamento di cpu_percent()."""
    PORT_STATS = ("rx_packets", "rx_bytes", "tx_packets", "tx_bytes", "rx_dropped", "tx_dropped")

    def __init__(self, net, procs, port_path, cpu_path, interval=1.0):
        super().__init__(daemon=True)
        self.procs = {n: p for n, p in procs.items() if p is not None}
        self.port_path, self.cpu_path, self.interval = port_path, cpu_path, interval
        self.ports = []                      # (switch, intf, peer)
        for l in net.links:
            for a, b in ((l.intf1, l.intf2), (l.intf2, l.intf1)):
                if a.node in net.switches:
                    self.ports.append((a.node.name, a.name, b.node.name))
        self._halt = threading.Event()

    @staticmethod
    def _stat(intf, name):
        try:
            with open(f"/sys/class/net/{intf}/statistics/{name}") as f:
                return int(f.read())
        except (OSError, ValueError):
            return -1

    def run(self):
        with open(self.port_path, "w", newline="") as fp, open(self.cpu_path, "w", newline="") as fc:
            wp, wc = csv.writer(fp), csv.writer(fc)
            wp.writerow(["ts", "switch", "intf", "peer"] + list(self.PORT_STATS))
            wc.writerow(["ts", "name", "user_s", "sys_s", "idle_s", "iowait_s", "steal_s",
                         "ctx_vol", "ctx_invol", "threads"])
            nxt = time()
            while not self._halt.is_set():
                ts = round(time(), 3)
                for sw, intf, peer in self.ports:
                    wp.writerow([ts, sw, intf, peer] + [self._stat(intf, s) for s in self.PORT_STATS])
                for name, p in self.procs.items():
                    try:
                        t, c = p.cpu_times(), p.num_ctx_switches()
                        wc.writerow([ts, name, t.user, t.system, "", "", "",
                                     c.voluntary, c.involuntary, p.num_threads()])
                    except psutil.Error:
                        pass
                st = psutil.cpu_times()
                g = lambda k: getattr(st, k, 0.0)
                wc.writerow([ts, "system", g("user") + g("nice"), g("system") + g("irq") + g("softirq"),
                             g("idle"), g("iowait"), g("steal"), "", "", ""])
                fp.flush()
                fc.flush()
                nxt += self.interval
                self._halt.wait(max(0.0, nxt - time()))

    def stop(self):
        self._halt.set()
        if self.is_alive():
            self.join(timeout=5)


class SetupProbe(threading.Thread):
    """Latenza di setup di flow NUOVE. Ogni coppia viene sondata con un solo ping ogni
    `period` s (> hard_timeout): la flow e' scaduta, quindi il primo pacchetto passa dal
    controller (packet-in + flow-mod, in entrambe le direzioni). ARP e' statico, quindi
    il tempo misurato e' quello del controller. Gruppi: 'victim' (percorso via switch
    attaccato) e 'control' (non ci passa): la differenza isola l'effetto della tabella
    piena dal rumore della VM. Se una flow e' tenuta calda dal replay l'RTT e' ~0.05 ms
    (si vede e si filtra in analisi)."""

    def __init__(self, net, pairs, out_path, period, timeout=2):
        super().__init__(daemon=True)
        self.net, self.pairs, self.period, self.timeout = net, pairs, period, timeout
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._workers = []
        self._closed = False
        self._f = open(out_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["ts", "src", "dst", "group", "rtt_ms", "lost"])

    def run(self):
        if not self.pairs:
            return
        step = self.period / len(self.pairs)
        nxt, i = time(), 0
        while not self._halt.is_set():
            s, d, g = self.pairs[i % len(self.pairs)]
            w = threading.Thread(target=self._probe, args=(s, d, g), daemon=True)
            w.start()
            self._workers.append(w)
            i += 1
            nxt += step
            self._halt.wait(max(0.0, nxt - time()))

    def _probe(self, s, d, g):
        ts, rtt, p = time(), None, None
        try:
            h, dst = self.net.get(s), self.net.get(d)
            p = h.popen(["ping", "-c", "1", "-n", "-W", str(self.timeout), dst.IP()],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out, _ = p.communicate(timeout=self.timeout + 3)
            m = _RTT_RE.search(out)
            rtt = float(m.group(1)) if m else None
        except Exception:
            try:
                if p is not None:
                    p.kill()
            except Exception:
                pass
        with self._lock:
            if not self._closed:
                self._w.writerow([round(ts, 3), s, d, g, "" if rtt is None else rtt, int(rtt is None)])
                self._f.flush()

    def stop(self):
        self._halt.set()
        if self.is_alive():
            self.join(timeout=5)
        for w in self._workers:
            w.join(timeout=self.timeout + 4)
        with self._lock:
            if not self._closed:
                self._closed = True
                self._f.close()


# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Test online FTF detection/mitigation")
    ap.add_argument("--controller", choices=sorted(CONTROLLER_APPS), default="ftfm",
                    help="controller sotto test: 'ftfm' o 'baseline' (learning switch)")
    ap.add_argument("--no-cli", action="store_true", help="non aprire la CLI a fine traffico")
    ap.add_argument("--idle", type=float, default=10.0,
                    help="secondi senza traffico prima dei replay (min: hard_timeout)")
    ap.add_argument("--warmup", type=float, default=60.0,
                    help="secondi di solo traffico benigno prima dell'attacco (>= 55 per avere "
                         "30 s di fase benigna pulita nell'analisi)")
    ap.add_argument("--tail", type=float, default=30.0,
                    help="secondi di traffico dopo la fine dell'attacco (misura il recupero)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--ft-threshold", type=float, default=0.8,
                    help="FLOW_TABLE_THRESHOLD del controller FTFM (default 0.8, come nel test precedente)")
    ap.add_argument("--valid-match", choices=("src", "any"), default="src",
                    help="una flow e' 'valida' se il MAC reale e' in dl_src (src) o ovunque (any)")
    ap.add_argument("--probe-pairs", type=int, default=8,
                    help="coppie per gruppo (victim/control) del probe di setup")
    ap.add_argument("--seed", type=int, default=69)
    # isolamento delle risorse
    ap.add_argument("--controller-cpus", default=None, help="core del controller (es. '0')")
    ap.add_argument("--ovs-cpus", default=None, help="core di ovs-vswitchd (es. '1')")
    ap.add_argument("--isolate", action="store_true",
                    help="auto: controller su core 0, OVS su core 1, resto sugli altri (>= 4 core)")
    # campagna
    ap.add_argument("--run-id", type=int, default=None, help="(interno) id della run")
    ap.add_argument("--pairs", type=int, default=0,
                    help="se > 0: esegue N coppie ftfm/baseline con ordine alternato e analizza")
    ap.add_argument("--cooldown", type=float, default=20.0, help="pausa tra le run di una campagna")
    return ap.parse_args()


def sample_attack_rate(npkts, flow_table_size, profile):
    attack_profiles = {
        "aggressive": (10, 60),
        "moderate":   (120, 300),
        "stealth":    (350, 550),
    }
    min_anp = {"aggressive": 10, "moderate": 7, "stealth": 2}
    anms = {
        "aggressive": (100, 200),
        "moderate":   (30, 70),
        "stealth":    (5, 20),
    }
    t_min, t_max = attack_profiles[profile]
    fill_time = rng.uniform(t_min, t_max)
    anp = max(min_anp[profile], math.ceil(flow_table_size / fill_time))
    anm = rng.uniform(anms[profile][0], anms[profile][1])
    return anp, int(anm), fill_time


# ---------------------------------------------------------------------------
# Una run
# ---------------------------------------------------------------------------
def main(args) -> None:
    pcap_prefix = current + "/pcaps/scenario_016/"
    topo = TestOnlineTopology()
    flow_table_size = 100
    idle_timeout = 10
    hard_timeout = 15
    k_factor = 2
    npkts = flow_table_size*2
    anp = 20
    anm = 30
    fill_time = None
    profile = "aggressive"
    log_path = current + "/logs/" + args.controller
    os.makedirs(log_path, exist_ok=True)
    log_timestamp = args.run_id or int(time())
    P = lambda name: f"{log_path}/{name}_{log_timestamp}"

    if args.warmup < 55:
        info("*** ATTENZIONE: warmup < 55 s: la fase benigna pulita (dopo lo scarto iniziale "
             "dell'analisi) sara' < 30 s\n")

    info(f"*** CONTROLLER ({args.controller})\n")
    controller_envs = {
        "REPORT_PATH": f"{P('report')}.txt",   # solo ftfm
        "FLOW_TABLE_SIZE": f"{flow_table_size}",
        "IDLE_TIMEOUT": f"{idle_timeout}",
        "HARD_TIMEOUT": f"{hard_timeout}",
        "K_FACTOR": f"{k_factor}",
        "MAX_COLLATERAL_PORT": "10",
        "LOG_LEVEL": args.log_level,
        "FLOW_TABLE_THRESHOLD": f"{args.ft_threshold}",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    controller = start_controller(
        controller_envs,
        log_path=f"{P('controller')}.log",
        app=CONTROLLER_APPS[args.controller], verbose=False
    )

    if controller.poll() is not None:
        raise RuntimeError(f"Il controller è terminato subito con codice {controller.returncode}, controlla controller.log")

    if not wait_for_controller(port=6653, timeout=20):
        controller.terminate()
        raise RuntimeError("Timeout: il controller non si è messo in ascolto sulla porta 6653")

    ryu_proc = find_ryu(controller)

    # Pinning: controller (e opzionalmente OVS) su core dedicati, tutto il resto
    # (host Mininet, replay, attacco, questo script) sugli altri. Va fatto PRIMA di
    # creare la rete: i processi figli ereditano l'affinita'.
    ctl_cpus, ovs_cpus = plan_cpus(args)
    rest_cpus = None
    if ctl_cpus or ovs_cpus:
        rest = set(os.sched_getaffinity(0)) - (ctl_cpus or set()) - (ovs_cpus or set())
        if rest:
            if ctl_cpus:
                pin_process(ryu_proc, ctl_cpus)
            os.sched_setaffinity(0, rest)
            rest_cpus = rest
            info(f"*** PINNING: controller={sorted(ctl_cpus or [])} ovs={sorted(ovs_cpus or [])} "
                 f"resto={sorted(rest)}\n")
        else:
            info("*** PINNING ignorato: nessun core rimane libero per il resto della simulazione\n")
            ctl_cpus = ovs_cpus = None

    meta = {
        "run_id": log_timestamp, "controller": args.controller,
        "flow_table_size": flow_table_size, "idle_timeout": idle_timeout,
        "hard_timeout": hard_timeout, "k_factor": k_factor,
        "npkts": npkts, "anp": anp, "anm": anm,
        "idle_baseline_s": max(hard_timeout, args.idle),
        "controller_cpus": args.controller_cpus, "ovs_cpus": args.ovs_cpus, "isolate": args.isolate,
        "pinning_requested": {"controller": sorted(ctl_cpus or []), "ovs": sorted(ovs_cpus or []),
                              "rest": sorted(rest_cpus or [])},
        "controller_envs": controller_envs,
        "controller_app": CONTROLLER_APPS[args.controller],
        "controller_app_sha1": file_sha1(CONTROLLER_APPS[args.controller]),
        "ryu_pid": ryu_proc.pid,
        "profile": profile, "fill_time": fill_time,
        "warmup": args.warmup, "tail": args.tail, "valid_match": args.valid_match,
        "args": vars(args), "env": env_info(), "started": time(),
    }
    meta_path = f"{log_path}/meta_{log_timestamp}.json"
    write_json(meta_path, meta)

    net = mon = net_mon = sampler = probe = None
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

        ovs_proc = find_by_name("ovs-vswitchd")
        if ovs_cpus and rest_cpus and ovs_proc is not None:
            pin_process(ovs_proc, ovs_cpus)

        # topologia reale: switch vittima, collegamenti host-switch, MAC "validi"
        sw_of = host_switch_map(net)
        attacker, atk_dst = topo.att_hosts
        victim = victim_switch_name(topo, sw_of, attacker)
        probe_pairs = build_probe_pairs([s.name for s in net.switches], sw_of, victim,
                                        {attacker, atk_dst}, args.probe_pairs,
                                        random.Random(args.seed))
        legit = [h for h in net.hosts if h.name != attacker]
        valid_macs = ([f"dl_src={h.MAC()}" for h in legit] if args.valid_match == "src"
                      else [h.MAC() for h in legit])

        info("*** START TRAFFIC\n")
        metadata = load_metadata(pcap_prefix)
        offsets = metadata["offsets"]
        packet_counts = metadata["packet_counts"]
        info(f"Traffic packet count: {packet_counts}\n")

        procs = {}
        stop_event = threading.Event()

        info(f"*** [FTF ATTACK] ({attacker} -> {atk_dst}), NPkts={npkts}, ANP={anp}, ANM={anm}\n")
        attack = FTFAttack(
            attacker=attacker,
            probe_dst=atk_dst,
            npacket=npkts,
            added_new_packets=anp,
            added_new_macs=anm,
            n=1,
            t_max=hard_timeout,
            net=net,
            out_file=f"{P('ftf')}.csv",
            local_ip=True,
        )

        (src, dst) = topo.conn_hosts
        h_src, h_dst = net.get(src, dst)
        wait_for_network(h_src, h_dst)

        # I monitor partono DOPO wait_for_network e dopo l'attesa della scadenza delle flow.
        mon = ResourceMonitor(
            {"ryu": ryu_proc, "ovs": ovs_proc},
            f"{P('resources')}.csv",
        )
        net_mon = NetworkMonitor(
            net, label=args.controller,
            switches=[sw.name for sw in net.switches],
            ping_pairs=topo.ping_pairs,
            host_macs=valid_macs,
            out_prefix=P("network"),
        )
        sampler = CounterSampler(net, {"ryu": ryu_proc, "ovs": ovs_proc},
                                 f"{P('portstats')}.csv", f"{P('cpuacct')}.csv")
        probe = SetupProbe(net, probe_pairs, f"{P('setup')}.csv", period=hard_timeout + 1)

        meta.update({
            "victim_switch": victim, "host_switch": sw_of,
            "attacker": attacker, "attack_dst": atk_dst,
            "ping_pairs": getattr(topo, "ping_pairs", None),
            "probe_pairs": probe_pairs, "probe_period_s": hard_timeout + 1,
            "valid_macs_example": valid_macs[:2],
            "affinity_actual": {"main": sorted(os.sched_getaffinity(0)),
                                "ryu": affinity_masks(ryu_proc),
                                "ovs": affinity_masks(ovs_proc) if ovs_proc else None},
        })
        write_json(meta_path, meta)

        events = EventLog(f"{P('events')}.csv")
        mark = events.mark

        net_mon.start()
        mon.start()
        sampler.start()
        mark("idle_start")
        sleep(max(hard_timeout, args.idle))

        # I replay partono ORA: launch_time va calcolato qui (prima era calcolato prima
        # dell'attesa idle e risultava gia' nel passato di ~15 s).
        launch_time = time() + 1.0
        threads = []
        for h in net.hosts:
            if h.name != attacker:
                ip = h.IP()
                pcap_file = f"{pcap_prefix}traffic_{ip.replace('.', '_')}.pcap"
                offset = offsets.get(ip, 0.0)
                threads.append(threading.Thread(
                    target=replay_worker,
                    args=(h, pcap_file, launch_time, -1, offset, procs, stop_event)))

        t0 = time()
        mark("benign_start")
        for t in threads:
            t.start()
        probe.start()
        sleep(args.warmup)

        mark("attack_start")
        attack.start()
        attack.join()
        mark("attack_end")
        mark(f"attack_worker_rc={attack.returncode}")

        sleep(args.tail)
        kill_replay_traffic(procs, stop_event)
        for t in threads:
            t.join(timeout=5)
        mark("traffic_end")
        info(f"*** END TRAFFIC: Session duration {math.ceil(time() - t0)}\n")

        meta.update({"ended": time(), "attack_rc": attack.returncode,
                     "affinity_end": {"ryu": affinity_masks(ryu_proc),
                                      "ovs": affinity_masks(ovs_proc) if ovs_proc else None}})
        write_json(meta_path, meta)

        # I monitor si fermano prima della CLI: il tempo al prompt non e' misura.
        probe.stop()
        sampler.stop()
        mon.stop()
        net_mon.stop()

        if not args.no_cli:
            CLI(net)

    finally:
        for m in (probe, sampler, net_mon, mon):     # stop() e' idempotente
            if m is not None:
                m.stop()
        sleep(30)
        if net is not None:
            net.stop()
        stop_controller(controller)
        subprocess.run(['sudo', 'mn', '-c'], capture_output=True)


# ---------------------------------------------------------------------------
# Campagna: N coppie, ordine alternato, poi analisi
# ---------------------------------------------------------------------------
def orchestrate(args):
    session_id = int(time())
    session_path = f"{current}/logs/session_{session_id}.json"
    os.makedirs(os.path.dirname(session_path), exist_ok=True)
    pass_args = ["--no-cli", "--idle", str(args.idle), "--warmup", str(args.warmup),
                 "--tail", str(args.tail), "--log-level", args.log_level,
                 "--ft-threshold", str(args.ft_threshold), "--valid-match", args.valid_match,
                 "--probe-pairs", str(args.probe_pairs), "--seed", str(args.seed)]
    if args.controller_cpus:
        pass_args += ["--controller-cpus", args.controller_cpus]
    if args.ovs_cpus:
        pass_args += ["--ovs-cpus", args.ovs_cpus]
    if args.isolate:
        pass_args.append("--isolate")

    runs = []
    for i in range(args.pairs):
        # ordine alternato: la deriva temporale della VM non favorisce sempre lo stesso controller
        order = ["ftfm", "baseline"] if i % 2 == 0 else ["baseline", "ftfm"]
        for pos, ctl in enumerate(order):
            rid = int(time())
            info(f"\n*** CAMPAGNA: coppia {i + 1}/{args.pairs}, run {ctl} (id {rid})\n")
            rc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                 "--controller", ctl, "--run-id", str(rid)] + pass_args).returncode
            runs.append({"controller": ctl, "dir": f"{current}/logs/{ctl}", "rid": rid,
                         "pair": i, "pos": pos, "rc": rc})
            write_json(session_path, runs)          # salvataggio incrementale
            sleep(args.cooldown)

    info(f"*** CAMPAGNA terminata. Sessione: {session_path}\n")
    try:
        import compare_runs
        compare_runs.main(["--session", session_path, "--out", f"{current}/logs/compare_{session_id}"])
    except ImportError:
        info("*** compare_runs.py non trovato: lancialo a mano con --session\n")


if __name__ == '__main__':
    setLogLevel('info')
    _args = parse_args()
    if _args.pairs > 0:
        orchestrate(_args)
    else:
        main(_args)