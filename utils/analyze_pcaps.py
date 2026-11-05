#!/usr/bin/env python3
"""
Analisi dei pcap di traffico benevolo, per TUTTI gli scenari in una sola esecuzione.

Struttura attesa:
    <root>/scenario_000/traffic_10_0_0_1.pcap ... traffic_10_0_0_N.pcap
    <root>/scenario_001/...
(se <root> contiene direttamente i traffic_10_0_0_*.pcap viene trattato come
 un singolo scenario)

Per ogni pcap estrae: volume, dimensione pacchetti, inter-arrival, burstiness,
protocolli, direzione, flussi 5-tuple, peer, flag TCP, DNS.

DATI PER STIMARE HARD E IDLE TIMEOUT (evidenziati nell'output)
  Il controller (simple_switch_14 modificato) installa regole con match
        in_port + eth_src + eth_dst + ipv4_src + ipv4_dst
  quindi UNA regola = UNA COPPIA DIREZIONALE DI HOST (src IP -> dst IP), che
  raccoglie tutte le connessioni tra i due host. Questa coppia e' chiamata PAIR.

  - GAP tra pacchetti consecutivi di una PAIR  -> IDLE timeout (la regola scade
    esattamente quando il silenzio della coppia supera l'idle timeout)
  - DURATA dei flussi TCP 5-tuple               -> HARD timeout: serve solo come
    scala temporale di applicazioni realistiche (la regola viene rinnovata dal
    controller ad ogni scadenza, quindi l'hard timeout e' un periodo di refresh
    scelto, non una proprieta' misurabile del traffico). La durata della PAIR
    coincide quasi con la durata della cattura e non e' informativa.
  - gap e durate anche per TCP / UDP / ALL / servizio; gap massimo per flusso;
    soglie (% oltre un dato timeout); tabella percentile -> secondi; classi
    di applicazione suggerite; variabilita' tra scenari.

Output (cartella -o): summary.csv (per host), scenario_summary.csv,
  timeout_summary.csv, timeout_thresholds.csv, details.json

Uso:
    pip install scapy
    python3 analyze_pcaps.py -d /path/.../pcaps/traffics -o output -j 4
    python3 analyze_pcaps.py -d /path/.../traffics/scenario_000      # un solo scenario
    python3 analyze_pcaps.py --verbose                                # anche report per host
    python3 analyze_pcaps.py --hard-bands 55-70,80-93,95-99 --idle-bands 92-95,95-98,98-99.5
"""
import argparse
import csv
import glob
import json
import os
import re
import socket
import statistics as st
import sys
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from scapy.all import PcapReader, Ether, IP, IPv6, TCP, UDP, ICMP, ARP, DNS, DNSQR, Raw

COLOR = False
BANNER = "1;30;43"   # grassetto, nero su giallo
HEAD = "1;33"        # grassetto giallo
GOOD = "1;32"        # grassetto verde
WARN = "1;31"        # grassetto rosso

DEFAULT_HARD = [5, 10, 15, 30, 60, 120, 300]
DEFAULT_IDLE = [1, 2, 3, 5, 10, 15, 30, 60]
DEFAULT_HARD_BANDS = "55-70,80-93,95-99"
DEFAULT_IDLE_BANDS = "92-95,95-98,98-99.5"
BASE_PCTS = [50, 75, 90, 95, 99]
CLASS_NAMES = ["breve", "media", "lunga"]


def c(txt, code):
    return f"\033[{code}m{txt}\033[0m" if COLOR else txt


# ----------------------------------------------------------------- helper --
def fmt_p(p):
    return f"{p:g}"


def pct_sorted(s, p):
    """Percentile p (0-100) su lista gia' ordinata, interpolazione lineare."""
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c_ = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c_] - s[f]) * (k - f)


def describe_sorted(s):
    if not s:
        return dict(min=0, max=0, mean=0, median=0, std=0, p90=0, p95=0, p99=0)
    return dict(
        min=s[0],
        max=s[-1],
        mean=st.fmean(s),
        median=st.median(s),
        std=st.pstdev(s) if len(s) > 1 else 0.0,
        p90=pct_sorted(s, 90),
        p95=pct_sorted(s, 95),
        p99=pct_sorted(s, 99),
    )


def describe(vals):
    return describe_sorted(sorted(vals))


def service_name(proto, port):
    try:
        return socket.getservbyport(port, proto)
    except OSError:
        return str(port)


def host_ip_from_filename(path):
    m = re.search(r"traffic_(\d+)_(\d+)_(\d+)_(\d+)\.pcap", os.path.basename(path))
    return ".".join(m.groups()) if m else None


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def parse_list(s):
    return [float(x) if "." in x else int(x) for x in s.split(",") if x.strip()]


def parse_bands(s):
    bands = []
    for tok in s.split(","):
        lo, hi = tok.split("-")
        bands.append((float(lo), float(hi)))
    return bands


# ------------------------------------------- raccolta dati per i timeout --
def new_group():
    # dur    : durata di ogni flusso/coppia con >= 2 pacchetti      (-> hard timeout)
    # gap    : gap tra pacchetti consecutivi dello stesso flusso/coppia (-> idle timeout)
    # maxgap : gap massimo di ogni flusso/coppia con >= 2 pacchetti
    return {"dur": array("d"), "gap": array("d"), "maxgap": array("d")}


def new_raw():
    return {
        "TCP": new_group(), "UDP": new_group(), "PAIR": new_group(),
        "svc": defaultdict(new_group),
        "single": 0,        # flussi con 1 solo pacchetto (esclusi dalle durate)
        "capture_s": 0.0,   # durata della cattura
    }


def merge_raw(raws):
    m = new_raw()
    for r in raws:
        for g in ("TCP", "UDP", "PAIR"):
            for k in ("dur", "gap", "maxgap"):
                m[g][k].extend(r[g][k])
        for label, grp in r["svc"].items():
            for k in ("dur", "gap", "maxgap"):
                m["svc"][label][k].extend(grp[k])
        m["single"] += r["single"]
        m["capture_s"] = max(m["capture_s"], r["capture_s"])
    return m


def combine(*groups):
    out = new_group()
    for g in groups:
        for k in ("dur", "gap", "maxgap"):
            out[k].extend(g[k])
    return out


def _above(s, t):
    """% di valori > t in una lista ordinata."""
    return 100.0 * (len(s) - bisect_right(s, t)) / len(s) if s else 0.0


def group_stats(g, hard_c, idle_c, pcts):
    d, gp, mg = sorted(g["dur"]), sorted(g["gap"]), sorted(g["maxgap"])
    return {
        "n_flows": len(d),
        "n_gaps": len(gp),
        "duration": describe_sorted(d),
        "gap": describe_sorted(gp),
        "max_gap_per_flow": describe_sorted(mg),
        "pctl": {
            "duration": {fmt_p(p): pct_sorted(d, p) for p in pcts},
            "gap": {fmt_p(p): pct_sorted(gp, p) for p in pcts},
        },
        # % di flussi piu' lunghi di T: la regola scadrebbe per hard timeout
        "hard_exceed_pct_flows": {str(t): _above(d, t) for t in hard_c},
        # % di gap piu' lunghi di T: ogni gap cosi' causa un nuovo Packet-In
        "idle_exceed_pct_gaps": {str(t): _above(gp, t) for t in idle_c},
        # % di flussi/coppie con almeno un gap piu' lungo di T
        "idle_exceed_pct_flows": {str(t): _above(mg, t) for t in idle_c},
    }


def make_classes(groups, hard_bands, idle_bands):
    td = groups["TCP"]["pctl"]["duration"]
    pg = groups["PAIR"]["pctl"]["gap"]
    out = []
    for i, (hb, ib) in enumerate(zip(hard_bands, idle_bands)):
        out.append({
            "name": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"classe{i + 1}",
            "hard_pct": list(hb),
            "hard_s": [td[fmt_p(hb[0])], td[fmt_p(hb[1])]],
            "idle_pct": list(ib),
            "idle_s": [pg[fmt_p(ib[0])], pg[fmt_p(ib[1])]],
        })
    return out


def timeout_stats(raw, cfg, top_services=10):
    hard_c, idle_c, pcts = cfg["hard_c"], cfg["idle_c"], cfg["pcts"]
    groups_raw = {
        "TCP": raw["TCP"],
        "UDP": raw["UDP"],
        "ALL": combine(raw["TCP"], raw["UDP"]),
        "PAIR": raw["PAIR"],
    }
    groups = {name: group_stats(g, hard_c, idle_c, pcts) for name, g in groups_raw.items()}
    svc_sorted = sorted(raw["svc"].items(), key=lambda kv: len(kv[1]["dur"]), reverse=True)
    return {
        "flows_multi_packet": groups["ALL"]["n_flows"],
        "flows_single_packet": raw["single"],
        "capture_s": raw["capture_s"],
        "hard_candidates": hard_c,
        "idle_candidates": idle_c,
        "pcts": pcts,
        "groups": groups,
        "classes": make_classes(groups, cfg["hard_bands"], cfg["idle_bands"]),
        "by_service": {
            label: group_stats(g, hard_c, idle_c, [])
            for label, g in svc_sorted[:top_services] if g["dur"]
        },
    }


# ---------------------------------------------------------- analisi pcap --
def analyze(path, cfg):
    host = host_ip_from_filename(path)

    sizes, iats = [], []
    ts_first = ts_last = None
    prev_ts = None
    total_bytes = 0

    l3 = Counter()          # IPv4, IPv6, ARP, other
    l4 = Counter()          # TCP, UDP, ICMP, other
    services = Counter()    # servizi per porta (pacchetti)
    service_bytes = Counter()
    tcp_flags = Counter()
    syn_count = 0
    dns_queries = Counter()
    dst_ips = Counter()     # solo traffico uscente
    src_ips = Counter()     # solo traffico entrante
    direction = Counter()   # pkts out / in / other
    dir_bytes = Counter()
    dst_ports = Counter()
    per_sec_pkts = defaultdict(int)
    per_sec_bytes = defaultdict(int)
    payload_sizes = []

    raw = new_raw()

    # flussi bidirezionali 5-tuple:
    #   key -> [first_ts, last_ts, pkts, bytes, max_gap, proto, label]
    flows = {}
    # PAIR = coppia DIREZIONALE (src IP -> dst IP) = una regola OpenFlow:
    #   (src, dst) -> [first_ts, last_ts, pkts, max_gap]
    pairs = {}

    with PcapReader(path) as reader:
        for pkt in reader:
            ts = float(pkt.time)
            size = int(getattr(pkt, "wirelen", None) or len(pkt))

            if ts_first is None:
                ts_first = ts
            ts_last = ts
            if prev_ts is not None:
                iats.append(max(ts - prev_ts, 0.0))
            prev_ts = ts

            sizes.append(size)
            total_bytes += size
            sec = int(ts - ts_first)
            per_sec_pkts[sec] += 1
            per_sec_bytes[sec] += size

            # ---- L3
            ip = None
            if pkt.haslayer(IP):
                ip = pkt[IP]
                l3["IPv4"] += 1
            elif pkt.haslayer(IPv6):
                ip = pkt[IPv6]
                l3["IPv6"] += 1
            elif pkt.haslayer(ARP):
                l3["ARP"] += 1
                continue
            else:
                l3["Other"] += 1
                continue

            src, dst = ip.src, ip.dst

            # ---- PAIR direzionale (src -> dst): corrisponde a una regola OpenFlow
            pk = (src, dst)
            pr = pairs.get(pk)
            if pr is None:
                pairs[pk] = [ts, ts, 1, 0.0]
            else:
                g = max(ts - pr[1], 0.0)
                raw["PAIR"]["gap"].append(g)
                pr[1] = max(ts, pr[1])
                pr[2] += 1
                pr[3] = max(pr[3], g)

            # ---- direzione
            if host and src == host:
                direction["out"] += 1
                dir_bytes["out"] += size
                dst_ips[dst] += 1
            elif host and dst == host:
                direction["in"] += 1
                dir_bytes["in"] += size
                src_ips[src] += 1
            else:
                direction["other"] += 1
                dir_bytes["other"] += size

            # ---- L4
            sport = dport = None
            proto_name = None
            if pkt.haslayer(TCP):
                t = pkt[TCP]
                l4["TCP"] += 1
                proto_name = "tcp"
                sport, dport = t.sport, t.dport
                flags = str(t.flags)
                for f in flags:
                    tcp_flags[f] += 1
                if "S" in flags and "A" not in flags:
                    syn_count += 1
                payload_sizes.append(len(t.payload))
            elif pkt.haslayer(UDP):
                u = pkt[UDP]
                l4["UDP"] += 1
                proto_name = "udp"
                sport, dport = u.sport, u.dport
                payload_sizes.append(len(u.payload))
                if pkt.haslayer(DNS) and pkt.haslayer(DNSQR) and pkt[DNS].qr == 0:
                    try:
                        qn = pkt[DNSQR].qname.decode(errors="ignore").rstrip(".")
                        dns_queries[qn] += 1
                    except Exception:
                        pass
            elif pkt.haslayer(ICMP):
                l4["ICMP"] += 1
                services["icmp"] += 1
                service_bytes["icmp"] += size
            else:
                l4["Other"] += 1

            if proto_name:
                # porta di servizio: quella well-known, altrimenti la destinazione
                if sport is not None and sport < 1024 and (dport >= 1024 or sport < dport):
                    svc_port = sport
                else:
                    svc_port = dport
                label = f"{service_name(proto_name, svc_port)}/{proto_name}"
                services[label] += 1
                service_bytes[label] += size
                if host and src == host:
                    dst_ports[f"{dport}/{proto_name}"] += 1

                key = (proto_name,) + tuple(sorted([(src, sport), (dst, dport)]))
                fl = flows.get(key)
                if fl is None:
                    flows[key] = [ts, ts, 1, size, 0.0, proto_name, label]
                else:
                    gap = max(ts - fl[1], 0.0)
                    fl[1] = max(ts, fl[1])
                    fl[2] += 1
                    fl[3] += size
                    fl[4] = max(fl[4], gap)
                    # gap tra due pacchetti consecutivi dello stesso flusso
                    raw[proto_name.upper()]["gap"].append(gap)
                    raw["svc"][label]["gap"].append(gap)

    n = len(sizes)
    duration = (ts_last - ts_first) if n > 1 else 0.0
    raw["capture_s"] = duration

    # ---- durate (solo >= 2 pacchetti: i flussi da 1 pkt hanno durata 0)
    for f in flows.values():
        if f[2] >= 2:
            d = f[1] - f[0]
            gp = raw[f[5].upper()]
            gp["dur"].append(d)
            gp["maxgap"].append(f[4])
            raw["svc"][f[6]]["dur"].append(d)
            raw["svc"][f[6]]["maxgap"].append(f[4])
        else:
            raw["single"] += 1
    for p in pairs.values():
        if p[2] >= 2:
            raw["PAIR"]["dur"].append(p[1] - p[0])
            raw["PAIR"]["maxgap"].append(p[3])

    flow_durs = [f[1] - f[0] for f in flows.values()]
    flow_pkts = [f[2] for f in flows.values()]
    flow_bytes = [f[3] for f in flows.values()]
    top_flows = sorted(flows.items(), key=lambda kv: kv[1][3], reverse=True)[:5]

    max_pps = max(per_sec_pkts.values()) if per_sec_pkts else 0
    max_bps = max(per_sec_bytes.values()) * 8 if per_sec_bytes else 0

    return {
        "file": os.path.basename(path),
        "host_ip": host,
        "packets": n,
        "bytes": total_bytes,
        "duration_s": duration,
        "avg_pps": n / duration if duration else 0,
        "avg_throughput_bps": total_bytes * 8 / duration if duration else 0,
        "peak_pps_1s": max_pps,
        "peak_throughput_bps_1s": max_bps,
        "packet_size": describe(sizes),
        "payload_size_l4": describe(payload_sizes),
        "iat_s": describe(iats),
        "l3": dict(l3),
        "l4": dict(l4),
        "direction_pkts": dict(direction),
        "direction_bytes": dict(dir_bytes),
        "top_services_pkts": services.most_common(10),
        "top_services_bytes": service_bytes.most_common(10),
        "top_dst_ips": dst_ips.most_common(10),
        "top_src_ips": src_ips.most_common(10),
        "unique_dst_ips": len(dst_ips),
        "unique_src_ips": len(src_ips),
        "top_dst_ports": dst_ports.most_common(10),
        "tcp_flags": dict(tcp_flags),
        "tcp_syn_no_ack": syn_count,
        "flows": {
            "count": len(flows),
            "duration_s": describe(flow_durs),
            "packets": describe(flow_pkts),
            "bytes": describe(flow_bytes),
            "top_by_bytes": [
                {"flow": " <-> ".join(f"{a}:{p}" for a, p in k[1:]) + f" ({k[0]})",
                 "packets": v[2], "bytes": v[3], "duration_s": v[1] - v[0]}
                for k, v in top_flows
            ],
        },
        "dns": {
            "queries_total": sum(dns_queries.values()),
            "unique_names": len(dns_queries),
            "top_names": dns_queries.most_common(10),
        },
        "timeout": timeout_stats(raw, cfg),
        "_raw": raw,   # dati grezzi per gli aggregati: non finiscono nel JSON
    }


def _task(args):
    """Worker (top-level per il multiprocessing)."""
    scenario, path, cfg = args
    try:
        r = analyze(path, cfg)
        r["scenario"] = scenario
        return r
    except Exception as e:   # pcap vuoto/corrotto: non blocca tutto il resto
        return {"error": f"{type(e).__name__}: {e}", "scenario": scenario, "file": os.path.basename(path)}


# ---------------------------------------------------------------- output --
def _tg(ts, grp):
    return ts["groups"][grp]


def stat_cols(prefix, d):
    return {
        f"{prefix}_p50": round(d["median"], 3), f"{prefix}_p90": round(d["p90"], 3),
        f"{prefix}_p95": round(d["p95"], 3), f"{prefix}_p99": round(d["p99"], 3),
        f"{prefix}_max": round(d["max"], 3),
    }


def timeout_cols(ts):
    row = {}
    row.update(stat_cols("tcp_dur", _tg(ts, "TCP")["duration"]))
    row.update(stat_cols("pair_gap", _tg(ts, "PAIR")["gap"]))
    return row


TIMEOUT_COLS = [f"{p}_{s}" for p in ("tcp_dur", "pair_gap") for s in ("p50", "p90", "p95", "p99", "max")]

SUMMARY_COLS = [
    "scenario", "file", "host_ip", "packets", "bytes", "duration_s", "avg_pps",
    "avg_throughput_bps", "pkt_mean", "pkt_median", "pkt_min", "pkt_max", "pkt_std",
    "iat_mean_ms", "iat_median_ms", "iat_p95_ms", "peak_pps_1s", "pct_tcp", "pct_udp",
    "pct_out", "flows", "unique_dst_ips", "top_service",
] + TIMEOUT_COLS


def flat_row(r):
    n = max(r["packets"], 1)
    out = r["direction_pkts"].get("out", 0)
    top = r["top_services_pkts"][0][0] if r["top_services_pkts"] else "-"
    row = {
        "scenario": r["scenario"], "file": r["file"], "host_ip": r["host_ip"],
        "packets": r["packets"], "bytes": r["bytes"],
        "duration_s": round(r["duration_s"], 2), "avg_pps": round(r["avg_pps"], 2),
        "avg_throughput_bps": round(r["avg_throughput_bps"], 1),
        "pkt_mean": round(r["packet_size"]["mean"], 1),
        "pkt_median": round(r["packet_size"]["median"], 1),
        "pkt_min": r["packet_size"]["min"], "pkt_max": r["packet_size"]["max"],
        "pkt_std": round(r["packet_size"]["std"], 1),
        "iat_mean_ms": round(r["iat_s"]["mean"] * 1000, 3),
        "iat_median_ms": round(r["iat_s"]["median"] * 1000, 3),
        "iat_p95_ms": round(r["iat_s"]["p95"] * 1000, 3),
        "peak_pps_1s": r["peak_pps_1s"],
        "pct_tcp": round(100 * r["l4"].get("TCP", 0) / n, 1),
        "pct_udp": round(100 * r["l4"].get("UDP", 0) / n, 1),
        "pct_out": round(100 * out / n, 1),
        "flows": r["flows"]["count"], "unique_dst_ips": r["unique_dst_ips"],
        "top_service": top,
    }
    row.update(timeout_cols(r["timeout"]))
    return row


def _row(label, n, d):
    return (f"  {label:<8}{n:>9}  {d['median']:>9.3f} {d['p90']:>9.3f} "
            f"{d['p95']:>9.3f} {d['p99']:>9.3f} {d['max']:>9.3f}")


def _header(title):
    return (c(f"  {title}", HEAD) + "\n"
            f"  {'gruppo':<8}{'n':>9}  {'mediana':>9} {'p90':>9} {'p95':>9} {'p99':>9} {'max':>9}")


def print_timeout(ts, title, full=True):
    """Blocco evidenziato con i dati per la stima di hard e idle timeout."""
    g = ts["groups"]
    print(c((" >>> " + title + " <<< ").center(76), BANNER))
    print(f"  Flussi con >=2 pkt: {ts['flows_multi_packet']}   "
          f"flussi da 1 solo pkt (esclusi): {ts['flows_single_packet']}   "
          f"durata cattura (max): {ts['capture_s']:.1f} s")

    print(_header("DURATA DEL FLUSSO [s]  ->  scala per HARD TIMEOUT"))
    for name in ("TCP", "UDP", "ALL", "PAIR"):
        print(_row(name, g[name]["n_flows"], g[name]["duration"]))

    print(_header("GAP TRA PACCHETTI CONSECUTIVI [s]  ->  IDLE TIMEOUT"))
    for name in ("PAIR", "TCP", "UDP", "ALL"):
        print(_row(name, g[name]["n_gaps"], g[name]["gap"]))

    print(_header("GAP MASSIMO DI OGNI FLUSSO / COPPIA [s]  (caso peggiore)"))
    for name in ("PAIR", "TCP", "UDP", "ALL"):
        print(_row(name, g[name]["n_flows"], g[name]["max_gap_per_flow"]))
    print("  PAIR = coppia direzionale src IP -> dst IP = UNA regola OpenFlow "
          "(match in_port+MAC+IP)")

    if full:
        print()
        print(c("  SOGLIE: quanti flussi/gap supererebbero un dato timeout?", HEAD))
        print(f"  {'hard T(s)':>10} | {'% flussi TCP durata>T':>22} {'(ALL)':>9} {'(PAIR)':>9}")
        for t in ts["hard_candidates"]:
            k = str(t)
            print(f"  {t:>10} | {g['TCP']['hard_exceed_pct_flows'][k]:>22.2f} "
                  f"{g['ALL']['hard_exceed_pct_flows'][k]:>9.2f} "
                  f"{g['PAIR']['hard_exceed_pct_flows'][k]:>9.2f}")
        print(f"  {'idle T(s)':>10} | {'% gap>T (PAIR)':>15} {'% coppie con gap>T':>19} | "
              f"{'% gap>T (TCP)':>14} {'% flussi con gap>T':>19}")
        for t in ts["idle_candidates"]:
            k = str(t)
            print(f"  {t:>10} | {g['PAIR']['idle_exceed_pct_gaps'][k]:>15.2f} "
                  f"{g['PAIR']['idle_exceed_pct_flows'][k]:>19.2f} | "
                  f"{g['TCP']['idle_exceed_pct_gaps'][k]:>14.2f} "
                  f"{g['TCP']['idle_exceed_pct_flows'][k]:>19.2f}")

        if ts["by_service"]:
            print()
            print(c("  PER SERVIZIO (top per numero di flussi)  durata / gap [s]", HEAD))
            print(f"  {'servizio':<14}{'flussi':>8} | {'durata p50':>10} {'p90':>8} {'p99':>8} "
                  f"| {'gap p50':>8} {'p90':>8} {'p99':>8}")
            for label, s in ts["by_service"].items():
                d, gp = s["duration"], s["gap"]
                print(f"  {label:<14}{s['n_flows']:>8} | {d['median']:>10.3f} {d['p90']:>8.3f} "
                      f"{d['p99']:>8.3f} | {gp['median']:>8.3f} {gp['p90']:>8.3f} {gp['p99']:>8.3f}")

        # tabella percentile -> secondi
        td = g["TCP"]["pctl"]["duration"]
        pg = g["PAIR"]["pctl"]["gap"]
        print()
        print(c("  PERCENTILE -> VALORE [s]", HEAD))
        print(f"  {'percentile':>10} | {'durata flusso TCP (hard)':>26} | {'gap PAIR (idle)':>16}")
        for p in ts["pcts"]:
            k = fmt_p(p)
            print(f"  {'p' + k:>10} | {td[k]:>26.3f} | {pg[k]:>16.3f}")

        # classi suggerite
        print()
        print(c("  CLASSI DI APPLICAZIONE SUGGERITE (euristica, da calibrare)", HEAD))
        print(f"  {'classe':<8} | {'hard timeout [s]':>20} {'percentili TCP':>15} | "
              f"{'idle timeout [s]':>20} {'percentili PAIR':>16}")
        for cl in ts["classes"]:
            hb = f"p{fmt_p(cl['hard_pct'][0])}-p{fmt_p(cl['hard_pct'][1])}"
            ib = f"p{fmt_p(cl['idle_pct'][0])}-p{fmt_p(cl['idle_pct'][1])}"
            hs = f"{cl['hard_s'][0]:.1f} - {cl['hard_s'][1]:.1f}"
            is_ = f"{cl['idle_s'][0]:.2f} - {cl['idle_s'][1]:.2f}"
            print(c(f"  {cl['name']:<8} | {hs:>20} {hb:>15} | {is_:>20} {ib:>16}", GOOD))
            if cl["idle_s"][1] >= cl["hard_s"][0]:
                print(c(f"    ATTENZIONE: nella classe '{cl['name']}' l'idle massimo "
                        f"({cl['idle_s'][1]:.1f} s) e' >= dell'hard minimo "
                        f"({cl['hard_s'][0]:.1f} s): serve idle < hard.", WARN))
        print("  Hard: le durate TCP danno solo la scala temporale di applicazioni realistiche;"
              " Idle: i gap PAIR sono la misura esatta per la regola.")

        tcp_max = g["TCP"]["duration"]["max"]
        cap = ts["capture_s"]
        if cap and tcp_max >= 0.95 * cap:
            print(c(f"  ATTENZIONE: la durata massima TCP ({tcp_max:.0f} s) e' ~ la durata della "
                    f"cattura ({cap:.0f} s): i flussi lunghi sono TRONCATI.", WARN))
        print("  Nota: gap/durate sono misurati nel pcap, non sulla regola OpenFlow reale.")
    print()


def print_report(r):
    ps, iat = r["packet_size"], r["iat_s"]
    print("=" * 76)
    print(f"{r['scenario']} / {r['file']}  (host {r['host_ip']})")
    print("=" * 76)
    print(f"Pacchetti: {r['packets']}   Byte: {fmt_bytes(r['bytes'])}   "
          f"Durata: {r['duration_s']:.1f} s")
    print(f"Rate medio: {r['avg_pps']:.2f} pps, {r['avg_throughput_bps']/1000:.2f} kbit/s   "
          f"(picco 1s: {r['peak_pps_1s']} pps, {r['peak_throughput_bps_1s']/1000:.1f} kbit/s)")
    print(f"Dimensione pacchetto [B]: min {ps['min']}  max {ps['max']}  "
          f"media {ps['mean']:.1f}  mediana {ps['median']:.0f}  std {ps['std']:.1f}  "
          f"p95 {ps['p95']:.0f}")
    print(f"Inter-arrival [ms]: media {iat['mean']*1000:.2f}  mediana {iat['median']*1000:.2f}  "
          f"p95 {iat['p95']*1000:.2f}  max {iat['max']*1000:.0f}")
    print(f"L3: {r['l3']}   L4: {r['l4']}")
    print(f"Direzione (pkt): {r['direction_pkts']}   (byte): {r['direction_bytes']}")
    print(f"Flussi: {r['flows']['count']}  | durata media {r['flows']['duration_s']['mean']:.1f} s  "
          f"| pkt/flusso medio {r['flows']['packets']['mean']:.1f}  "
          f"| byte/flusso medio {r['flows']['bytes']['mean']:.0f}")
    print(f"Peer: {r['unique_dst_ips']} dst IP unici (uscente), "
          f"{r['unique_src_ips']} src IP unici (entrante)")
    print("Top servizi (pkt):", ", ".join(f"{k}={v}" for k, v in r["top_services_pkts"][:6]))
    print("Top dst IP:", ", ".join(f"{k}={v}" for k, v in r["top_dst_ips"][:5]))
    if r["dns"]["queries_total"]:
        print(f"DNS: {r['dns']['queries_total']} query, {r['dns']['unique_names']} nomi unici")
    print()
    print_timeout(r["timeout"], f"TIMEOUT - {r['scenario']}/{r['file']}", full=False)


def aggregate(results, cfg):
    """Aggregato di una lista di host (uno scenario, o tutti). Ritorna (dict, raw)."""
    tot_pkts = sum(r["packets"] for r in results)
    tot_bytes = sum(r["bytes"] for r in results)
    services = Counter()
    l4 = Counter()
    for r in results:
        for k, v in r["top_services_pkts"]:
            services[k] += v
        l4.update(r["l4"])
    merged = merge_raw([r["_raw"] for r in results])
    return {
        "hosts": len(results),
        "total_packets": tot_pkts,
        "total_bytes": tot_bytes,
        "mean_packet_size": tot_bytes / tot_pkts if tot_pkts else 0,
        "packets_per_host": describe([r["packets"] for r in results]),
        "duration_s": describe([r["duration_s"] for r in results]),
        "avg_pps_per_host": describe([r["avg_pps"] for r in results]),
        "avg_bps_per_host": describe([r["avg_throughput_bps"] for r in results]),
        "total_flows": sum(r["flows"]["count"] for r in results),
        "l4_total": dict(l4),
        "top_services_pkts": services.most_common(10),
        "timeout": timeout_stats(merged, cfg),
    }, merged


def print_scenario_table(per_scen):
    print(c(" >>> TIMEOUT PER SCENARIO <<< ".center(110), BANNER))
    print(f"  {'scenario':<14}{'host':>5}{'pkt':>9}{'flussi':>8} | "
          f"{'TCP durata [s]: p50':>20} {'p90':>8} {'p99':>8} {'max':>8} | "
          f"{'PAIR gap [s]: p90':>18} {'p95':>7} {'p99':>8} {'max':>8}")
    for name, a in per_scen.items():
        ts = a["timeout"]
        d, gp = _tg(ts, "TCP")["duration"], _tg(ts, "PAIR")["gap"]
        trunc = "*" if ts["capture_s"] and d["max"] >= 0.95 * ts["capture_s"] else " "
        print(f"  {name:<14}{a['hosts']:>5}{a['total_packets']:>9}{a['total_flows']:>8} | "
              f"{d['median']:>20.3f} {d['p90']:>8.2f} {d['p99']:>8.2f} {d['max']:>7.1f}{trunc} | "
              f"{gp['p90']:>18.3f} {gp['p95']:>7.2f} {gp['p99']:>8.2f} {gp['max']:>8.1f}")
    print("  (* = durata massima TCP ~ durata cattura: flussi lunghi troncati)")
    print()


def print_variability(per_scen):
    metrics = [
        ("TCP durata p50 [s]", "TCP", "duration", "median"),
        ("TCP durata p90 [s]", "TCP", "duration", "p90"),
        ("TCP durata p99 [s]", "TCP", "duration", "p99"),
        ("PAIR gap p95 [s]", "PAIR", "gap", "p95"),
        ("PAIR gap p99 [s]", "PAIR", "gap", "p99"),
    ]
    print(c("  VARIABILITA' TRA SCENARI (quanto cambiano i percentili da uno scenario all'altro)", HEAD))
    print(f"  {'metrica':<20} | {'min':>9} {'(scenario)':<14} | {'mediana':>9} | {'max':>9} {'(scenario)':<14}")
    for label, grp, kind, key in metrics:
        nkey = "n_flows" if kind == "duration" else "n_gaps"
        vals = [(a["timeout"]["groups"][grp][kind][key], name) for name, a in per_scen.items()
                if a["timeout"]["groups"][grp][nkey] > 0]
        if not vals:
            continue
        vals.sort()
        lo, hi = vals[0], vals[-1]
        med = st.median(v for v, _ in vals)
        print(f"  {label:<20} | {lo[0]:>9.3f} {lo[1]:<14} | {med:>9.3f} | {hi[0]:>9.3f} {hi[1]:<14}")
    print()


def write_csvs(out_dir, results, per_scen, total):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLS)
        for r in results:
            row = flat_row(r)
            w.writerow([row[k] for k in SUMMARY_COLS])

    with open(os.path.join(out_dir, "scenario_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "hosts", "packets", "bytes", "flows"] + TIMEOUT_COLS)
        for name, a in list(per_scen.items()) + [("ALL_SCENARIOS", total)]:
            tc = timeout_cols(a["timeout"])
            w.writerow([name, a["hosts"], a["total_packets"], a["total_bytes"], a["total_flows"]]
                       + [tc[k] for k in TIMEOUT_COLS])

    scopes = [(n, a["timeout"]) for n, a in per_scen.items()] + [("ALL_SCENARIOS", total["timeout"])]
    with open(os.path.join(out_dir, "timeout_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "group", "kind", "n", "min", "p50", "p90", "p95", "p99", "max", "mean"])
        for scope, ts in scopes:
            items = list(ts["groups"].items()) + [(f"svc:{k}", v) for k, v in ts["by_service"].items()]
            for gname, s in items:
                for kind, key, n in (("flow_duration_s", "duration", s["n_flows"]),
                                     ("intra_flow_gap_s", "gap", s["n_gaps"]),
                                     ("max_gap_per_flow_s", "max_gap_per_flow", s["n_flows"])):
                    d = s[key]
                    w.writerow([scope, gname, kind, n] +
                               [round(d[k], 4) for k in ("min", "median", "p90", "p95", "p99", "max", "mean")])

    with open(os.path.join(out_dir, "timeout_thresholds.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "group", "kind", "T_s", "pct_exceeding"])
        for scope, ts in scopes:
            for gname, s in ts["groups"].items():
                for kind, key in (("hard: flows with duration > T", "hard_exceed_pct_flows"),
                                  ("idle: gaps > T", "idle_exceed_pct_gaps"),
                                  ("idle: flows/pairs with a gap > T", "idle_exceed_pct_flows")):
                    for t, v in s[key].items():
                        w.writerow([scope, gname, kind, t, round(v, 3)])


# ------------------------------------------------------------------ main --
def discover(root):
    pat = "traffic_10_0_0_*.pcap"
    if glob.glob(os.path.join(root, pat)):
        return [(os.path.basename(os.path.normpath(root)), root)]
    dirs = sorted(d for d in glob.glob(os.path.join(root, "scenario_*")) if os.path.isdir(d))
    return [(os.path.basename(d), d) for d in dirs]


def main():
    global COLOR
    default_dir = os.path.join(os.environ.get("PROGETTO", "."), "traffics")
    ap = argparse.ArgumentParser(description="Statistiche sui pcap di traffico benevolo (tutti gli scenari)")
    ap.add_argument("-d", "--dir", default=default_dir,
                    help=f"cartella con le sotto-cartelle scenario_* (o un singolo scenario) (default: {default_dir})")
    ap.add_argument("-o", "--out", default="pcap_stats", help="cartella di output")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="processi paralleli (un pcap per processo; occhio alla RAM)")
    ap.add_argument("--hard-candidates", default=",".join(map(str, DEFAULT_HARD)),
                    help="soglie di hard timeout da valutare [s], separate da virgola")
    ap.add_argument("--idle-candidates", default=",".join(map(str, DEFAULT_IDLE)),
                    help="soglie di idle timeout da valutare [s], separate da virgola")
    ap.add_argument("--hard-bands", default=DEFAULT_HARD_BANDS,
                    help="bande di percentile (durata TCP) per le classi, es. 55-70,80-93,95-99")
    ap.add_argument("--idle-bands", default=DEFAULT_IDLE_BANDS,
                    help="bande di percentile (gap PAIR) per le classi, es. 92-95,95-98,98-99.5")
    ap.add_argument("--verbose", action="store_true", help="stampa anche il report di ogni host")
    ap.add_argument("--no-color", action="store_true", help="disattiva l'evidenziazione ANSI")
    args = ap.parse_args()

    COLOR = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ

    hard_bands, idle_bands = parse_bands(args.hard_bands), parse_bands(args.idle_bands)
    if len(hard_bands) != len(idle_bands):
        sys.exit("--hard-bands e --idle-bands devono avere lo stesso numero di bande")
    edges = {p for b in hard_bands + idle_bands for p in b}
    cfg = {
        "hard_c": parse_list(args.hard_candidates),
        "idle_c": parse_list(args.idle_candidates),
        "pcts": sorted(set(BASE_PCTS) | edges),
        "hard_bands": hard_bands,
        "idle_bands": idle_bands,
    }

    scenarios = discover(args.dir)
    if not scenarios:
        sys.exit(f"Nessuno scenario trovato in {args.dir}")

    tasks = []
    for sname, sdir in scenarios:
        files = sorted(
            glob.glob(os.path.join(sdir, "traffic_10_0_0_*.pcap")),
            key=lambda p: int(re.search(r"_(\d+)\.pcap$", p).group(1)),
        )
        tasks += [(sname, f, cfg) for f in files]
    if not tasks:
        sys.exit(f"Nessun pcap trovato in {args.dir}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[+] {len(scenarios)} scenari, {len(tasks)} pcap, {args.jobs} processi", file=sys.stderr)

    by_scen = defaultdict(list)
    done = 0

    def handle(r):
        nonlocal done
        done += 1
        if "error" in r:
            print(f"[!] {r['scenario']}/{r['file']}: {r['error']} (saltato)", file=sys.stderr)
            return
        print(f"[{done}/{len(tasks)}] {r['scenario']}/{r['file']}", file=sys.stderr)
        by_scen[r["scenario"]].append(r)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(_task, tasks):
                handle(r)
    else:
        for t in tasks:
            handle(_task(t))

    if args.verbose:
        for sname, _ in scenarios:
            for r in by_scen.get(sname, []):
                print_report(r)

    per_scen, scen_raws, all_results = {}, [], []
    for sname, _ in scenarios:
        res = by_scen.get(sname, [])
        if not res:
            continue
        a, raw = aggregate(res, cfg)
        per_scen[sname] = a
        scen_raws.append(raw)
        for r in res:
            r.pop("_raw", None)   # libera memoria: l'aggregato ha gia' i dati
        all_results += res
        if args.verbose:
            print_timeout(a["timeout"], f"TIMEOUT - {sname} (tutti gli host)", full=True)

    # aggregato su tutti gli scenari
    tot_pkts = sum(a["total_packets"] for a in per_scen.values())
    tot_bytes = sum(a["total_bytes"] for a in per_scen.values())
    services, l4 = Counter(), Counter()
    for a in per_scen.values():
        for k, v in a["top_services_pkts"]:
            services[k] += v
        l4.update(a["l4_total"])
    merged = merge_raw(scen_raws)
    total = {
        "hosts": sum(a["hosts"] for a in per_scen.values()),
        "total_packets": tot_pkts,
        "total_bytes": tot_bytes,
        "mean_packet_size": tot_bytes / tot_pkts if tot_pkts else 0,
        "total_flows": sum(a["total_flows"] for a in per_scen.values()),
        "l4_total": dict(l4),
        "top_services_pkts": services.most_common(10),
        "timeout": timeout_stats(merged, cfg),
    }

    print("#" * 76)
    print(f"AGGREGATO SU {len(per_scen)} SCENARI")
    print("#" * 76)
    print(f"Host: {total['hosts']}  Pacchetti totali: {tot_pkts}  Byte totali: {fmt_bytes(tot_bytes)}  "
          f"Flussi: {total['total_flows']}")
    print(f"Dimensione media pacchetto (pesata): {total['mean_packet_size']:.1f} B   L4: {total['l4_total']}")
    print("Top servizi:", ", ".join(f"{k}={v}" for k, v in total["top_services_pkts"][:8]))
    print()
    print_scenario_table(per_scen)
    print_variability(per_scen)
    print_timeout(total["timeout"], "TIMEOUT - AGGREGATO SU TUTTI GLI SCENARI", full=True)

    write_csvs(args.out, all_results, per_scen, total)
    with open(os.path.join(args.out, "details.json"), "w") as fh:
        json.dump({
            "aggregate": total,
            "per_scenario": per_scen,
            "per_host": [{k: v for k, v in r.items() if k != "_raw"} for r in all_results],
        }, fh, indent=2, default=str)

    print(f"Salvati in {args.out}/: summary.csv, scenario_summary.csv, timeout_summary.csv, "
          f"timeout_thresholds.csv, details.json")


if __name__ == "__main__":
    main()
