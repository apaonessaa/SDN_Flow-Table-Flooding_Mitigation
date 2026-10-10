#!/usr/bin/env python3
"""
Analisi dei pcap di traffico benevolo (scenario_000).

Per ogni file traffic_10_0_0_X.pcap estrae:
  - volume: pacchetti, byte, durata, pps, throughput medio
  - dimensione pacchetti: min / max / media / mediana / dev.std / percentili
  - inter-arrival time (IAT): media / mediana / p95 / max
  - burstiness: picco di pps e throughput su finestre di 1 s
  - distribuzione protocolli (L3, L4, applicativi per porta)
  - direzione (uscente/entrante rispetto all'IP dell'host simulato)
  - flussi (5-tuple bidirezionali): numero, durata, pacchetti/flusso
  - peer: IP destinazione unici, top destinazioni, top porte
  - TCP: flag, connessioni iniziate (SYN senza ACK)
  - DNS: query uniche e nomi piu' richiesti

Output: tabella a console + summary.csv + details.json (+ aggregato).

Uso:
    pip install scapy
    python3 analyze_pcaps.py                       # usa $PROGETTO/traffic/scenario_000
    python3 analyze_pcaps.py -d /path/ai/pcap -o risultati
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
from collections import Counter, defaultdict

from scapy.all import PcapReader, Ether, IP, IPv6, TCP, UDP, ICMP, ARP, DNS, DNSQR, Raw


# ----------------------------------------------------------------- helper --
def pct(sorted_vals, p):
    """Percentile p (0-100) su lista gia' ordinata, interpolazione lineare."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def describe(vals):
    """Statistiche descrittive di una lista numerica."""
    if not vals:
        return dict(min=0, max=0, mean=0, median=0, std=0, p90=0, p95=0, p99=0)
    s = sorted(vals)
    return dict(
        min=s[0],
        max=s[-1],
        mean=st.fmean(s),
        median=st.median(s),
        std=st.pstdev(s) if len(s) > 1 else 0.0,
        p90=pct(s, 90),
        p95=pct(s, 95),
        p99=pct(s, 99),
    )


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


# ---------------------------------------------------------- analisi pcap --
def analyze(path):
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

    # flussi bidirezionali: key -> [first_ts, last_ts, pkts, bytes]
    flows = {}

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
                    flows[key] = [ts, ts, 1, size]
                else:
                    fl[1] = ts
                    fl[2] += 1
                    fl[3] += size

    n = len(sizes)
    duration = (ts_last - ts_first) if n > 1 else 0.0

    flow_durs = [f[1] - f[0] for f in flows.values()]
    flow_pkts = [f[2] for f in flows.values()]
    flow_bytes = [f[3] for f in flows.values()]
    top_flows = sorted(flows.items(), key=lambda kv: kv[1][3], reverse=True)[:5]

    max_pps = max(per_sec_pkts.values()) if per_sec_pkts else 0
    max_bps = max(per_sec_bytes.values()) * 8 if per_sec_bytes else 0

    res = {
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
    }
    return res


# ---------------------------------------------------------------- output --
SUMMARY_COLS = [
    ("file", "File"), ("host_ip", "Host"), ("packets", "Pkts"), ("bytes", "Bytes"),
    ("duration_s", "Durata(s)"), ("avg_pps", "pps"), ("avg_throughput_bps", "bps"),
    ("pkt_mean", "Size media"), ("pkt_median", "Size mediana"),
    ("pkt_min", "Size min"), ("pkt_max", "Size max"), ("pkt_std", "Size std"),
    ("iat_mean_ms", "IAT medio(ms)"), ("iat_median_ms", "IAT mediana(ms)"),
    ("iat_p95_ms", "IAT p95(ms)"), ("peak_pps_1s", "Picco pps"),
    ("pct_tcp", "%TCP"), ("pct_udp", "%UDP"), ("pct_out", "%uscente"),
    ("flows", "Flussi"), ("unique_dst_ips", "Dst IP unici"),
    ("top_service", "Servizio principale"),
]


def flat_row(r):
    n = max(r["packets"], 1)
    out = r["direction_pkts"].get("out", 0)
    top = r["top_services_pkts"][0][0] if r["top_services_pkts"] else "-"
    return {
        "file": r["file"], "host_ip": r["host_ip"], "packets": r["packets"],
        "bytes": r["bytes"], "duration_s": round(r["duration_s"], 2),
        "avg_pps": round(r["avg_pps"], 2),
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


def print_report(r):
    ps, iat = r["packet_size"], r["iat_s"]
    print("=" * 72)
    print(f"{r['file']}  (host {r['host_ip']})")
    print("=" * 72)
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
        print(f"DNS: {r['dns']['queries_total']} query, {r['dns']['unique_names']} nomi unici; "
              + ", ".join(k for k, _ in r["dns"]["top_names"][:5]))
    print()


def aggregate(results):
    tot_pkts = sum(r["packets"] for r in results)
    tot_bytes = sum(r["bytes"] for r in results)
    # media pesata della dimensione pacchetto
    mean_size = tot_bytes / tot_pkts if tot_pkts else 0
    durs = [r["duration_s"] for r in results]
    services = Counter()
    for r in results:
        for k, v in r["top_services_pkts"]:
            services[k] += v
    l4 = Counter()
    for r in results:
        l4.update(r["l4"])
    return {
        "hosts": len(results),
        "total_packets": tot_pkts,
        "total_bytes": tot_bytes,
        "mean_packet_size": mean_size,
        "packets_per_host": describe([r["packets"] for r in results]),
        "duration_s": describe(durs),
        "avg_pps_per_host": describe([r["avg_pps"] for r in results]),
        "avg_bps_per_host": describe([r["avg_throughput_bps"] for r in results]),
        "total_flows": sum(r["flows"]["count"] for r in results),
        "l4_total": dict(l4),
        "top_services_pkts": services.most_common(10),
    }


# ------------------------------------------------------------------ main --
def main():
    default_dir = os.path.join(os.environ.get("PROGETTO", "."), "traffics", "scenario_000")
    ap = argparse.ArgumentParser(description="Statistiche sui pcap di traffico benevolo")
    ap.add_argument("-d", "--dir", default=default_dir, help=f"cartella pcap (default: {default_dir})")
    ap.add_argument("-o", "--out", default="pcap_stats", help="cartella di output")
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.dir, "traffic_10_0_0_*.pcap")),
        key=lambda p: int(re.search(r"_(\d+)\.pcap$", p).group(1)),
    )
    if not files:
        sys.exit(f"Nessun pcap trovato in {args.dir}")

    os.makedirs(args.out, exist_ok=True)
    results = []
    for f in files:
        print(f"[+] Analizzo {os.path.basename(f)} ...", file=sys.stderr)
        r = analyze(f)
        results.append(r)
        print_report(r)

    agg = aggregate(results)
    print("#" * 72)
    print("AGGREGATO SCENARIO")
    print("#" * 72)
    print(f"Host: {agg['hosts']}  Pacchetti totali: {agg['total_packets']}  "
          f"Byte totali: {fmt_bytes(agg['total_bytes'])}")
    print(f"Dimensione media pacchetto (pesata): {agg['mean_packet_size']:.1f} B")
    print(f"Pacchetti per host: media {agg['packets_per_host']['mean']:.0f}  "
          f"min {agg['packets_per_host']['min']}  max {agg['packets_per_host']['max']}")
    print(f"Durata per host [s]: media {agg['duration_s']['mean']:.1f}  "
          f"min {agg['duration_s']['min']:.1f}  max {agg['duration_s']['max']:.1f}")
    print(f"Rate medio per host: {agg['avg_pps_per_host']['mean']:.2f} pps, "
          f"{agg['avg_bps_per_host']['mean']/1000:.2f} kbit/s")
    print(f"Flussi totali: {agg['total_flows']}   L4: {agg['l4_total']}")
    print("Top servizi:", ", ".join(f"{k}={v}" for k, v in agg["top_services_pkts"][:8]))

    # CSV riassuntivo
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([h for _, h in SUMMARY_COLS])
        for r in results:
            row = flat_row(r)
            w.writerow([row[k] for k, _ in SUMMARY_COLS])

    # JSON dettagliato
    json_path = os.path.join(args.out, "details.json")
    with open(json_path, "w") as fh:
        json.dump({"aggregate": agg, "per_host": results}, fh, indent=2, default=str)

    print(f"\nSalvati: {csv_path}, {json_path}")


if __name__ == "__main__":
    main()