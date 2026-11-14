#!/usr/bin/env python3
"""
Monitor Mininet per i test online (confronto FTFM vs controller base).

Campiona in due thread: flow rule (totali/valide) e latenza ping (RTT medio e
packet loss per coppia). Ogni campione ha un timestamp epoch (`ts`), allineabile
alla colonna `timestamp` del report dei controller.

Output in OUTPUT_DIR, con prefisso <label>_<data>:
  _flows.csv, _latency.csv, _events.csv, _summary.txt

Uso dalla CLI Mininet (label = nome del controller sotto test):
    mininet> py monitor = start_monitor(net, label="ftfm")      # oppure "baseline"
    mininet> py monitor.mark("attack_start")
    mininet> py FloodAttack(...).start()
    mininet> py monitor.mark("attack_end")
    mininet> py monitor.stop()
"""
import os
import re
import csv
import time
import threading
import subprocess
import statistics
from datetime import datetime
from typing import Optional, List

from mininet.net import Mininet
from mininet.log import info, error

import signal

_PING_RE = re.compile(r'^\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=([\d.]+) ms')
_TX_RE = re.compile(r'(\d+) packets transmitted')
MONITOR_INTERVAL = 1.0
HOST_MACS = ['00:00:00:00:00:01', '00:00:00:00:00:02',
            '00:00:00:00:00:03', '00:00:00:00:00:04']

# Una flow e' "valida" se matcha un MAC degli host reali (o e' la table-miss verso il controller)
_VALID_RE = re.compile('|'.join(HOST_MACS) + '|actions=CONTROLLER', re.I)


def get_flow_counts(switch_name: str, valid_re=_VALID_RE):
    """(totale, valide) delle flow rule dello switch con una sola chiamata a
    ovs-ofctl. Il totale include table-miss ed eventuali entry iniettate
    dall'attacco. Restituisce (-1, -1) in caso di errore."""
    try:
        r = subprocess.run(['ovs-ofctl', '-O', 'OpenFlow14', 'dump-flows', switch_name],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            error(f"[monitor] ovs-ofctl su {switch_name}: {r.stderr}\n")
            return -1, -1
        lines = [l for l in r.stdout.splitlines()
                 if l.strip() and not l.startswith('OFPST_FLOW')]
        return len(lines), sum(1 for l in lines if valid_re.search(l))
    except Exception as exc:
        error(f"[monitor] get_flow_counts({switch_name}): {exc}\n")
        return -1, -1

class NetworkMonitor:
    """Campiona flow rule e latenza finche' non viene chiamato stop() o scade
    `duration`. start() ritorna subito: la CLI Mininet resta libera."""

    def __init__(self, net: Mininet, switches, ping_pairs, out_prefix, interval: float = MONITOR_INTERVAL,
                 duration: Optional[float] = None, label: str = "monitor", host_macs=None):
        """switches / ping_pairs / host_macs: se omessi si usano le costanti del
        modulo. out_prefix: path (senza suffisso) dei file di output"""
        self.net, self.interval, self.duration, self.label = net, interval, duration, label
        self.switches = list(switches)
        self.ping_pairs = list(ping_pairs)
        self.valid_re = (re.compile('|'.join(host_macs) + '|actions=CONTROLLER', re.I)
                         if host_macs else _VALID_RE)
        self.out_prefix = out_prefix
        self._stopped, self._report = False, None
        self._stop = threading.Event()
        self._t0: Optional[float] = None
        self.flow_rows: List[tuple] = []   # (ts, switch, total, valid)
        self.lat_rows: List[tuple] = []    # (ts, src, dst, rtt_ms, loss_pct)
        self.ping_interval = 1.0
        self.ping_procs = []     # (src, dst, file, popen, path)
        self._flows_thread = threading.Thread(target=self._loop, args=("flow", self._sample_flows), daemon=True)

    def _loop(self, name, sample):
        info(f"[monitor:{self.label}] ▶ thread {name} avviato\n")
        while not self._stop.is_set():
            now = time.time()
            t = now - self._t0
            if self.duration is not None and t >= self.duration:
                break
            sample(now, t)
            self._stop.wait(timeout=self.interval)
        info(f"[monitor:{self.label}] ■ thread {name} terminato\n")

    def _sample_flows(self, ts, t):
        parts = []
        for sw in self.switches:
            total, valid = get_flow_counts(sw, self.valid_re)
            self.flow_rows.append((round(ts, 3), sw, total, valid))
            parts.append(f"{sw}=tot:{total}/val:{valid}")
        info(f"  [flow t={t:6.1f}s] {'  '.join(parts)}\n")

    def start_pings(net, out_dir):
        procs = []
        for s, d in PING_PAIRS:
            h, dst = net.get(s), net.get(d)
            f = open(f"{out_dir}/ping_{s}_{d}.txt", "w")
            procs.append((f, h.popen(["ping", "-D", "-i", "1", dst.IP()],
                                    stdout=f, stderr=f)))
        return procs

    def _start_pings(self):
        base = self.out_prefix
        d = f"{base}_ping"
        os.makedirs(d, exist_ok=True)
        for s, dn in self.ping_pairs:
            src, dst = self.net.get(s), self.net.get(dn)
            if src is None or dst is None:
                error(f"[monitor] Host non trovato: {s} o {dn}\n")
                continue
            path = f"{d}/{s}_{dn}.txt"
            f = open(path, "w")
            p = src.popen(["ping", "-D", "-i", str(self.ping_interval), "-W", "1", dst.IP()],
                        stdout=f, stderr=subprocess.STDOUT)
            self.ping_procs.append((s, dn, f, p, path))

    def _collect_pings(self):
        for s, dn, f, p, path in self.ping_procs:
            try:
                p.send_signal(signal.SIGINT)      # SIGINT: ping stampa il riepilogo finale
                p.wait(timeout=5)
            except Exception:
                p.kill()
            f.close()
            got, ref, total = {}, None, 0
            for line in open(path):
                m = _PING_RE.match(line)
                if m:
                    ts, seq, rtt = float(m.group(1)), int(m.group(2)), float(m.group(3))
                    got[seq] = (ts, rtt)
                    ref = ref or (ts, seq)
                m = _TX_RE.search(line)
                if m:
                    total = int(m.group(1))
            if not got:
                continue
            ts_ref, seq_ref = ref
            for seq in range(1, max(max(got), total) + 1):
                if seq in got:
                    ts, rtt = got[seq]
                    self.lat_rows.append((round(ts, 3), s, dn, rtt, 0.0))
                else:
                    ts = ts_ref + (seq - seq_ref) * self.ping_interval   # stima per i persi
                    self.lat_rows.append((round(ts, 3), s, dn, '', 100.0))
        self.lat_rows.sort(key=lambda r: r[0])

    def start(self) -> 'NetworkMonitor':
        self._t0 = time.time()
        self._start_pings()
        self._flows_thread.start()
        info(f"[monitor:{self.label}] Avviato (intervallo={self.interval}s"
             f"{f', durata={self.duration}s' if self.duration else ''}).\n")
        return self

    def wait(self, timeout: Optional[float] = None) -> None:
        self._flows_thread.join(timeout=timeout)

    def stop(self, write_report: bool = True) -> Optional[str]:
        """Ferma il monitor e (default) scrive i file. Restituisce il prefisso dei file."""
        if self._stopped:       # idempotente: si puo' chiamare anche nel finally
            return self._report
        self._stopped = True
        self._stop.set()
        self.wait(timeout=15)    # un campione di latenza puo' durare qualche secondo
        self._collect_pings()
        info(f"[monitor:{self.label}] Fermato. Campioni: flow={len(self.flow_rows)}, "
             f"latenza={len(self.lat_rows)}\n")
        self._report = generate_report(self) if write_report else None
        return self._report

    def is_running(self) -> bool:
        return not self._stop.is_set()


def _summary(m: NetworkMonitor) -> List[str]:
    dur = (m.flow_rows[-1][0] - m.flow_rows[0][0]) if m.flow_rows else None
    out = [f"Label: {m.label}",
           f"Intervallo: {m.interval}s",
           f"Durata: {dur:.1f}s" if dur is not None else "Durata: N/A"]
    for sw in m.switches:
        rows = [r for r in m.flow_rows if r[1] == sw]
        if rows:
            out.append(f"{sw}: picco totali={max(r[2] for r in rows)}  "
                       f"picco valide={max(r[3] for r in rows)}")
    rtts = [r[3] for r in m.lat_rows if r[3] != '']
    if rtts:
        p95 = statistics.quantiles(rtts, n=20)[-1] if len(rtts) > 1 else rtts[0]
        out.append(f"RTT ms min/avg/p95/max: {min(rtts):.3f} / {statistics.mean(rtts):.3f} / "
                   f"{p95:.3f} / {max(rtts):.3f}")
    if m.lat_rows:
        out.append(f"Packet loss medio: {statistics.mean(r[4] for r in m.lat_rows):.1f}%  "
                   f"(pacchetti persi: {sum(1 for r in m.lat_rows if r[4] >= 100)}/{len(m.lat_rows)})")
    return out


def generate_report(m: NetworkMonitor) -> str:
    base = m.out_prefix
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for suffix, header, rows in (
        ("flows",   ["ts", "switch", "flows_total", "flows_valid"],        m.flow_rows),
        ("latency", ["ts", "src", "dst", "rtt_ms", "loss_pct"],           m.lat_rows),
    ):
        with open(f"{base}_{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    summary = "\n".join(_summary(m))
    with open(f"{base}_summary.txt", "w") as f:
        f.write(summary + "\n")
    info(f"[monitor] Report salvato in: {base}_*\n{summary}\n")
    return base


# ---------------------------------------------------------------------------
# Funzioni di convenienza per la CLI Mininet
# ---------------------------------------------------------------------------
def _get_net(net: Optional[Mininet]) -> Optional[Mininet]:
    if net is None:
        import builtins
        net = getattr(builtins, 'net', None)
        if net is None:
            error("[monitor] Impossibile ottenere l'istanza Mininet.\n")
    return net


def start_monitor(net: Optional[Mininet] = None, interval: float = MONITOR_INTERVAL,
                  duration: Optional[float] = None, label: str = "manual",
                  **kwargs) -> Optional[NetworkMonitor]:
    """Avvia il monitor in background (non bloccante). kwargs: vedi NetworkMonitor."""
    net = _get_net(net)
    return NetworkMonitor(net, interval, duration, label, **kwargs).start() if net else None

def start_monitor(net: Optional[Mininet], switches, ping_pairs, out_prefix,
                  interval: float = MONITOR_INTERVAL,
                  duration: Optional[float] = None, label: str = "manual",
                  **kwargs) -> Optional[NetworkMonitor]:
    """Avvia il monitor in background (non bloccante)."""
    net = _get_net(net)
    return (NetworkMonitor(net, switches, ping_pairs, out_prefix,
                           interval=interval, duration=duration, label=label,
                           **kwargs).start() if net else None)

def monitor_sdn(net: Optional[Mininet] = None, duration: float = 60.0,
                interval: float = MONITOR_INTERVAL, label: str = "auto") -> str:
    """Versione bloccante: campiona per `duration` secondi e scrive il report."""
    net = _get_net(net)
    if net is None:
        return ""
    info(f"\n*** SDN Network Monitor — avvio (durata={duration}s) ***\n")
    m = NetworkMonitor(net, interval, duration, label).start()
    m.wait()
    return m.stop()


if __name__ == '__main__':
    print(__doc__)