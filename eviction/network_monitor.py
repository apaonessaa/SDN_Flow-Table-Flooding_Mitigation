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

SWITCHES         = ['s1', 's2', 's3']
PING_PAIRS       = [('h1', 'h3'), ('h1', 'h4'), ('h2', 'h3'), ('h2', 'h4')]
PING_COUNT       = 3
MONITOR_INTERVAL = 1.0
HOST_MACS        = ['00:00:00:00:00:01', '00:00:00:00:00:02',
                    '00:00:00:00:00:03', '00:00:00:00:00:04']
OUTPUT_DIR       = os.path.expanduser('~/Desktop/progetto/reports')

# Una flow e' "valida" se matcha un MAC degli host reali (o e' la table-miss verso il controller)
_VALID_RE = re.compile('|'.join(HOST_MACS) + '|actions=CONTROLLER', re.I)


def get_flow_counts(switch_name: str):
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
        return len(lines), sum(1 for l in lines if _VALID_RE.search(l))
    except Exception as exc:
        error(f"[monitor] get_flow_counts({switch_name}): {exc}\n")
        return -1, -1


def measure_latency(net: Mininet, src_name: str, dst_name: str, count: int = PING_COUNT):
    """Ping src -> dst. Restituisce (RTT medio in ms o None, packet loss in %)."""
    src, dst = net.get(src_name), net.get(dst_name)
    if src is None or dst is None:
        error(f"[monitor] Host non trovato: {src_name} o {dst_name}\n")
        return None, 100.0
    out = src.cmd(f'ping -c {count} -i 0.2 -W 1 {dst.IP()}')
    loss = re.search(r'([\d.]+)% packet loss', out)
    rtt = re.search(r'= [\d.]+/([\d.]+)/', out)      # rtt min/avg/max/mdev = a/b/c/d
    return (float(rtt.group(1)) if rtt else None,
            float(loss.group(1)) if loss else 100.0)


class NetworkMonitor:
    """Campiona flow rule e latenza finche' non viene chiamato stop() o scade
    `duration`. start() ritorna subito: la CLI Mininet resta libera."""

    def __init__(self, net: Mininet, interval: float = MONITOR_INTERVAL,
                 duration: Optional[float] = None, label: str = "monitor"):
        self.net, self.interval, self.duration, self.label = net, interval, duration, label
        self._stop = threading.Event()
        self._t0: Optional[float] = None
        self.flow_rows: List[tuple] = []   # (ts, t, switch, total, valid)
        self.lat_rows: List[tuple] = []    # (ts, t, src, dst, rtt_ms, loss_pct)
        self.marks: List[tuple] = []       # (ts, t, text)
        self._threads = [
            threading.Thread(target=self._loop, args=("flow", self._sample_flows), daemon=True),
            threading.Thread(target=self._loop, args=("latenza", self._sample_latency), daemon=True),
        ]

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
        for sw in SWITCHES:
            total, valid = get_flow_counts(sw)
            self.flow_rows.append((round(ts, 3), round(t, 2), sw, total, valid))
            parts.append(f"{sw}=tot:{total}/val:{valid}")
        info(f"  [flow t={t:6.1f}s] {'  '.join(parts)}\n")

    def _sample_latency(self, ts, t):
        rtts, losses = [], []
        for src, dst in PING_PAIRS:
            rtt, loss = measure_latency(self.net, src, dst)
            self.lat_rows.append((round(ts, 3), round(t, 2), src, dst,
                                  '' if rtt is None else rtt, loss))
            if rtt is not None:
                rtts.append(rtt)
            losses.append(loss)
        avg = f"{statistics.mean(rtts):.3f} ms" if rtts else "N/A"
        info(f"  [rtt  t={t:6.1f}s] media={avg:<10} loss medio={statistics.mean(losses):.0f}%\n")

    def mark(self, text: str):
        """Segna un evento (es. attack_start / attack_end) per i grafici."""
        now = time.time()
        self.marks.append((round(now, 3), round(now - self._t0, 2), text))
        info(f"[monitor:{self.label}] evento: {text}\n")

    def start(self) -> 'NetworkMonitor':
        self._t0 = time.time()
        for th in self._threads:
            th.start()
        info(f"[monitor:{self.label}] Avviato (intervallo={self.interval}s"
             f"{f', durata={self.duration}s' if self.duration else ''}).\n")
        return self

    def wait(self, timeout: Optional[float] = None) -> None:
        for th in self._threads:
            th.join(timeout=timeout)

    def stop(self, write_report: bool = True, output_dir: str = OUTPUT_DIR) -> Optional[str]:
        """Ferma il monitor e (default) scrive i file. Restituisce il prefisso dei file."""
        self._stop.set()
        self.wait(timeout=15)    # un campione di latenza puo' durare qualche secondo
        info(f"[monitor:{self.label}] Fermato. Campioni: flow={len(self.flow_rows)}, "
             f"latenza={len(self.lat_rows)}\n")
        return generate_report(self, output_dir) if write_report else None

    def is_running(self) -> bool:
        return not self._stop.is_set()


def _summary(m: NetworkMonitor) -> List[str]:
    out = [f"Label: {m.label}",
           f"Intervallo: {m.interval}s",
           f"Durata: {m.flow_rows[-1][1]:.1f}s" if m.flow_rows else "Durata: N/A"]
    for sw in SWITCHES:
        rows = [r for r in m.flow_rows if r[2] == sw]
        if rows:
            out.append(f"{sw}: picco totali={max(r[3] for r in rows)}  "
                       f"picco valide={max(r[4] for r in rows)}")
    rtts = [r[4] for r in m.lat_rows if r[4] != '']
    if rtts:
        p95 = statistics.quantiles(rtts, n=20)[-1] if len(rtts) > 1 else rtts[0]
        out.append(f"RTT ms min/avg/p95/max: {min(rtts):.3f} / {statistics.mean(rtts):.3f} / "
                   f"{p95:.3f} / {max(rtts):.3f}")
    if m.lat_rows:
        out.append(f"Packet loss medio: {statistics.mean(r[5] for r in m.lat_rows):.1f}%  "
                   f"(ping con 100% loss: {sum(1 for r in m.lat_rows if r[5] >= 100)}/{len(m.lat_rows)})")
    return out


def generate_report(m: NetworkMonitor, output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"{m.label}_{datetime.now():%Y%m%d_%H%M%S}")
    for suffix, header, rows in (
        ("flows",   ["ts", "t", "switch", "flows_total", "flows_valid"], m.flow_rows),
        ("latency", ["ts", "t", "src", "dst", "rtt_ms", "loss_pct"],     m.lat_rows),
        ("events",  ["ts", "t", "event"],                                m.marks),
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
                  duration: Optional[float] = None, label: str = "manual") -> Optional[NetworkMonitor]:
    """Avvia il monitor in background (non bloccante)."""
    net = _get_net(net)
    return NetworkMonitor(net, interval, duration, label).start() if net else None


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
