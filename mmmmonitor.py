#!/usr/bin/env python3

"""
SDN Network Monitor
====================
Monitora in continuo la rete SDN descritta in main_topo.py raccogliendo:
  - Numero totale di flow rule installate su ciascuno switch (s1, s2, s3)
  - Numero di flow rule "valide" (che matchano gli host reali h1-h4, quindi
    distinguibili da eventuali entry spurie iniettate da un attacco di
    flow table flooding)
  - Latenza RTT tra le coppie di host (h1-h3, h1-h4, h2-h3, h2-h4)
 
A differenza della versione precedente, questo monitor NON gestisce più
l'attacco: è pensato per essere avviato dalla CLI di Mininet in modo NON
bloccante. Una volta avviato continua a campionare in background su due
thread separati, lasciando la CLI libera per lanciare nel frattempo un
attacco (es. FloodAttack) o qualsiasi altro comando.
 
Quando viene fermato, genera automaticamente un report testuale su file
con l'andamento temporale di tutte le metriche raccolte.
 
Utilizzo dalla CLI Mininet (workflow consigliato)
--------------------------------------------------
    mininet> py monitor = start_monitor(net)
    mininet> py FloodAttack(switch=s2, iface=s2.intfList()[2], npacket=1000, threads=4, bsize=100, bdelay=1).start()
    ... attendere, eseguire ping, ecc. ...
    mininet> py monitor.stop()
 
Il monitor scrive il path del report appena generato. Si può anche fissare
una durata massima, dopo la quale si ferma da solo:
    mininet> py monitor = start_monitor(net, duration=60)
 
Esiste anche una variante bloccante e self-contained, utile per uno
snapshot temporizzato senza dover gestire manualmente start/stop:
    mininet> py monitor_sdn(duration=60)
"""


import os
import time
import threading
import subprocess
import statistics
from datetime import datetime
from typing import Optional, List, Tuple, Dict

from mininet.net import Mininet
from mininet.log import info, error

SWITCHES         = ['s1', 's2', 's3']
PING_PAIRS       = [('h1', 'h3'), ('h1', 'h4'), ('h2', 'h3'), ('h2', 'h4')]
PING_COUNT       = 3
MONITOR_INTERVAL = 1.0
HOST_MACS        = ['00:00:00:00:00:01', '00:00:00:00:00:02',
                     '00:00:00:00:00:03', '00:00:00:00:00:04']
OUTPUT_DIR       = os.path.expanduser('~/Desktop/progetto/reports')

def get_flow_count(switch_name: str) -> int:
    """Numero totale di flow rule installate sullo switch (incluse quelle di
    default, es. table-miss, ed eventuali entry iniettate da un attacco)."""
    try:
        result = subprocess.run(
            ['ovs-ofctl', '-O', 'OpenFlow14', 'dump-flows', switch_name],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            error(f"[monitor] Errore ovs-ofctl su {switch_name}: {result.stderr}\n")
            return -1
        lines = [l for l in result.stdout.strip().splitlines()
                 if l.strip() and not l.startswith('OFPST_FLOW')]
        return len(lines)
    except Exception as exc:
        error(f"[monitor] Eccezione get_flow_count({switch_name}): {exc}\n")
        return -1

def get_valid_flow_count(switch_name: str) -> int:
    """Numero di flow rule che matchano effettivamente uno dei MAC degli
    host reali della rete (h1-h4). Utile per distinguere il traffico
    legittimo da eventuali entry spurie iniettate da un attacco."""
    try:
        mac_pattern = '|'.join(HOST_MACS)
        comando = (
            f"ovs-ofctl -O OpenFlow14 dump-flows {switch_name} | "
            f"grep -Ei '({mac_pattern})' | wc -l"
        )
        result = subprocess.run(
            comando, shell=True, capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            error(f"[monitor] Errore grep su {switch_name}: {result.stderr}\n")
            return -1
        return int(result.stdout.strip())
    except Exception as exc:
        error(f"[monitor] Eccezione get_valid_flow_count({switch_name}): {exc}\n")
        return -1

def snapshot_flows(label: str = "") -> Dict[str, Dict[str, int]]:
    """
    Acquisisce numero totale e numero di flow valide su tutti gli switch.
    Restituisce {switch: {'total': n, 'valid': n}}.
    """
    if label:
        info(f"\n--- FLOW TABLE SNAPSHOT — {label} ---\n")
    counts = {}
    for sw in SWITCHES:
        total = get_flow_count(sw)
        valid = get_valid_flow_count(sw)
        counts[sw] = {'total': total, 'valid': valid}
        if label:
            info(f"  {sw}: totale={total}  valide={valid}\n")
    return counts


# ---------------------------------------------------------------------------
# Raccolta latenza (RTT ping)
# ---------------------------------------------------------------------------

def measure_latency(net: Mininet, src_name: str, dst_name: str,
                    count: int = PING_COUNT) -> Optional[float]:
    """Esegue `count` ping da src a dst e restituisce l'RTT medio in ms."""
    src = net.get(src_name)
    dst = net.get(dst_name)
    if src is None or dst is None:
        error(f"[monitor] Host non trovato: {src_name} o {dst_name}\n")
        return None

    dst_ip = dst.IP()
    result = src.cmd(f'ping -c {count} -W 1 {dst_ip}')

    for line in result.splitlines():
        if 'rtt min/avg/max' in line or 'round-trip min/avg/max' in line:
            try:
                stats_part = line.split('=')[1].strip()
                avg_rtt = float(stats_part.split('/')[1])
                return avg_rtt
            except (IndexError, ValueError):
                pass
    return None


def snapshot_latency(net: Mininet, label: str = "") -> Dict[str, Optional[float]]:
    """Misura la latenza RTT per tutte le coppie in PING_PAIRS."""
    if label:
        info(f"\n--- LATENZA RTT SNAPSHOT — {label} ---\n")
    results = {}
    for src_name, dst_name in PING_PAIRS:
        rtt = measure_latency(net, src_name, dst_name)
        key = f"{src_name}→{dst_name}"
        results[key] = rtt
        if label:
            rtt_str = f"{rtt:.3f} ms" if rtt is not None else "N/A (loss)"
            info(f"  {key}: {rtt_str}\n")
    return results


# ---------------------------------------------------------------------------
# Monitoraggio continuo in background
# ---------------------------------------------------------------------------

class NetworkMonitor:
    """
    Campiona, in due thread paralleli indipendenti, flow rule (totali e
    valide) e latenza RTT ad intervalli regolari, finché non viene chiamato
    stop() oppure scade la durata massima (duration).

    Pensato per essere avviato dalla CLI Mininet senza bloccarla: start()
    lancia i thread come daemon e ritorna immediatamente, lasciando la CLI
    libera per lanciare nel frattempo un attacco o altri comandi.
    """

    def __init__(self, net: Mininet,
                 interval: float = MONITOR_INTERVAL,
                 duration: Optional[float] = None,
                 label: str = "monitor"):
        self.net      = net
        self.interval = interval
        self.duration = duration
        self.label    = label
        self._stop    = threading.Event()
        self._start_time: Optional[float] = None

        self.flow_samples:    List[Tuple[float, Dict[str, Dict[str, int]]]] = []
        self.latency_samples: List[Tuple[float, Dict[str, Optional[float]]]] = []

        self._t_flow    = threading.Thread(target=self._flow_loop,    daemon=True)
        self._t_latency = threading.Thread(target=self._latency_loop, daemon=True)

    # ---- thread flow --------------------------------------------------
    def _flow_loop(self):
        t0 = self._start_time
        info(f"[monitor:{self.label}] ▶ Thread flow avviato\n")
        while not self._stop.is_set():
            elapsed = time.time() - t0
            if self.duration is not None and elapsed >= self.duration:
                break
            counts = {sw: {'total': get_flow_count(sw),
                            'valid': get_valid_flow_count(sw)} for sw in SWITCHES}
            self.flow_samples.append((elapsed, counts))
            per_sw = "  ".join(
                f"{sw}=tot:{counts[sw]['total']}/val:{counts[sw]['valid']}"
                for sw in SWITCHES
            )
            info(f"  [flow t={elapsed:6.1f}s] {per_sw}\n")
            self._stop.wait(timeout=self.interval)
        info(f"[monitor:{self.label}] ■ Thread flow terminato\n")

    # ---- thread latenza -------------------------------------------------
    def _latency_loop(self):
        t0 = self._start_time
        info(f"[monitor:{self.label}] ▶ Thread latenza avviato\n")
        while not self._stop.is_set():
            elapsed = time.time() - t0
            if self.duration is not None and elapsed >= self.duration:
                break
            rtts = {}
            for src, dst in PING_PAIRS:
                key = f"{src}→{dst}"
                rtts[key] = measure_latency(self.net, src, dst)
            self.latency_samples.append((elapsed, rtts))

            valid = [v for v in rtts.values() if v is not None]
            avg_str = f"{statistics.mean(valid):.3f} ms" if valid else "N/A"
            losses = sum(1 for v in rtts.values() if v is None)
            info(f"  [rtt  t={elapsed:6.1f}s] media={avg_str:<10} loss={losses}/{len(PING_PAIRS)}\n")
            self._stop.wait(timeout=self.interval)
        info(f"[monitor:{self.label}] ■ Thread latenza terminato\n")

    # ---- controllo --------------------------------------------------------
    def start(self) -> 'NetworkMonitor':
        self._start_time = time.time()
        self._t_flow.start()
        self._t_latency.start()
        info(f"[monitor:{self.label}] Avviato (intervallo={self.interval}s"
             f"{f', durata={self.duration}s' if self.duration else ''}).\n")
        return self

    def wait(self, timeout: Optional[float] = None) -> None:
        """Blocca finché entrambi i thread non terminano (utile solo se è
        stata impostata una `duration`, altrimenti non termineranno mai)."""
        self._t_flow.join(timeout=timeout)
        self._t_latency.join(timeout=timeout)

    def stop(self, write_report: bool = True, output_dir: str = OUTPUT_DIR) -> Optional[str]:
        """Ferma il monitor, attende la chiusura dei thread e (default)
        genera un report su file. Restituisce il path del report."""
        self._stop.set()
        self._t_flow.join(timeout=10)
        self._t_latency.join(timeout=10)
        info(f"[monitor:{self.label}] Fermato. "
             f"Campioni raccolti: flow={len(self.flow_samples)}, "
             f"latenza={len(self.latency_samples)}\n")
        if write_report:
            return generate_report(self, output_dir=output_dir)
        return None

    def is_running(self) -> bool:
        return not self._stop.is_set()


# ---------------------------------------------------------------------------
# Generazione report su file
# ---------------------------------------------------------------------------

def generate_report(monitor: NetworkMonitor, output_dir: str = OUTPUT_DIR) -> str:
    """
    Scrive su file un report testuale con l'andamento temporale di flow rule
    (totali e valide) e latenza RTT raccolti da monitor.
    Restituisce il path del file generato.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f"monitor_report_{monitor.label}_{timestamp}.txt")

    duration_str = (f"{monitor.flow_samples[-1][0]:.1f}s"
                     if monitor.flow_samples else "N/A")

    with open(path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  SDN NETWORK MONITOR — REPORT\n")
        f.write(f"  Label: {monitor.label}\n")
        f.write(f"  Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Intervallo campionamento: {monitor.interval}s\n")
        f.write(f"  Durata totale monitoraggio: {duration_str}\n")
        f.write("=" * 70 + "\n\n")

        # --- Flow rules ---
        f.write("[1] FLOW RULES (totale/valide) NEL TEMPO\n")
        f.write("-" * 70 + "\n")
        header = (f"{'t (s)':>8}  " +
                  "  ".join(f"{sw:>14}" for sw in SWITCHES) +
                  f"  {'TOT/VAL':>12}\n")
        f.write(header)
        f.write("-" * 70 + "\n")
        for elapsed, counts in monitor.flow_samples:
            row = f"{elapsed:>8.1f}  "
            tot_sum = val_sum = 0
            for sw in SWITCHES:
                t = counts[sw]['total']
                v = counts[sw]['valid']
                row += f"{f'{t}/{v}':>14}  "
                if t >= 0:
                    tot_sum += t
                if v >= 0:
                    val_sum += v
            row += f"{f'{tot_sum}/{val_sum}':>12}\n"
            f.write(row)

        if monitor.flow_samples:
            tot_peak = max(sum(c[sw]['total'] for sw in SWITCHES if c[sw]['total'] >= 0)
                            for _, c in monitor.flow_samples)
            val_peak = max(sum(c[sw]['valid'] for sw in SWITCHES if c[sw]['valid'] >= 0)
                            for _, c in monitor.flow_samples)
            f.write("-" * 70 + "\n")
            f.write(f"Picco flow totali: {tot_peak}    Picco flow valide: {val_peak}\n")

        # --- Latenza ---
        f.write("\n[2] LATENZA RTT (ms) NEL TEMPO\n")
        f.write("-" * 70 + "\n")
        keys = [f"{s}→{d}" for s, d in PING_PAIRS]
        header2 = (f"{'t (s)':>8}  " +
                   "  ".join(f"{k:>10}" for k in keys) +
                   f"  {'media':>8}  {'loss':>6}\n")
        f.write(header2)
        f.write("-" * 70 + "\n")
        for elapsed, rtts in monitor.latency_samples:
            row = f"{elapsed:>8.1f}  "
            for k in keys:
                v = rtts.get(k)
                row += f"{(f'{v:.2f}' if v is not None else 'loss'):>10}  "
            valid = [v for v in rtts.values() if v is not None]
            avg_str = f"{statistics.mean(valid):.2f}" if valid else "N/A"
            losses = sum(1 for v in rtts.values() if v is None)
            row += f"{avg_str:>8}  {losses}/{len(keys):<4}\n"
            f.write(row)

        all_rtts = [v for _, r in monitor.latency_samples for v in r.values() if v is not None]
        if all_rtts:
            f.write("-" * 70 + "\n")
            f.write(f"RTT min/avg/max sull'intero periodo: "
                    f"{min(all_rtts):.3f} / {statistics.mean(all_rtts):.3f} / {max(all_rtts):.3f} ms\n")

        f.write("\n" + "=" * 70 + "\n")

    info(f"[monitor] Report salvato in: {path}\n")
    return path


# ---------------------------------------------------------------------------
# Funzioni di convenienza per la CLI Mininet
# ---------------------------------------------------------------------------

def start_monitor(net: Optional[Mininet] = None,
                  interval: float = MONITOR_INTERVAL,
                  duration: Optional[float] = None,
                  label: str = "manual") -> Optional[NetworkMonitor]:
    """
    Avvia il monitoraggio in background e ritorna immediatamente l'oggetto
    NetworkMonitor, lasciando la CLI Mininet libera per lanciare un attacco
    o altri comandi nel frattempo.

    Esempio dalla CLI:
        mininet> py monitor = start_monitor(net)
        mininet> py FloodAttack(switch=s2, iface=s2.intfList()[2], npacket=1000, threads=4, bsize=100, bdelay=1).start()
        ... attendere, fare ping, ecc. ...
        mininet> py monitor.stop()
    """
    if net is None:
        import builtins as _b
        net = getattr(_b, 'net', None)
        if net is None:
            error("[monitor] Impossibile ottenere l'istanza Mininet.\n")
            return None
    m = NetworkMonitor(net, interval=interval, duration=duration, label=label)
    m.start()
    return m


def monitor_sdn(net: Optional[Mininet] = None,
               duration: float = 60.0,
               interval: float = MONITOR_INTERVAL) -> str:
    """
    Versione bloccante e self-contained: avvia il monitor, campiona per
    `duration` secondi, poi genera e salva il report.

    NON gestisce alcun attacco: se vuoi correlare i dati a un attacco,
    lancialo tu manualmente (es. in un'altra finestra/xterm) mentre questa
    funzione gira, oppure usa il workflow non bloccante
    start_monitor()/monitor.stop().

    Restituisce il path del file di report generato.
    """
    if net is None:
        import builtins as _b
        net = getattr(_b, 'net', None)
        if net is None:
            error("[monitor] Impossibile ottenere l'istanza Mininet.\n")
            return ""

    info(f"\n*** SDN Network Monitor — avvio (durata={duration}s) ***\n")
    m = NetworkMonitor(net, interval=interval, duration=duration, label="auto")
    m.start()
    m.wait()
    return m.stop()


if __name__ == '__main__':
    print("Questo script è pensato per essere importato o usato dalla CLI Mininet.")
    print()
    print("Workflow consigliato (non bloccante, attacco indipendente):")
    print("  mininet> py monitor = start_monitor(net)")
    print("  mininet> py FloodAttack(switch=s2, iface=s2.intfList()[2], "
          "npacket=1000, threads=4, bsize=100, bdelay=1).start()")
    print("  ... attendere / fare ping ...")
    print("  mininet> py monitor.stop()")
    print()
    print("Workflow alternativo (bloccante, durata fissa):")
    print("  mininet> py monitor_sdn(duration=60)")