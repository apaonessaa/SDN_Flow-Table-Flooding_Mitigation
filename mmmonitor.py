#!/usr/bin/env python3
"""
SDN Network Monitor
===================
Monitora la rete SDN descritta in main_topo.py raccogliendo:
  - Numero di flow rule nelle flow table degli switch (s1, s2, s3)
  - Latenza RTT tra coppie di host (h1-h3, h1-h4, h2-h3, h2-h4)

Confronta lo stato della rete PRIMA, DURANTE e DOPO un attacco Flow Table Flooding.

Utilizzo (dalla CLI Mininet):
    py monitor_sdn()               # monitoraggio completo con attacco
    py monitor_sdn(attack=False)   # solo snapshot senza attacco
"""

import time
import threading
import subprocess
import statistics
from typing import Optional
from mininet.net import Mininet
from mininet.log import info, error

from flow_table_flood import FloodAttack


# ---------------------------------------------------------------------------
# Parametri configurabili
# ---------------------------------------------------------------------------
SWITCHES         = ['s1', 's2', 's3']
PING_PAIRS       = [('h1', 'h3'), ('h1', 'h4'), ('h2', 'h3'), ('h2', 'h4')]
PING_COUNT       = 3          # numero di ping per coppia (ridotto per campionamento rapido)
MONITOR_INTERVAL = 1.0        # secondi tra campionamenti durante l'attacco


# ---------------------------------------------------------------------------
# Raccolta flow rule
# ---------------------------------------------------------------------------

def get_flow_count(switch_name: str) -> int:
    """Restituisce il numero di flow rule installate sullo switch via ovs-ofctl."""
    nn = test(switch_name)
    info(f'Numero di regole valide su {switch_name}: {nn}\n')
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

def test(switch_name: str) -> int:
    try:
        comando = (
            f"sudo ovs-ofctl -O OpenFlow14 dump-flows {switch_name} | "
            f"grep -Ei '(00:00:00:00:00:01|00:00:00:00:00:02|00:00:00:00:00:03|00:00:00:00:00:04)' | "
            f"wc -l"
        )
        result = subprocess.run(
            comando,
            shell=True, 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            error(f"[monitor] Errore ovs-ofctl su {switch_name}: {result.stderr}\n")
            return -1
        lines = int(result.stdout.strip())
        return lines
    except Exception as exc:
        error(f"[monitor] Eccezione test({switch_name}): {exc}\n")
        return -1


def snapshot_flows(label: str) -> dict:
    """
    Acquisisce il numero di flow rule su tutti gli switch.
    Stampa una tabella formattata e restituisce un dizionario {switch: count}.
    """
    info(f"\n{'='*50}\n")
    info(f"  FLOW TABLE SNAPSHOT — {label}\n")
    info(f"{'='*50}\n")
    info(f"  {'Switch':<10} {'Flow Rules':>12}\n")
    info(f"  {'-'*24}\n")

    counts = {}
    for sw in SWITCHES:
        n = get_flow_count(sw)
        counts[sw] = n
        info(f"  {sw:<10} {n:>12}\n")

    total = sum(v for v in counts.values() if v >= 0)
    info(f"  {'-'*24}\n")
    info(f"  {'TOTALE':<10} {total:>12}\n")
    info(f"{'='*50}\n")
    return counts


# ---------------------------------------------------------------------------
# Raccolta latenza (RTT ping)
# ---------------------------------------------------------------------------

def measure_latency(net: Mininet, src_name: str, dst_name: str,
                    count: int = PING_COUNT) -> Optional[float]:
    """
    Esegue `count` ping da src a dst e restituisce l'RTT medio in ms.
    Ritorna None in caso di errore o 100% packet loss.
    """
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

    for line in result.splitlines():
        if 'packet loss' in line and '100%' in line:
            return None

    return None


def snapshot_latency(net: Mininet, label: str) -> dict:
    """
    Misura la latenza RTT per tutte le coppie in PING_PAIRS.
    Stampa una tabella e restituisce {coppia: rtt_ms | None}.
    """
    info(f"\n{'='*50}\n")
    info(f"  LATENZA RTT SNAPSHOT — {label}\n")
    info(f"{'='*50}\n")
    info(f"  {'Coppia':<14} {'RTT medio (ms)':>16}\n")
    info(f"  {'-'*32}\n")

    results = {}
    for src_name, dst_name in PING_PAIRS:
        rtt = measure_latency(net, src_name, dst_name)
        key = f"{src_name}→{dst_name}"
        results[key] = rtt
        rtt_str = f"{rtt:.3f}" if rtt is not None else "N/A (loss)"
        info(f"  {key:<14} {rtt_str:>16}\n")

    valid_rtts = [v for v in results.values() if v is not None]
    if valid_rtts:
        avg = statistics.mean(valid_rtts)
        info(f"  {'-'*32}\n")
        info(f"  {'Media globale':<14} {avg:>13.3f} ms\n")

    info(f"{'='*50}\n")
    return results


# ---------------------------------------------------------------------------
# Monitoraggio continuo DURANTE l'attacco (flow + latenza in parallelo)
# ---------------------------------------------------------------------------

class AttackMonitor:
    """
    Campiona flow rule e latenza RTT in parallelo durante l'attacco.

    Due thread indipendenti scrivono nei rispettivi buffer:
      - flow_samples  : lista di (elapsed_s, {switch: count})
      - latency_samples: lista di (elapsed_s, {coppia: rtt_ms|None})

    I thread si fermano quando stop() viene chiamato o il timeout scade.
    """

    def __init__(self, net: Mininet,
                 interval: float = MONITOR_INTERVAL,
                 attacker=None):
        self.net      = net
        self.interval = interval
        self.attacker = attacker   # se fornito, il monitor segue la vita dell'attacco
        self._stop    = threading.Event()
        self.flow_samples:    list = []
        self.latency_samples: list = []
        self._t_flow    = threading.Thread(target=self._flow_loop,    daemon=True)
        self._t_latency = threading.Thread(target=self._latency_loop, daemon=True)

    # ---- thread flow -------------------------------------------------------
    def _flow_loop(self):
        t0 = time.time()
        info("\n[monitor] ▶ Thread flow avviato\n")
        while not self._stop.is_set():
            elapsed = time.time() - t0
            # Termina se: timeout superato OPPURE l'attacco è finito
            attack_done = self.attacker is not None and not self.attacker.is_running()
            if attack_done:
                break
            counts = {sw: get_flow_count(sw) for sw in SWITCHES}
            self.flow_samples.append((elapsed, counts))
            totale = sum(v for v in counts.values() if v >= 0)
            per_sw = "  ".join(f"{sw}={counts[sw]}" for sw in SWITCHES)
            info(f"  [flow  t={elapsed:5.1f}s]  totale={totale:>5}  {per_sw}\n")
            self._stop.wait(timeout=self.interval)
        info("[monitor] ■ Thread flow terminato\n")

    # ---- thread latenza ----------------------------------------------------
    def _latency_loop(self):
        t0 = time.time()
        info("[monitor] ▶ Thread latenza avviato\n")
        while not self._stop.is_set():
            elapsed = time.time() - t0
            attack_done = self.attacker is not None and not self.attacker.is_running()
            if attack_done:
                break
            rtts = {}
            for src, dst in PING_PAIRS:
                key = f"{src}→{dst}"
                rtts[key] = measure_latency(self.net, src, dst)
            self.latency_samples.append((elapsed, rtts))

            valid = [v for v in rtts.values() if v is not None]
            avg_str = f"{statistics.mean(valid):.3f} ms" if valid else "N/A"
            losses  = sum(1 for v in rtts.values() if v is None)
            info(f"  [rtt   t={elapsed:5.1f}s]  media={avg_str:<12}  loss={losses}/{len(PING_PAIRS)}\n")
            self._stop.wait(timeout=self.interval)
        info("[monitor] ■ Thread latenza terminato\n")

    # ---- controllo ---------------------------------------------------------
    def start(self):
        self._t_flow.start()
        self._t_latency.start()

    def stop(self):
        self._stop.set()
        self._t_flow.join(timeout=10)
        self._t_latency.join(timeout=10)

    def wait(self):
        self._t_flow.join()
        self._t_latency.join()


def print_during_summary(monitor: AttackMonitor) -> None:
    """Stampa un riepilogo aggregato dei campioni raccolti durante l'attacco."""
    info(f"\n{'='*60}\n")
    info(f"  RIEPILOGO MONITORAGGIO DURANTE L'ATTACCO\n")
    info(f"{'='*60}\n")

    # --- flow durante ---
    if monitor.flow_samples:
        info(f"\n  Flow rule totali nel tempo:\n")
        info(f"  {'t (s)':>8}  {'s1':>6}  {'s2':>6}  {'s3':>6}  {'TOTALE':>8}\n")
        info(f"  {'-'*40}\n")
        for elapsed, counts in monitor.flow_samples:
            tot = sum(v for v in counts.values() if v >= 0)
            info(f"  {elapsed:>8.1f}  "
                 + "  ".join(f"{counts.get(sw,0):>6}" for sw in SWITCHES)
                 + f"  {tot:>8}\n")
        # max flow osservati
        max_totals = [sum(v for v in c.values() if v >= 0)
                      for _, c in monitor.flow_samples]
        info(f"\n  Picco flow rule totali: {max(max_totals)}\n")

    # --- latenza durante ---
    if monitor.latency_samples:
        info(f"\n  Latenza RTT media nel tempo:\n")
        info(f"  {'t (s)':>8}  {'RTT medio (ms)':>16}  {'Loss':>6}\n")
        info(f"  {'-'*36}\n")
        for elapsed, rtts in monitor.latency_samples:
            valid = [v for v in rtts.values() if v is not None]
            losses = sum(1 for v in rtts.values() if v is None)
            avg_str = f"{statistics.mean(valid):.3f}" if valid else "N/A"
            info(f"  {elapsed:>8.1f}  {avg_str:>16}  {losses:>3}/{len(PING_PAIRS)}\n")

        all_rtts = [v for _, r in monitor.latency_samples
                    for v in r.values() if v is not None]
        if all_rtts:
            info(f"\n  RTT min/avg/max durante attacco: "
                 f"{min(all_rtts):.3f} / {statistics.mean(all_rtts):.3f} / {max(all_rtts):.3f} ms\n")

    info(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Confronto e report finale
# ---------------------------------------------------------------------------

def print_comparison(
    flows_before: dict,   flows_after: dict,
    latency_before: dict, latency_after: dict,
    monitor: Optional[AttackMonitor] = None
) -> None:
    """Stampa un report comparativo PRIMA / DURANTE / DOPO l'attacco."""

    info(f"\n{'#'*60}\n")
    info(f"  REPORT COMPARATIVO — PRIMA vs DOPO FLOW TABLE FLOOD\n")
    info(f"{'#'*60}\n")

    # --- Flow rules ---
    info("\n  [1] FLOW RULES PER SWITCH\n")
    info(f"  {'Switch':<10} {'Prima':>8} {'Dopo':>8} {'Delta':>8}\n")
    info(f"  {'-'*36}\n")
    for sw in SWITCHES:
        before = flows_before.get(sw, 0)
        after  = flows_after.get(sw, 0)
        delta  = after - before
        sign   = '+' if delta >= 0 else ''
        info(f"  {sw:<10} {before:>8} {after:>8} {sign}{delta:>7}\n")

    total_before = sum(v for v in flows_before.values() if v >= 0)
    total_after  = sum(v for v in flows_after.values() if v >= 0)
    delta_total  = total_after - total_before
    sign = '+' if delta_total >= 0 else ''
    info(f"  {'-'*36}\n")
    info(f"  {'TOTALE':<10} {total_before:>8} {total_after:>8} {sign}{delta_total:>7}\n")

    # Picco durante l'attacco
    if monitor and monitor.flow_samples:
        max_during = max(sum(v for v in c.values() if v >= 0)
                         for _, c in monitor.flow_samples)
        info(f"  {'PICCO':<10} {'':>8} {max_during:>8}\n")

    # --- Latenza ---
    info("\n  [2] LATENZA RTT (ms)\n")
    # Calcola media durante l'attacco per coppia
    during_avg: dict = {}
    if monitor and monitor.latency_samples:
        for key in [f"{s}→{d}" for s, d in PING_PAIRS]:
            vals = [r[key] for _, r in monitor.latency_samples
                    if key in r and r[key] is not None]
            during_avg[key] = statistics.mean(vals) if vals else None

    has_during = bool(during_avg)
    if has_during:
        info(f"  {'Coppia':<14} {'Prima':>9} {'Durante':>9} {'Dopo':>9} {'Δ tot':>9}\n")
        info(f"  {'-'*52}\n")
    else:
        info(f"  {'Coppia':<14} {'Prima':>10} {'Dopo':>10} {'Delta':>10}\n")
        info(f"  {'-'*46}\n")

    all_keys = set(latency_before) | set(latency_after)
    for key in sorted(all_keys):
        b = latency_before.get(key)
        a = latency_after.get(key)
        b_str = f"{b:.3f}" if b is not None else "N/A"
        a_str = f"{a:.3f}" if a is not None else "N/A"
        if b is not None and a is not None:
            d = a - b
            sign2 = '+' if d >= 0 else ''
            d_str = f"{sign2}{d:.3f}"
        else:
            d_str = "N/A"
        if has_during:
            dur = during_avg.get(key)
            dur_str = f"{dur:.3f}" if dur is not None else "N/A"
            info(f"  {key:<14} {b_str:>9} {dur_str:>9} {a_str:>9} {d_str:>9}\n")
        else:
            info(f"  {key:<14} {b_str:>10} {a_str:>10} {d_str:>10}\n")

    # --- Conclusioni ---
    info(f"\n{'#'*60}\n")
    if delta_total > 0:
        info(f"  ⚠  L'attacco ha iniettato {delta_total} flow rule aggiuntive.\n")
    else:
        info(f"  ✓  Nessuna variazione significativa nelle flow table.\n")

    valid_b = [v for v in latency_before.values() if v is not None]
    valid_a = [v for v in latency_after.values() if v is not None]
    if valid_b and valid_a:
        avg_b = statistics.mean(valid_b)
        avg_a = statistics.mean(valid_a)
        pct   = ((avg_a - avg_b) / avg_b * 100) if avg_b > 0 else 0
        sign2 = '+' if pct >= 0 else ''
        info(f"  ⏱  Latenza media: {avg_b:.3f} ms → {avg_a:.3f} ms "
             f"({sign2}{pct:.1f} %)\n")

    if has_during:
        valid_d = [v for v in during_avg.values() if v is not None]
        if valid_d and valid_b:
            avg_d = statistics.mean(valid_d)
            avg_b = statistics.mean(valid_b)
            pct_d = ((avg_d - avg_b) / avg_b * 100) if avg_b > 0 else 0
            sign2 = '+' if pct_d >= 0 else ''
            info(f"  ⏱  Latenza picco durante: {avg_d:.3f} ms "
                 f"({sign2}{pct_d:.1f} % vs baseline)\n")

    info(f"{'#'*60}\n\n")


# ---------------------------------------------------------------------------
# Funzione principale — chiamabile dalla CLI Mininet
# ---------------------------------------------------------------------------

def monitor_sdn(net: Optional[Mininet] = None, attack: bool = True) -> None:
    """
    Esegue il monitoraggio completo della rete SDN.

    Parametri
    ---------
    net    : istanza Mininet (se None, viene recuperata da builtins)
    attack : se True esegue l'attacco FloodAttack tra i due snapshot
    """
    if net is None:
        import builtins as _b
        net = getattr(_b, 'net', None)
        if net is None:
            error("[monitor] Impossibile ottenere l'istanza Mininet.\n")
            return

    info("\n*** SDN Network Monitor — avvio\n")

    # ---- SNAPSHOT PRE-ATTACCO ----
    flows_before   = snapshot_flows("PRE-ATTACCO")
    latency_before = snapshot_latency(net, "PRE-ATTACCO")

    atk_monitor: Optional[AttackMonitor] = None

    if attack:
        info("\n*** Avvio attacco Flow Table Flooding...\n")
        try:
            import builtins as _b
            _FloodAttack = getattr(_b, 'FloodAttack', None)
            if _FloodAttack is None:
                _FloodAttack = FloodAttack

            attacker = _FloodAttack(
                switch=net['s2'],
                iface=net['s2'].intfList()[2],
                npacket=10000,
                threads=4,
                bsize=50,
                bdelay=2
            )

            # Avvia il monitor PRIMA dell'attacco, così il primo campione
            # cattura lo stato iniziale appena l'attacco parte
            # Avvia l'attacco in modo NON bloccante PRIMA di creare il monitor,
            # così attacker.is_running() è già True quando i thread partono
            attacker.start()

            atk_monitor = AttackMonitor(net,
                                        interval=MONITOR_INTERVAL,
                                        attacker=attacker)
            atk_monitor.start()

            info(f"    Attacco e monitor avviati in parallelo "
                 f"(campione ogni {MONITOR_INTERVAL}s)\n\n")

            # Attende che entrambi i thread del monitor terminino.
            # I thread si fermano da soli quando l'attacco finisce
            # (attacker.is_running() → False).
            atk_monitor.wait()

            # Ferma l'attacco solo se è ancora in corso (timeout scaduto prima)
            if attacker.is_running():
                info("[monitor] Timeout raggiunto — fermo l'attacco.\n")
                attacker.stop()
            else:
                info("[monitor] Attacco completato — monitor terminato di conseguenza.\n")

            info("*** Fase di attacco terminata.\n")
            print_during_summary(atk_monitor)

        except Exception as exc:
            error(f"[monitor] Errore durante l'attacco: {exc}\n")
    else:
        info("\n[monitor] Modalità senza attacco — solo snapshot.\n")

    # ---- SNAPSHOT POST-ATTACCO ----
    flows_after   = snapshot_flows("POST-ATTACCO")
    latency_after = snapshot_latency(net, "POST-ATTACCO")

    # ---- REPORT FINALE ----
    print_comparison(flows_before, flows_after,
                     latency_before, latency_after,
                     monitor=atk_monitor)


# ---------------------------------------------------------------------------
# Entry point diretto
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Questo script è pensato per essere importato o usato dalla CLI Mininet.")
    print("Esempio d'uso:")
    print("  py monitor_sdn()              # monitoraggio con attacco")
    print("  py monitor_sdn(attack=False)  # solo snapshot")
