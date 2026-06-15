import re
import time
import threading
import os
import signal

from mininet.log import info
from mininet.node import Host, OVSSwitch

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
IPERF_DURATION  = 300   # durata totale iperf (s) – interrotto dal monitor se
                         # l'utente fa Ctrl+C prima
IPERF_INTERVAL  = 1     # intervallo di report iperf (s)
POLL_INTERVAL   = 5     # ogni quanti secondi rileggere le metriche
PING_COUNT      = 20    # ping per ogni campione RTT


# ---------------------------------------------------------------------------
# Utilità: parsing
# ---------------------------------------------------------------------------
def _parse_iperf_line(line: str):
    """
    Estrae (bandwidth_Mbps, jitter_ms, loss_pct) da una riga UDP di iperf.
    Restituisce (None, None, None) se la riga non è riconosciuta.

    Formato atteso (iperf -u):
    [  3]  0.0- 1.0 sec  1.19 MBytes  10.0 Mbits/sec  0.123 ms    0/  850 (0%)
    """
    bw = jitter = loss = None

    bw_m = re.search(r'([\d.]+)\s+Mbits/sec', line)
    if bw_m:
        bw = float(bw_m.group(1))

    jitter_m = re.search(r'([\d.]+)\s+ms', line)
    if jitter_m:
        jitter = float(jitter_m.group(1))

    loss_m = re.search(r'\(\s*([\d.]+)%\)', line)
    if loss_m:
        loss = float(loss_m.group(1))

    return bw, jitter, loss


def _last_data_line(path: str):
    """Restituisce l'ultima riga non vuota del file, oppure ''."""
    try:
        with open(path) as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        return lines[-1] if lines else ''
    except FileNotFoundError:
        return ''


# ---------------------------------------------------------------------------
# Host stats (ping RTT)
# ---------------------------------------------------------------------------
def h_stats(src: Host, dst: Host, nping: int = PING_COUNT):
    """Esegue nping ping e restituisce l'RTT medio (ms) o None."""
    raw = src.cmd(f'ping -c {nping} -q {dst.IP()}')
    # Formato: rtt min/avg/max/mdev = 0.123/0.456/0.789/0.012 ms
    m = re.search(r'rtt min/avg/max/mdev = [\d.]+/([\d.]+)/', raw)
    if m:
        return float(m.group(1))
    # Fallback: media manuale
    matches = re.findall(r'time=([\d.]+)', raw)
    if matches:
        return sum(float(t) for t in matches) / len(matches)
    return None


def print_h_stats(src: Host, dst: Host, nping: int = PING_COUNT):
    rtt = h_stats(src, dst, nping)
    rtt_str = f"{rtt:.2f}" if rtt is not None else "timeout"
    print(f"{'Source':>10} | {'Destination':>11} | {'AVG RTT (ms)':>12} | pings")
    print("-" * 55)
    print(f"{src.name:>10} | {dst.name:>11} | {rtt_str:>12} | {nping}")


# ---------------------------------------------------------------------------
# Switch stats (flow rules)
# ---------------------------------------------------------------------------
def s_stats(sw: OVSSwitch) -> int:
    raw = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-flows {sw.name}')
    return sum(1 for line in raw.splitlines() if 'cookie=' in line)


def print_s_stats(sw: OVSSwitch):
    n = s_stats(sw)
    print(f"{'Switch':>8} | {'Flow rules':>10}")
    print("-" * 25)
    print(f"{sw.name:>8} | {n:>10}")


# ---------------------------------------------------------------------------
# iperf (avvia server + client UDP in background)
# ---------------------------------------------------------------------------
def iperf_start(client: Host, server: Host, bandwidth: str = '10M'):
    """
    Avvia iperf server sul server e client su client.
    I log vengono scritti in /tmp/iperf_{server.name}.txt
    """
    log = f'/tmp/iperf_{server.name}.txt'
    server.cmd('pkill -f "iperf -s" 2>/dev/null; true')
    client.cmd('pkill -f "iperf -c" 2>/dev/null; true')

    server.cmd(f'iperf -s -u -i {IPERF_INTERVAL} > {log} 2>&1 &')
    time.sleep(0.5)
    client.cmd(
        f'iperf -c {server.IP()} -u -b {bandwidth} '
        f'-t {IPERF_DURATION} -i {IPERF_INTERVAL} '
        f'>> {log} 2>&1 &'
    )
    return log


def iperf_stop(client: Host, server: Host):
    """Ferma server e client iperf."""
    server.cmd('pkill -f "iperf -s" 2>/dev/null; true')
    client.cmd('pkill -f "iperf -c" 2>/dev/null; true')


def iperf_read_last(log_path: str):
    """Legge l'ultima metrica disponibile dal log iperf."""
    line = _last_data_line(log_path)
    return _parse_iperf_line(line)


# ---------------------------------------------------------------------------
# Monitor thread
# ---------------------------------------------------------------------------
class NetworkMonitor:
    """
    Thread che campiona periodicamente le metriche e le accumula in liste.
    Viene fermato chiamando .stop() oppure quando l'evento _stop è settato.
    """

    def __init__(self, client: Host, serverA: Host, serverB: Host,
                 victim: OVSSwitch, poll: int = POLL_INTERVAL):
        self.client  = client
        self.serverA = serverA
        self.serverB = serverB
        self.victim  = victim
        self.poll    = poll

        # Strutture dati risultati
        self.timestamps: list[float] = []

        self.rtt_A:   list[float] = []
        self.rtt_B:   list[float] = []

        self.bw_A:     list[float] = []
        self.jitter_A: list[float] = []
        self.loss_A:   list[float] = []

        self.bw_B:     list[float] = []
        self.jitter_B: list[float] = []
        self.loss_B:   list[float] = []

        self.flows:    list[int]   = []

        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._log_A = ''
        self._log_B = ''
        self._t0    = time.time()

    # ------------------------------------------------------------------
    def start(self):
        info('[monitor] Avvio iperf verso serverA e serverB…\n')
        self._log_A = iperf_start(self.client, self.serverA)
        #self._log_B = iperf_start(self.client, self.serverB)
        self._t0    = time.time()
        self._thread.start()
        info('[monitor] Monitoraggio avviato. Usa Ctrl+C sulla CLI per fermare.\n')

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.poll + 2)
        iperf_stop(self.client, self.serverA)
        #iperf_stop(self.client, self.serverB)
        info('[monitor] Monitoraggio fermato.\n')

    # ------------------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            t = time.time() - self._t0
            self.timestamps.append(t)

            # RTT
            rA = h_stats(self.client, self.serverA, nping=PING_COUNT)
            rB = h_stats(self.client, self.serverB, nping=PING_COUNT)
            self.rtt_A.append(rA if rA is not None else float('nan'))
            self.rtt_B.append(rB if rB is not None else float('nan'))

            # iperf
            bwA, jA, lA = iperf_read_last(self._log_A)
            bwB, jB, lB = iperf_read_last(self._log_B)

            self.bw_A.append(bwA     if bwA     is not None else float('nan'))
            self.jitter_A.append(jA  if jA      is not None else float('nan'))
            self.loss_A.append(lA    if lA      is not None else float('nan'))

            self.bw_B.append(bwB     if bwB     is not None else float('nan'))
            self.jitter_B.append(jB  if jB      is not None else float('nan'))
            self.loss_B.append(lB    if lB      is not None else float('nan'))

            # Flow rules
            self.flows.append(s_stats(self.victim))

            info(
                f'[monitor] t={t:.0f}s | '
                f'rttA={self.rtt_A[-1]:.1f}ms rttB={self.rtt_B[-1]:.1f}ms | '
                f'flows={self.flows[-1]}\n'
            )

            self._stop.wait(self.poll)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_results(mon: NetworkMonitor, out_dir: str = '/tmp'):
    """Genera e salva i grafici dai dati raccolti."""
    try:
        import matplotlib
        matplotlib.use('Agg')          # non-interactive (nessun display richiesto)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('[monitor] matplotlib non trovato – installa con: pip install matplotlib')
        return

    ts = mon.timestamps
    if not ts:
        print('[monitor] Nessun dato raccolto, grafici non generati.')
        return

    os.makedirs(out_dir, exist_ok=True)
    x = np.array(ts)

    def _safe(lst):
        return np.array(lst, dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('Network Monitor – Risultati', fontsize=14, fontweight='bold')

    # --- RTT ---
    ax = axes[0]
    ax.plot(x, _safe(mon.rtt_A), label=f'RTT → {mon.serverA.name}', marker='o', ms=3)
    ax.plot(x, _safe(mon.rtt_B), label=f'RTT → {mon.serverB.name}', marker='s', ms=3)
    ax.set_ylabel('RTT (ms)')
    ax.set_title('Round-Trip Time')
    ax.legend(); ax.grid(True, alpha=.3)

    # --- Bandwidth ---
    ax = axes[1]
    ax.plot(x, _safe(mon.bw_A), label=f'BW → {mon.serverA.name}', marker='o', ms=3)
    ax.plot(x, _safe(mon.bw_B), label=f'BW → {mon.serverB.name}', marker='s', ms=3)
    ax.set_ylabel('Bandwidth (Mbits/s)')
    ax.set_title('Throughput UDP (iperf)')
    ax.legend(); ax.grid(True, alpha=.3)

    # --- Jitter + Packet loss ---
    ax = axes[2]
    ax2b = ax.twinx()
    ax.plot(x, _safe(mon.jitter_A), label=f'Jitter {mon.serverA.name}', color='C2', marker='o', ms=3)
    ax.plot(x, _safe(mon.jitter_B), label=f'Jitter {mon.serverB.name}', color='C3', marker='s', ms=3)
    ax2b.plot(x, _safe(mon.loss_A), '--', label=f'Loss {mon.serverA.name}', color='C4', ms=3)
    ax2b.plot(x, _safe(mon.loss_B), '--', label=f'Loss {mon.serverB.name}', color='C5', ms=3)
    ax.set_ylabel('Jitter (ms)')
    ax2b.set_ylabel('Packet Loss (%)')
    ax.set_title('Jitter & Packet Loss')
    lines_a, lab_a = ax.get_legend_handles_labels()
    lines_b, lab_b = ax2b.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, lab_a + lab_b, loc='upper left', fontsize=8)
    ax.grid(True, alpha=.3)

    # --- Flow rules ---
    ax = axes[3]
    ax.step(x, _safe(mon.flows), where='post', color='C6', label=f'{mon.victim.name} flows')
    ax.set_ylabel('Flow rules')
    ax.set_xlabel('Tempo (s)')
    ax.set_title(f'Flow Rules su {mon.victim.name}')
    ax.legend(); ax.grid(True, alpha=.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'network_monitor.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[monitor] Grafico salvato in: {out_path}')

    # --- Tabella riassuntiva a console ---
    def _fmt(lst):
        arr = _safe(lst)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return 'N/A', 'N/A', 'N/A'
        return f'{valid.mean():.2f}', f'{valid.min():.2f}', f'{valid.max():.2f}'

    print('\n=== Riepilogo metriche ===')
    hdr = f"{'Metrica':<30} {'Media':>10} {'Min':>10} {'Max':>10}"
    print(hdr)
    print('-' * len(hdr))
    for label, data in [
        (f'RTT → {mon.serverA.name} (ms)',       mon.rtt_A),
        (f'RTT → {mon.serverB.name} (ms)',       mon.rtt_B),
        (f'BW → {mon.serverA.name} (Mbps)',      mon.bw_A),
        (f'BW → {mon.serverB.name} (Mbps)',      mon.bw_B),
        (f'Jitter → {mon.serverA.name} (ms)',    mon.jitter_A),
        (f'Jitter → {mon.serverB.name} (ms)',    mon.jitter_B),
        (f'Loss → {mon.serverA.name} (%)',       mon.loss_A),
        (f'Loss → {mon.serverB.name} (%)',       mon.loss_B),
        (f'Flow rules ({mon.victim.name})',       mon.flows),
    ]:
        avg, mn, mx = _fmt(data)
        print(f'{label:<30} {avg:>10} {mn:>10} {mx:>10}')
    print()


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------
def start_monitor(client: Host, serverA: Host, serverB: Host,
                  victim: OVSSwitch, poll: int = POLL_INTERVAL) -> NetworkMonitor:
    """Avvia il monitoraggio in background e restituisce l'oggetto monitor."""
    mon = NetworkMonitor(client, serverA, serverB, victim, poll)
    mon.start()
    return mon


def stop_monitor_and_plot(mon: NetworkMonitor, out_dir: str = '/tmp'):
    """Ferma il monitor e genera i grafici. Chiamare dopo CLI(net)."""
    mon.stop()
    plot_results(mon, out_dir)
