"""
network_monitor.py
==================
Monitor per reti Mininet: misura la capienza delle Flow Table (OVS)
e le performance di rete (latenza, banda, jitter, perdita pacchetti).
Al termine genera un file CSV con tutte le statistiche raccolte.

Utilizzo
--------
    from network_monitor import NetworkMonitor

    monitor = NetworkMonitor(
        net,
        hosts=[net.get('h1'), net.get('h2')],
        switches=[net.get('s1'), net.get('s2')],
        interval=10,          # secondi tra un campionamento e l'altro
        csv_path='stats.csv', # path del file CSV di output
        ping_count=5,
        iperf_bandwidth='10M',
    )
    monitor.start()
    # ... la tua logica di rete ...
    monitor.stop()           # blocca il monitor e scrive il CSV
"""

import csv
import re
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Deque

from mininet.net import Mininet
from mininet.node import Host, OVSSwitch

# ---------------------------------------------------------------------------
# Costanti di default
# ---------------------------------------------------------------------------
PING_COUNT      = 5
IPERF_DURATION  = 300
IPERF_INTERVAL  = 1

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------

@dataclass
class FlowTableSample:
    """Singolo campione della flow table di uno switch."""
    timestamp:   str
    switch_name: str
    flow_count:  int
    # Statistiche aggiuntive estratte da ovs-ofctl
    table_id:    Optional[int]  = None
    active_count: Optional[int] = None
    lookup_count: Optional[int] = None
    matched_count: Optional[int] = None


@dataclass
class PingSample:
    """Latenza RTT tra due host (ping)."""
    timestamp:  str
    src:        str
    dst:        str
    rtt_avg_ms: Optional[float]
    packet_loss_pct: Optional[float] = None


@dataclass
class IperfSample:
    """Metrica iperf UDP per un intervallo di tempo."""
    timestamp:   str
    src:         str
    dst:         str
    interval_s:  float
    bandwidth_mbps: Optional[float]
    jitter_ms:      Optional[float]
    loss_pct:       Optional[float]


# ---------------------------------------------------------------------------
# Funzioni di misura (riutilizzabili autonomamente)
# ---------------------------------------------------------------------------

def h_stats(src: Host, dst: Host, nping: int = PING_COUNT) -> tuple:
    """
    Esegue ping da src a dst e restituisce (rtt_avg_ms, loss_pct).
    Restituisce (None, None) in caso di errore.
    """
    raw = src.cmd(f'ping -c {nping} -q {dst.IP()}')

    rtt_avg = None
    m = re.search(r'rtt min/avg/max/mdev = [\d.]+/([\d.]+)/', raw)
    if m:
        rtt_avg = float(m.group(1))

    loss = None
    l = re.search(r'(\d+)%\s+packet loss', raw)
    if l:
        loss = float(l.group(1))

    return rtt_avg, loss


def s_stats(sw: OVSSwitch) -> int:
    """Conta il numero di flow attivi sullo switch (OpenFlow 1.3)."""
    raw = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-flows {sw.name}')
    return sum(1 for line in raw.splitlines() if 'cookie=' in line)


def s_stats_detailed(sw: OVSSwitch) -> List[Dict]:
    """
    Restituisce statistiche dettagliate per ogni tabella dello switch.
    Legge l'output di `ovs-ofctl dump-tables`.
    """
    raw = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-tables {sw.name}')
    results = []
    for line in raw.splitlines():
        # Riga tipo: "  0: classifier: wild=0x3ff, max=1000000, active=3"
        m = re.match(
            r'\s*(\d+):.*?active=(\d+)(?:.*?lookup=(\d+))?(?:.*?matched=(\d+))?',
            line
        )
        if m:
            results.append({
                'table_id':     int(m.group(1)),
                'active_count': int(m.group(2)),
                'lookup_count': int(m.group(3)) if m.group(3) else None,
                'matched_count': int(m.group(4)) if m.group(4) else None,
            })
    return results


def _parse_iperf_line(line: str):
    """
    Analizza una riga dell'output iperf UDP e restituisce
    (bandwidth_mbps, jitter_ms, loss_pct).
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


class _IperfStream:
    """
    Avvia iperf client+server interamente in memoria, senza file su disco.

    Strategia
    ---------
    - Il server viene avviato tramite ``Host.popen()``, che restituisce un
      oggetto ``subprocess.Popen`` collegato alle pipe stdin/stdout/stderr
      del namespace di rete dell'host.
    - Un thread di lettura consuma stdout riga per riga e memorizza i
      campioni analizzati in un ``deque`` thread-safe.
    - Il client viene avviato allo stesso modo e il suo stdout viene
      scartato (non ci serve: tutte le statistiche UDP compaiono sul server).
    - ``stop()`` termina entrambi i processi e il thread di lettura.
    """

    def __init__(
        self,
        client: Host,
        server: Host,
        bandwidth: str = '10M',
        duration: int = IPERF_DURATION,
        interval: int = IPERF_INTERVAL,
    ):
        self.src_name  = client.name
        self.dst_name  = server.name
        self._client   = client
        self._server   = server
        self._bandwidth = bandwidth
        self._duration  = duration
        self._interval  = interval

        self.samples: Deque[IperfSample] = deque()

        self._srv_proc  = None   # Popen del server
        self._cli_proc  = None   # Popen del client
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_evt  = threading.Event()

    # ------------------------------------------------------------------

    def start(self):
        """Avvia server, client e thread di lettura."""
        # Termina eventuali istanze precedenti
        self._client.cmd('pkill -f "iperf -c" 2>/dev/null; true')
        self._server.cmd('pkill -f "iperf -s" 2>/dev/null; true')
        time.sleep(0.2)

        # Server: stdout catturato dalla pipe, nessun file su disco
        self._srv_proc = self._server.popen(
            f'iperf -s -u -i {self._interval}',
            shell=True,
            stdout=-1,   # subprocess.PIPE
            stderr=-1,
        )
        time.sleep(0.3)

        # Client: stdout scartato (/dev/null); le statistiche server bastano
        self._cli_proc = self._client.popen(
            f'iperf -c {self._server.IP()} -u -b {self._bandwidth} '
            f'-t {self._duration} -i {self._interval}',
            shell=True,
            stdout=-1,
            stderr=-1,
        )

        self._stop_evt.clear()
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f'iperf-reader-{self.src_name}-{self.dst_name}',
        )
        self._reader_thread.start()
        log.debug('[IperfStream] avviato %s->%s', self.src_name, self.dst_name)

    def stop(self):
        """Termina i processi e il thread di lettura."""
        self._stop_evt.set()

        for proc in (self._cli_proc, self._srv_proc):
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if self._reader_thread:
            self._reader_thread.join(timeout=5)

        log.debug('[IperfStream] fermato %s->%s (%d campioni)',
                  self.src_name, self.dst_name, len(self.samples))

    def drain(self) -> List[IperfSample]:
        """Svuota e restituisce tutti i campioni accumulati."""
        result = []
        while self.samples:
            result.append(self.samples.popleft())
        return result

    # ------------------------------------------------------------------

    def _read_loop(self):
        """Thread: legge stdout del server riga per riga."""
        try:
            for raw_line in self._srv_proc.stdout:
                if self._stop_evt.is_set():
                    break
                line = raw_line.decode('utf-8', errors='replace').rstrip()
                if 'Mbits/sec' not in line:
                    continue

                interval_s = 0.0
                iv_m = re.search(r'([\d.]+)-([\d.]+)\s+sec', line)
                if iv_m:
                    interval_s = float(iv_m.group(2)) - float(iv_m.group(1))

                bw, jitter, loss = _parse_iperf_line(line)
                if bw is not None:
                    self.samples.append(IperfSample(
                        timestamp=datetime.now().isoformat(timespec='seconds'),
                        src=self.src_name,
                        dst=self.dst_name,
                        interval_s=interval_s,
                        bandwidth_mbps=bw,
                        jitter_ms=jitter,
                        loss_pct=loss,
                    ))
        except Exception as exc:
            if not self._stop_evt.is_set():
                log.warning('[IperfStream] Errore lettura %s->%s: %s',
                            self.src_name, self.dst_name, exc)


# ---------------------------------------------------------------------------
# Monitor principale
# ---------------------------------------------------------------------------

class NetworkMonitor:
    """
    Monitor periodico per una rete Mininet.

    Parametri
    ---------
    net              : istanza Mininet (opzionale, usato solo per il cleanup)
    hosts            : coppie (src, dst) OPPURE lista di Host (campionate a coppie consecutive)
    switches         : lista di OVSSwitch da monitorare
    interval         : secondi tra un campionamento e il successivo
    csv_path         : percorso del file CSV generato allo stop()
    ping_count       : numero di ping per misura RTT
    iperf_bandwidth  : banda target per iperf (es. '10M')
    enable_iperf     : attiva/disattiva la raccolta iperf
    enable_ping      : attiva/disattiva la raccolta ping
    enable_flow_table: attiva/disattiva la raccolta flow table
    """

    def __init__(
        self,
        net: Optional[Mininet],
        hosts: List[Host],
        switches: List[OVSSwitch],
        interval: float = 10.0,
        csv_path: str = 'mininet_stats.csv',
        ping_count: int = PING_COUNT,
        iperf_bandwidth: str = '10M',
        enable_iperf: bool = True,
        enable_ping: bool = True,
        enable_flow_table: bool = True,
    ):
        self.net              = net
        self.interval         = interval
        self.csv_path         = csv_path
        self.ping_count       = ping_count
        self.iperf_bandwidth  = iperf_bandwidth
        self.enable_iperf     = enable_iperf
        self.enable_ping      = enable_ping
        self.enable_flow_table = enable_flow_table

        # Normalizza la lista host in coppie (src, dst)
        self.host_pairs: List[tuple] = self._build_pairs(hosts)
        self.switches = switches

        # Buffer dei campioni
        self._flow_samples:  List[FlowTableSample] = []
        self._ping_samples:  List[PingSample]      = []
        self._iperf_samples: List[IperfSample]     = []

        # Stream iperf in memoria per ogni coppia
        self._iperf_streams: Dict[str, _IperfStream] = {}

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def start(self):
        """Avvia il monitor in un thread separato."""
        log.info('[NetworkMonitor] Avvio monitor...')
        if self.enable_iperf:
            self._start_iperf_streams()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='NetMonitor')
        self._thread.start()
        log.info('[NetworkMonitor] Thread avviato (interval=%.1fs)', self.interval)

    def stop(self):
        """
        Ferma il monitor, raccoglie i dati iperf residui
        e scrive il file CSV.
        """
        log.info('[NetworkMonitor] Stop richiesto...')
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        if self.enable_iperf:
            self._collect_iperf()
        self._write_csv()
        log.info('[NetworkMonitor] CSV salvato in: %s', self.csv_path)

    def snapshot(self):
        """Forza un campionamento immediato (utile per test manuali)."""
        self._collect_all()

    # ------------------------------------------------------------------
    # Implementazione interna
    # ------------------------------------------------------------------

    def _build_pairs(self, hosts: List[Host]) -> List[tuple]:
        """Costruisce coppie (src, dst) dalla lista di host."""
        if not hosts:
            return []
        # Se sono già tuple/liste di 2 elementi
        if isinstance(hosts[0], (list, tuple)):
            return [(h[0], h[1]) for h in hosts]
        # Altrimenti coppie consecutive
        pairs = []
        for i in range(0, len(hosts) - 1, 2):
            pairs.append((hosts[i], hosts[i + 1]))
        # Se c'è un host dispari, abbinalo al primo
        if len(hosts) % 2 == 1:
            pairs.append((hosts[-1], hosts[0]))
        return pairs

    def _start_iperf_streams(self):
        """Avvia gli stream iperf in memoria per tutte le coppie."""
        for src, dst in self.host_pairs:
            key = f'{src.name}->{dst.name}'
            stream = _IperfStream(
                client=src,
                server=dst,
                bandwidth=self.iperf_bandwidth,
            )
            stream.start()
            self._iperf_streams[key] = stream
            log.debug('[NetworkMonitor] IperfStream avviato: %s', key)

    def _collect_iperf(self):
        """Ferma gli stream iperf e raccoglie tutti i campioni accumulati."""
        for key, stream in self._iperf_streams.items():
            stream.stop()
            samples = stream.drain()
            self._iperf_samples.extend(samples)
            log.debug('[NetworkMonitor] iperf %s: %d campioni raccolti', key, len(samples))

    def _collect_flow_tables(self):
        ts = datetime.now().isoformat(timespec='seconds')
        for sw in self.switches:
            flow_count = s_stats(sw)
            details    = s_stats_detailed(sw)

            if details:
                for d in details:
                    self._flow_samples.append(FlowTableSample(
                        timestamp=ts,
                        switch_name=sw.name,
                        flow_count=flow_count,
                        table_id=d['table_id'],
                        active_count=d['active_count'],
                        lookup_count=d['lookup_count'],
                        matched_count=d['matched_count'],
                    ))
            else:
                self._flow_samples.append(FlowTableSample(
                    timestamp=ts,
                    switch_name=sw.name,
                    flow_count=flow_count,
                ))

    def _collect_ping(self):
        ts = datetime.now().isoformat(timespec='seconds')
        for src, dst in self.host_pairs:
            rtt_avg, loss = h_stats(src, dst, self.ping_count)
            self._ping_samples.append(PingSample(
                timestamp=ts,
                src=src.name,
                dst=dst.name,
                rtt_avg_ms=rtt_avg,
                packet_loss_pct=loss,
            ))
            log.debug('[NetworkMonitor] ping %s->%s: rtt=%.2f ms loss=%.0f%%',
                      src.name, dst.name,
                      rtt_avg if rtt_avg else -1,
                      loss if loss else 0)

    def _collect_all(self):
        if self.enable_flow_table:
            self._collect_flow_tables()
        if self.enable_ping:
            self._collect_ping()

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._collect_all()
            except Exception as exc:
                log.warning('[NetworkMonitor] Errore durante il campionamento: %s', exc)
            self._stop_event.wait(self.interval)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _write_csv(self):
        """Scrive tutti i campioni in un CSV multi-sezione."""
        rows = []

        # --- Flow Table ---
        for s in self._flow_samples:
            rows.append({
                'type':           'flow_table',
                'timestamp':      s.timestamp,
                'node':           s.switch_name,
                'metric':         'flow_count',
                'value':          s.flow_count,
                'table_id':       s.table_id if s.table_id is not None else '',
                'active_count':   s.active_count if s.active_count is not None else '',
                'lookup_count':   s.lookup_count if s.lookup_count is not None else '',
                'matched_count':  s.matched_count if s.matched_count is not None else '',
                'src':            '',
                'dst':            '',
                'rtt_avg_ms':     '',
                'packet_loss_pct': '',
                'interval_s':    '',
                'bandwidth_mbps': '',
                'jitter_ms':     '',
                'loss_pct':      '',
            })

        # --- Ping ---
        for s in self._ping_samples:
            rows.append({
                'type':           'ping',
                'timestamp':      s.timestamp,
                'node':           '',
                'metric':         'rtt_avg',
                'value':          s.rtt_avg_ms if s.rtt_avg_ms is not None else '',
                'table_id':       '',
                'active_count':   '',
                'lookup_count':   '',
                'matched_count':  '',
                'src':            s.src,
                'dst':            s.dst,
                'rtt_avg_ms':     s.rtt_avg_ms if s.rtt_avg_ms is not None else '',
                'packet_loss_pct': s.packet_loss_pct if s.packet_loss_pct is not None else '',
                'interval_s':    '',
                'bandwidth_mbps': '',
                'jitter_ms':     '',
                'loss_pct':      '',
            })

        # --- iPerf ---
        for s in self._iperf_samples:
            rows.append({
                'type':           'iperf',
                'timestamp':      s.timestamp,
                'node':           '',
                'metric':         'bandwidth',
                'value':          s.bandwidth_mbps if s.bandwidth_mbps is not None else '',
                'table_id':       '',
                'active_count':   '',
                'lookup_count':   '',
                'matched_count':  '',
                'src':            s.src,
                'dst':            s.dst,
                'rtt_avg_ms':     '',
                'packet_loss_pct': '',
                'interval_s':    s.interval_s,
                'bandwidth_mbps': s.bandwidth_mbps if s.bandwidth_mbps is not None else '',
                'jitter_ms':     s.jitter_ms if s.jitter_ms is not None else '',
                'loss_pct':      s.loss_pct if s.loss_pct is not None else '',
            })

        if not rows:
            log.warning('[NetworkMonitor] Nessun dato da scrivere nel CSV.')
            return

        fieldnames = list(rows[0].keys())
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        log.info('[NetworkMonitor] CSV scritto: %d righe in %s', len(rows), self.csv_path)

    # ------------------------------------------------------------------
    # Proprietà di sola lettura per ispezione
    # ------------------------------------------------------------------

    @property
    def flow_samples(self) -> List[FlowTableSample]:
        return list(self._flow_samples)

    @property
    def ping_samples(self) -> List[PingSample]:
        return list(self._ping_samples)

    @property
    def iperf_samples(self) -> List[IperfSample]:
        return list(self._iperf_samples)

    def summary(self) -> str:
        """Restituisce un breve riepilogo testuale dei dati raccolti."""
        lines = [
            '=== NetworkMonitor Summary ===',
            f'  Flow table samples : {len(self._flow_samples)}',
            f'  Ping samples       : {len(self._ping_samples)}',
            f'  iPerf samples      : {len(self._iperf_samples)}',
        ]
        if self._ping_samples:
            valid = [s.rtt_avg_ms for s in self._ping_samples if s.rtt_avg_ms]
            if valid:
                lines.append(f'  RTT avg (overall)  : {sum(valid)/len(valid):.2f} ms')
        if self._iperf_samples:
            valid_bw = [s.bandwidth_mbps for s in self._iperf_samples if s.bandwidth_mbps]
            if valid_bw:
                lines.append(f'  BW avg (overall)   : {sum(valid_bw)/len(valid_bw):.2f} Mbps')
        return '\n'.join(lines)
