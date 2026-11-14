"""
Monitor di CPU/RAM per i processi della simulazione (controller Ryu, OVS).

Va salvato come utils/resource_monitor.py. Produce due CSV:
  - <out>.csv          : un campione ogni `interval` secondi per processo
  - <out>_events.csv   : i marker di fase (idle_start, attack_start, ...)
Entrambi usano time.time(), lo stesso orologio del report del controller,
quindi si allineano senza conversioni.
"""
import csv
import threading
import time
import os

import psutil


def find_ryu(popen):
    """
    Ritorna il processo ryu-manager vero. Se il controller e' stato lanciato
    tramite shell o sudo, popen.pid e' quello del wrapper: si scende
    nell'albero dei figli e si prende il piu' profondo che contiene
    'ryu-manager' nella command line.
    """
    root = psutil.Process(popen.pid)
    match = None
    for p in [root] + root.children(recursive=True):
        try:
            if any("ryu-manager" in a or "ryu.cmd" in a for a in p.cmdline()):
                match = p
        except psutil.NoSuchProcess:
            pass
    return match or root


def find_by_name(name):
    for p in psutil.process_iter(["name"]):
        if p.info["name"] == name:
            return p
    return None


def pin_process(proc, cpus):
    """
    Fissa TUTTI i thread esistenti di `proc` sui core indicati. Su Linux
    sched_setaffinity(pid) agisce solo sul thread principale, per questo si
    itera sui TID; i thread creati dopo ereditano la maschera.
    """
    for t in proc.threads():
        try:
            os.sched_setaffinity(t.id, cpus)
        except OSError:
            pass


class ResourceMonitor(threading.Thread):
    def __init__(self, procs, out_path, interval=0.5):
        super().__init__(daemon=True)
        # {"ryu": Process, "ovs": Process}; i processi None vengono ignorati
        self.procs = {n: p for n, p in procs.items() if p is not None}
        self.out_path = out_path
        self.interval = interval
        # Attenzione: NON chiamare questo attributo _stop, e' un metodo
        # interno di threading.Thread e sovrascriverlo rompe join().
        self._halt = threading.Event()

    def run(self):
        for p in self.procs.values():
            p.cpu_percent(None)        # il primo campione e' sempre 0: inizializzazione
        psutil.cpu_percent(None)

        with open(self.out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "name", "cpu_pct", "rss_mb", "uss_mb", "threads"])
            nxt = time.time()
            while not self._halt.is_set():
                ts = time.time()
                for name, p in self.procs.items():
                    try:
                        m = p.memory_full_info()
                        w.writerow([ts, name, p.cpu_percent(None),
                                    round(m.rss / 2**20, 2), round(m.uss / 2**20, 2),
                                    p.num_threads()])
                    except psutil.NoSuchProcess:
                        pass
                # CPU totale della VM e RAM usata, per accorgersi di contese
                w.writerow([ts, "system", psutil.cpu_percent(None),
                            round(psutil.virtual_memory().used / 2**20, 2), "", ""])
                f.flush()
                nxt += self.interval
                self._halt.wait(max(0.0, nxt - time.time()))

    def stop(self):
        """Idempotente: si puo' chiamare sia a fine traffico sia nel finally."""
        self._halt.set()
        if self.is_alive():
            self.join()
