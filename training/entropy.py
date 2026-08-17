"""
Strutture dati per calcolo incrementale di entropia e mean/CV su una
finestra scorrevole a dimensione fissa (sliding window), pensate per
essere usate al posto delle LifoQueue in FeatureExt.

Idea di base:
- H = -sum(p_i * log2(p_i)) con p_i = c_i / n
    = log2(n) - (1/n) * sum(c_i * log2(c_i))

  Chiamiamo S = sum(c_i * log2(c_i)). Quando un solo elemento entra o
  esce dalla finestra, cambia il count di UN solo simbolo: possiamo
  aggiornare S in O(1) sottraendo il vecchio termine e aggiungendo il
  nuovo, invece di ricalcolare la somma su tutti i simboli.

- Per mean e CV manteniamo somma e somma dei quadrati della finestra,
  anch'esse aggiornabili in O(1) quando un valore entra/esce.

Entrambe le strutture fanno un resync periodico (ricalcolo completo)
per contrastare la deriva numerica dovuta ai float, senza perdere il
vantaggio asintotico (il resync costa O(window_size) ma avviene ogni
N operazioni, non ad ogni pacchetto).
"""

import math
from collections import deque, Counter


class IncrementalEntropy:
    """Entropia di Shannon su una finestra scorrevole di valori categorici
    (es. IP sorgente, IP destinazione, MAC, dimensione payload)."""

    def __init__(self, maxlen, resync_every=10_000):
        self.window = deque(maxlen=maxlen)
        self.counts = Counter()
        self.S = 0.0  # sum(c * log2(c)) sui simboli correnti
        self._resync_every = resync_every
        self._ops_since_resync = 0

    @staticmethod
    def _term(c):
        return c * math.log2(c) if c > 0 else 0.0

    def add(self, value):
        # Se la finestra è piena, il prossimo append farà uscire
        # automaticamente l'elemento più vecchio (deque con maxlen).
        if len(self.window) == self.window.maxlen and self.window.maxlen:
            evicted = self.window[0]
            c = self.counts[evicted]
            self.S -= self._term(c)
            c -= 1
            if c == 0:
                del self.counts[evicted]
            else:
                self.counts[evicted] = c
                self.S += self._term(c)

        c = self.counts.get(value, 0)
        self.S -= self._term(c)
        c += 1
        self.counts[value] = c
        self.S += self._term(c)

        self.window.append(value)

        self._ops_since_resync += 1
        if self._ops_since_resync >= self._resync_every:
            self._resync()

    def _resync(self):
        self.counts = Counter(self.window)
        self.S = sum(self._term(c) for c in self.counts.values())
        self._ops_since_resync = 0

    def entropy(self):
        n = len(self.window)
        if n == 0:
            return 0.0
        return math.log2(n) - self.S / n


class IncrementalMeanCV:
    """Media e coefficiente di variazione su una finestra scorrevole di
    valori numerici (es. entropia di ogni singolo payload)."""

    def __init__(self, maxlen, resync_every=10_000):
        self.window = deque(maxlen=maxlen)
        self.sum = 0.0
        self.sumsq = 0.0
        self._resync_every = resync_every
        self._ops_since_resync = 0

    def add(self, value):
        if len(self.window) == self.window.maxlen and self.window.maxlen:
            evicted = self.window[0]
            self.sum -= evicted
            self.sumsq -= evicted * evicted

        self.window.append(value)
        self.sum += value
        self.sumsq += value * value

        self._ops_since_resync += 1
        if self._ops_since_resync >= self._resync_every:
            self._resync()

    def _resync(self):
        self.sum = sum(self.window)
        self.sumsq = sum(v * v for v in self.window)
        self._ops_since_resync = 0

    def mean_cv(self):
        n = len(self.window)
        if n == 0:
            return 0.0, 0.0
        mean = self.sum / n
        var = max(self.sumsq / n - mean * mean, 0.0)  # guard su errore float
        std = math.sqrt(var)
        cv = std / mean if mean != 0 else 0.0
        return mean, cv


class FeatureExt:
    """Versione di FeatureExt con calcolo incrementale delle feature
    di finestra: add_pkt_in è O(1) invece di O(WINDOW_SIZE), e
    get_pkt_in_features legge lo stato già aggregato invece di
    ricalcolare tutto ad ogni chiamata.
    """

    VALID_IP_RANGES = ['10.0.0.0/24']
    WINDOW_SIZE = 500
    FLOW_TABLE_CAPACITY = 50

    def __init__(self):
        # entropia sul valore di ogni singolo payload (per mean/CV)
        self.payload_entropy_stats = IncrementalMeanCV(self.WINDOW_SIZE)

        # entropie categoriche sulla finestra
        self.payload_size_entropy = IncrementalEntropy(self.WINDOW_SIZE)
        self.src_ip_entropy = IncrementalEntropy(self.WINDOW_SIZE)
        self.dst_ip_entropy = IncrementalEntropy(self.WINDOW_SIZE)
        self.src_mac_entropy = IncrementalEntropy(self.WINDOW_SIZE)
        self.dst_mac_entropy = IncrementalEntropy(self.WINDOW_SIZE)

    def add_pkt_in(self, payload, src_ip, dst_ip, src_mac, dst_mac):
        self.payload_entropy_stats.add(FeatureExt.shannon_entropy(payload))
        self.payload_size_entropy.add(len(payload))
        self.src_ip_entropy.add(src_ip)
        self.dst_ip_entropy.add(dst_ip)
        self.src_mac_entropy.add(src_mac)
        self.dst_mac_entropy.add(dst_mac)

    def get_pkt_in_features(self):
        return (
            self.payload_entropy_stats.mean_cv(),
            self.payload_size_entropy.entropy(),
            self.src_ip_entropy.entropy(),
            self.dst_ip_entropy.entropy(),
            self.src_mac_entropy.entropy(),
            self.dst_mac_entropy.entropy(),
        )

    # --- il resto della classe (get_ft_features, get_flow_feature,
    # calculate_paf, calculate_ft_psi, calculate_psi, effective_duration,
    # shannon_entropy statico per singolo payload) resta invariato,
    # quelle operano su flow correnti e non sulla finestra scorrevole. ---

    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        n = len(data)
        if n == 0:
            return 0.0
        counts = Counter(data)
        pbs = [nb / n for nb in counts.values()]
        return -sum(pb * math.log2(pb) for pb in pbs)


if __name__ == "__main__":
    # micro-verifica di correttezza contro il calcolo diretto
    import random

    fe = FeatureExt()
    ref_window = deque(maxlen=FeatureExt.WINDOW_SIZE)

    def ref_entropy(values):
        n = len(values)
        if n == 0:
            return 0.0
        counts = Counter(values)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    random.seed(0)
    for i in range(2000):
        src_ip = f"10.0.0.{random.randint(1, 20)}"
        fe.src_ip_entropy.add(src_ip)
        ref_window.append(src_ip)

        if i % 137 == 0:  # controllo a campione
            incr = fe.src_ip_entropy.entropy()
            ref = ref_entropy(list(ref_window))
            assert abs(incr - ref) < 1e-9, (i, incr, ref)

    print("OK: entropia incrementale coincide con quella ricalcolata da zero")
