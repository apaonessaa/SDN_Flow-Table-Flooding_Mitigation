from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_4
from ryu.lib.packet import packet, ethernet, ether_types, ipv4
from ryu.lib import hub

from eventlet import tpool

import os
import sys
import csv
import time
import threading
from collections import defaultdict, deque
from hashlib import sha256

import joblib
import numpy as np
import torch

# Modules loader
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
from mitigation.feature_ext import FeatureExt

# Architettura e padding delle finestre: unica sorgente, condivisa con il training.
from mitigation.tempbert_arch import TempBert, pad_window

# Ordine in cui FeatureExt restituisce le feature. L'ordine del modello
# (meta["features"]) puo' essere diverso: load_tempbert calcola la permutazione.
FTF_ORDER = [
    "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
    "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
    "c_used", "paf", "ft_psi", "durations_entropy",
    "pkts_entropy", "bytes_entropy", "mean_duration", "cvd",
    "mean_pkts", "cvp", "mean_bytes", "cvb",
]
MFR_ORDER = [
    "duration", "packet_count", "byte_count",
    "packets_per_seconds", "bytes_per_seconds", "psi",
]

# Una riga per ogni round di polling (anche quando la detection non parte).
# Per ogni fase (prep, infer, mitig): wall = tempo reale, cpu = tempo di calcolo
# effettivo; wall - cpu = attesa (scheduling del kernel o altri greenlet).
#   round_wall_s : durata totale del round
REPORT_COLUMNS = [
    "timestamp", "dpid", "round", "detection_run", "skipped",
    "flow_table_occupancy", "ftf_detected",
    "n_malicious_rules", "n_legit_rules", "n_fr_windows",
    "n_rules_wildcard",
    "n_eviction_msgs_wildcard", "n_eviction_msgs_strict", "n_eviction_msgs_total",
    "round_wall_s",
    "prep_wall_s", "prep_cpu_s",
    "infer_wall_s", "infer_cpu_s",
    "mitig_wall_s", "mitig_cpu_s",
]


def load_tempbert(model_dir, device, order):
    """model_dir/temp_bert/ contiene model.pt e meta.joblib (da train_tempBERT)."""
    d = os.path.join(model_dir, "temp_bert")
    meta = joblib.load(os.path.join(d, "meta.joblib"))
    params, features = meta["params"], meta["features"]

    model = TempBert(len(features), params["window"], params)
    model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=device))
    model.to(device).eval()

    return {
        "model": model, "scaler": meta["scaler"],
        "window": params["window"], "n_feat": len(features),
        "idx": [order.index(f) for f in features],   # ordine FeatureExt -> ordine modello
    }


class FTFMController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    FLOW_TABLE_SIZE = int(os.environ.get('FLOW_TABLE_SIZE', '50'))
    IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '10'))
    HARD_TIMEOUT = int(os.environ.get('HARD_TIMEOUT', '20'))
    K_FACTOR = int(os.environ.get('K_FACTOR', '2'))
    W_PKTIN = int(os.environ.get('W_PKTIN', '10'))
    MAX_COLLATERAL_PORT = int(os.environ.get('MAX_COLLATERAL_PORT', '0'))
    REPORT_PATH = os.environ.get('REPORT_PATH', '/tmp/report.txt')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    # Round di assenza dalle stats prima di eliminare la finestra MFR di una
    # regola. 0 = si elimina appena sparisce (la storia riparte da zero se
    # la stessa regola riappare).
    FR_PRUNE_GRACE = int(os.environ.get('FR_PRUNE_GRACE', '0'))

    FTF_MODEL_DIR = os.environ.get('FTF_MODEL_DIR', root + '/models/ftf_detector/2_100/')
    MFR_MODEL_DIR = os.environ.get('MFR_MODEL_DIR', root + '/models/mfr_detector/2/')

    FLOW_TABLE_THRESHOLD = 0.8

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.logger.setLevel(self.LOG_LEVEL)
        self.mac_to_port = {}
        self.datapaths = {}
        self.feature_ext = defaultdict(lambda: FeatureExt(window_size=self.W_PKTIN))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ftf = load_tempbert(self.FTF_MODEL_DIR, self.device, FTF_ORDER)
        self.mfr = load_tempbert(self.MFR_MODEL_DIR, self.device, MFR_ORDER)
        self.logger.info(f"TempBERT su {self.device}: FTF window={self.ftf['window']} "
                         f"n_feat={self.ftf['n_feat']}, MFR window={self.mfr['window']} n_feat={self.mfr['n_feat']}")

        # Finestre W_round: una per DPID (FT), una per (DPID, match_hash) (FR).
        # deque(maxlen=window): l'eviction del piu' vecchio e' automatica.
        self.w_round_per_ft = defaultdict(lambda: deque(maxlen=self.ftf["window"]))
        self.w_round_per_fr = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.mfr["window"])))
        self._round_idx = defaultdict(int)
        self._fr_last_seen = defaultdict(dict)

        # Inferenza in UN thread nativo: l'event loop di Ryu continua a servire
        # Packet-In ed echo; inferenze serializzate = misure non falsate.
        tpool.set_num_threads(1)
        self._detecting = set()   # DPID con una detection in corso

        self.T = float("{:.2f}".format(self.IDLE_TIMEOUT / self.K_FACTOR))
        self._init_report()
        self.collector = hub.spawn(self._collector)

    # ----------------------------------------------------------------- #
    # Report CSV
    # ----------------------------------------------------------------- #
    def _init_report(self):
        self._report_lock = threading.Lock()
        write_header = not os.path.exists(self.REPORT_PATH)
        self._report_file = open(self.REPORT_PATH, "a", newline="")
        self._report_writer = csv.writer(self._report_file)
        if write_header:
            self._report_writer.writerow(REPORT_COLUMNS)
            self._report_file.flush()

    def _write_report(self, **fields):
        row = [fields.get(c, 0) for c in REPORT_COLUMNS]
        with self._report_lock:
            self._report_writer.writerow(row)
            self._report_file.flush()

    def close(self):
        super(FTFMController, self).close()
        try:
            self._report_file.close()
        except Exception:
            pass

    def _collector(self):
        """Polling ogni T = idle_timeout / K (campionamento: T <= T_max / k, k >= 2)."""
        while True:
            for datapath in list(self.datapaths.values()):
                datapath.send_msg(datapath.ofproto_parser.OFPFlowStatsRequest(datapath))
            hub.sleep(self.T)

    # ----------------------------------------------------------------- #
    # Switch
    # ----------------------------------------------------------------- #
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Switch registrato: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if self.datapaths.pop(datapath.id, None) is not None:
                self.logger.info('Switch disconnesso: %016x', datapath.id)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst,
                                idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    def delete_flow(self, datapath, match, priority=1, strict=True):
        """OFPFC_DELETE_STRICT (match+priority) oppure OFPFC_DELETE (wildcard)."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        command = ofproto.OFPFC_DELETE_STRICT if strict else ofproto.OFPFC_DELETE
        mod = parser.OFPFlowMod(
            datapath=datapath, command=command,
            out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
            priority=priority, match=match)
        datapath.send_msg(mod)
        self.logger.info('[EVICTION] dpid=%016x priority=%s match=%s', datapath.id, priority, match)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst, src, dpid = eth.dst, eth.src, datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        ip = pkt.get_protocol(ipv4.ipv4) if eth.ethertype == ether_types.ETH_TYPE_IP else None

        if out_port != ofproto.OFPP_FLOOD:
            if ip is not None:
                match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP,
                                        eth_dst=dst, eth_src=src,
                                        ipv4_dst=ip.dst, ipv4_src=ip.src)
            else:
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions,
                          idle_timeout=self.IDLE_TIMEOUT, hard_timeout=self.HARD_TIMEOUT)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        datapath.send_msg(parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                              in_port=in_port, actions=actions, data=data))

        # Packet-In collector (solo pacchetti IPv4)
        if ip is None:
            return
        payload = pkt.protocols[-1]
        if not isinstance(payload, (bytes, bytearray)):
            payload = b""
        self.feature_ext[dpid].add_pkt_in(payload=payload, src_ip=ip.src, dst_ip=ip.dst,
                                          src_mac=src, dst_mac=dst)

    # ----------------------------------------------------------------- #
    # Flow stats -> round
    # ----------------------------------------------------------------- #
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        stats = [s for s in ev.msg.body if s.priority == 1]
        if not stats:
            # Tabella vuota (es. dopo un'eviction): le finestre delle regole sparite vanno potate.
            self._round_idx[dpid] += 1
            self._prune_fr_windows(dpid, self._round_idx[dpid])
            return
        hub.spawn(self._update_windows_and_maybe_detect, dpid, stats)

    def _prune_fr_windows(self, dpid, rnd):
        """Elimina le finestre FR delle regole non piu' presenti nelle stats."""
        seen, wins = self._fr_last_seen[dpid], self.w_round_per_fr[dpid]
        for h in [h for h, r in seen.items() if rnd - r > self.FR_PRUNE_GRACE]:
            seen.pop(h, None)
            wins.pop(h, None)

    def _update_windows_and_maybe_detect(self, dpid, stats):
        """La fase prep (feature + finestre + snapshot) non cede mai il controllo
        ad altri greenlet. Ogni fase e' misurata con wall e CPU dal StageTimer."""
        tm = StageTimer()
        ts0, t0 = time.time(), time.perf_counter()
        self._round_idx[dpid] += 1
        rnd = self._round_idx[dpid]
        fe = self.feature_ext[dpid]

        ftf_win = fr_snap = None
        with tm.stage("prep"):
            # Le finestre si aggiornano SEMPRE, anche sotto soglia di occupazione:
            # altrimenti il modello vedrebbe quasi solo padding quando serve.
            self.w_round_per_ft[dpid].append(self._extract_ftf_vector(fe, stats))

            mfr_rows = self._scale(self.mfr, [fe.get_flow_feature(s) for s in stats])  # 1 transform per round
            seen, hashes = self._fr_last_seen[dpid], []
            for stat, row in zip(stats, mfr_rows):
                h = sha256(str(stat.match).encode()).hexdigest()
                self.w_round_per_fr[dpid][h].append(row)
                seen[h] = rnd
                hashes.append(h)
            self._prune_fr_windows(dpid, rnd)

            occupancy = len(stats) / float(self.FLOW_TABLE_SIZE)
            ready = fe.has_enough_data() and occupancy >= self.FLOW_TABLE_THRESHOLD
            skipped = ready and dpid in self._detecting   # inferenza precedente ancora in corso
            detect = ready and not skipped

            # Snapshot prima di cedere il controllo: i round successivi modificano le deque.
            if detect:
                ftf_win = pad_window(self.w_round_per_ft[dpid], self.ftf["window"], self.ftf["n_feat"])
                fr_snap = [tuple(self.w_round_per_fr[dpid][h]) for h in hashes]

        res = None
        if detect:
            self._detecting.add(dpid)
            try:
                res = self._detect_and_mitigate(dpid, stats, ftf_win, fr_snap, tm)
            finally:
                self._detecting.discard(dpid)
        res = res or {}

        n_wc, n_strict = res.get("n_eviction_msgs_wildcard", 0), res.get("n_eviction_msgs_strict", 0)
        self._write_report(
            timestamp=round(ts0, 3), dpid=f"{dpid:016x}", round=rnd,
            detection_run=int(bool(res)), skipped=int(skipped),
            flow_table_occupancy=round(occupancy, 4),
            ftf_detected=res.get("ftf_detected", 0),
            n_malicious_rules=res.get("n_malicious_rules", 0),
            n_legit_rules=res.get("n_legit_rules", 0),
            n_fr_windows=len(self.w_round_per_fr[dpid]),
            n_rules_wildcard=res.get("n_rules_wildcard", 0),
            n_eviction_msgs_wildcard=n_wc, n_eviction_msgs_strict=n_strict,
            n_eviction_msgs_total=n_wc + n_strict,
            round_wall_s=round(time.perf_counter() - t0, 4),
            prep_wall_s=round(tm.wall["prep"], 4),   prep_cpu_s=round(tm.cpu["prep"], 4),
            infer_wall_s=round(tm.wall["infer"], 4), infer_cpu_s=round(tm.cpu["infer"], 4),
            mitig_wall_s=round(tm.wall["mitig"], 4), mitig_cpu_s=round(tm.cpu["mitig"], 4),
        )

    # ----------------------------------------------------------------- #
    # Feature -> matrice (n, F) nell'ordine del modello, scalata in un'unica transform
    # ----------------------------------------------------------------- #
    @staticmethod
    def _scale(model_info, rows):
        X = np.asarray(rows, dtype=np.float64)[:, model_info["idx"]]
        return model_info["scaler"].transform(X).astype(np.float32)

    def _extract_ftf_vector(self, fe, stats):
        ((pem, pec), pse, sie, die, sme, dme) = fe.get_pkt_in_features()
        (c_used, paf, ft_psi, de, pe, be,
         (md, cvd), (mp, cvp), (mb, cvb)) = fe.get_ft_features(stats, self.FLOW_TABLE_SIZE)
        raw = [pem, pec, pse, sie, die, sme, dme, c_used, paf, ft_psi,
               de, pe, be, md, cvd, mp, cvp, mb, cvb]          # = FTF_ORDER
        return self._scale(self.ftf, [raw])[0]

    # ----------------------------------------------------------------- #
    # Inferenza
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def _forward_many(self, model, windows, max_batch=512):
        """Gira in un thread nativo (tpool): tocca solo gli array ricevuti e il
        modello, mai le strutture condivise. Restituisce (predizioni, wall, cpu)."""
        t_wall, t_cpu = time.perf_counter(), time.thread_time()
        preds = []
        if windows:
            seqs = np.stack([s for s, _ in windows])        # (N, W, F)
            valids = np.stack([v for _, v in windows])      # (N, W)
            for i in range(0, len(windows), max_batch):
                logits = model(torch.from_numpy(seqs[i:i + max_batch]).to(self.device),
                               torch.from_numpy(valids[i:i + max_batch]).to(self.device))
                preds.extend(logits.argmax(-1).tolist())
        return preds, time.perf_counter() - t_wall, time.thread_time() - t_cpu

    def _infer(self, model, windows):
        """Sospende solo questo greenlet: l'event loop di Ryu resta libero."""
        return tpool.execute(self._forward_many, model, windows)

    # ----------------------------------------------------------------- #
    # Detection (FTF) + classificazione per-regola (MFR) + eviction
    # ----------------------------------------------------------------- #
    def _detect_and_mitigate(self, dpid, stats, ftf_win, fr_snap, tm):
        if self.datapaths.get(dpid) is None:
            return None  # switch disconnesso nel frattempo

        n_malicious = n_legit = n_rules_wildcard = n_wc_msgs = n_strict = 0

        # --- inferenza FTF. Wall: misurato qui (include l'attesa del thread nativo).
        #     CPU: misurata dentro il thread di inferenza e sommata a mano.
        with tm.stage("infer", cpu=False):
            (ftf_pred,), _, c = self._infer(self.ftf["model"], [ftf_win])
        tm.cpu["infer"] += c

        if ftf_pred:
            # --- mitigazione, parte 1: finestre per l'MFR
            with tm.stage("mitig"):
                windows = [pad_window(snap, self.mfr["window"], self.mfr["n_feat"])
                        for snap in fr_snap]

            # --- inferenza MFR
            with tm.stage("infer", cpu=False):
                mfr_preds, _, c = self._infer(self.mfr["model"], windows)
            tm.cpu["infer"] += c

            # --- mitigazione, parte 2: classificazione + invio delle delete
            with tm.stage("mitig"):
                malicious_stats = [s for s, p in zip(stats, mfr_preds) if p]
                legit_stats = [s for s, p in zip(stats, mfr_preds) if not p]
                n_malicious, n_legit = len(malicious_stats), len(legit_stats)
                self.logger.info(f"[!] FTF Attack Detected on DPID={dpid:016x}")
                datapath = self.datapaths.get(dpid)   # dopo l'attesa lo switch potrebbe essere sparito
                if datapath is not None and malicious_stats:
                    self.logger.info(f"[!] {n_malicious} regole malevole su DPID={dpid:016x}")
                    evicted, remaining, n_wc_msgs = self._evict_by_port(datapath, malicious_stats, legit_stats)
                    n_rules_wildcard = len(evicted)
                    for stat in remaining:
                        self.delete_flow(datapath, stat.match, priority=stat.priority, strict=True)
                        n_strict += 1

        return {
            "ftf_detected": int(bool(ftf_pred)),
            "n_malicious_rules": n_malicious, "n_legit_rules": n_legit,
            "n_rules_wildcard": n_rules_wildcard,
            "n_eviction_msgs_wildcard": n_wc_msgs,
            "n_eviction_msgs_strict": n_strict,
        }

    def _evict_by_port(self, datapath, malicious_stats, legit_stats):
        """Ritorna (regole coperte da wildcard, regole rimaste per il delete
        strict, numero di messaggi wildcard inviati)."""
        by_port = defaultdict(list)
        for stat in malicious_stats:
            by_port[stat.match['in_port']].append(stat)
        legit_by_port = defaultdict(int)
        for stat in legit_stats:
            legit_by_port[stat.match['in_port']] += 1

        evicted, remaining, n_wc_msgs = [], [], 0
        for in_port, group in by_port.items():
            collateral = legit_by_port.get(in_port, 0)
            # Una sola regola malevola: il delete strict e' gia' ottimo (1 msg, 0 collateral).
            if len(group) < 2 or collateral > self.MAX_COLLATERAL_PORT:
                remaining.extend(group)
                continue
            parser = datapath.ofproto_parser
            self.delete_flow(datapath, parser.OFPMatch(in_port=in_port), priority=1, strict=False)
            n_wc_msgs += 1
            self.logger.info('[EVICTION-L0] Dpid=%016x discarded in_port=%s covered=%d collateral=%d',
                             datapath.id, in_port, len(group), collateral)
            evicted.extend(group)
        return evicted, remaining, n_wc_msgs
