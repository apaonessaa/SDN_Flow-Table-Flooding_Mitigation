from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_4
from ryu.lib.packet import packet, ethernet, ether_types, ipv4
from ryu.lib import hub

import os
import re
import sys
import csv
import time
import threading
from collections import defaultdict, deque
from hashlib import md5
from eventlet import tpool
from contextlib import contextmanager

import joblib
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Modules loader
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
from mitigation.feature_ext import FeatureExt

from mitigation.tempbert_arch import TempBert, pad_window

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

REPORT_COLUMNS = [
    "timestamp", "dpid", "round", "detection_run", "skipped",
    "flow_table_occupancy", "ftf_detected",
    "ftf_run", "ftf_pred", "mfr_run",
    "n_malicious_rules", "n_malicious_real", "n_legit_rules", "n_fr_windows",
    "n_rules_wildcard",
    "n_eviction_msgs_wildcard", "n_eviction_msgs_strict", "n_eviction_msgs_total",
    "round_wall_s",
    "prep_wall_s", "prep_cpu_s",
    "infer_wall_s", "infer_cpu_s",
    "mitig_wall_s", "mitig_cpu_s",
    "n_blocked_pktin", "n_blocked_pairs",
    "n_pktin_total", "n_pktin_real", "n_blocked_pktin_real",
    "n_real_pairs_seen", "n_real_pairs_blocked_cum", "n_blocked_pairs_real",
    "n_pktin_atk", "n_blocked_pktin_atk"
]

_MAC_RE = re.compile(r'^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$')


def parse_macs(raw):
    """Accetta separatori , ; spazi e MAC con ':' o '-'. Ritorna (set valido, token scartati)."""
    macs, bad = set(), []
    for tok in re.split(r'[,;\s]+', raw.strip().lower()):
        if not tok:
            continue
        tok = tok.replace('-', ':')
        if _MAC_RE.match(tok):
            macs.add(tok)
        else:
            bad.append(tok)
    return macs, bad


class StageTimer:
    def __init__(self):
        self.wall = defaultdict(float)
        self.cpu = defaultdict(float)
    @contextmanager
    def stage(self, name, cpu=True):
        w, c = time.perf_counter(), time.thread_time()
        try:
            yield
        finally:
            self.wall[name] += time.perf_counter() - w
            if cpu:
                self.cpu[name] += time.thread_time() - c

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
    FLOW_TABLE_THRESHOLD = float(os.environ.get('FLOW_TABLE_THRESHOLD', '0.8'))
    REAL_MACS, _REAL_MACS_BAD = parse_macs(os.environ.get('REAL_MACS', ''))

    FTF_MODEL_DIR = os.environ.get('FTF_MODEL_DIR', root + '/models/ftf_detector/2_100/')
    MFR_MODEL_DIR = os.environ.get('MFR_MODEL_DIR', root + '/models/mfr_detector/2/')

    FR_PRUNE_GRACE  = int(os.environ.get('FR_PRUNE_GRACE', '3'))
    BLK_TTL         = float(os.environ.get('BLK_TTL', '30'))
    BLK_MIN_ROUNDS  = int(os.environ.get('BLK_MIN_ROUNDS', '2'))
    STRIKE_TTL      = float(os.environ.get('STRIKE_TTL', '30'))

    FTF_TTL         = float(os.environ.get('FTF_TTL', '30'))

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.logger.setLevel(self.LOG_LEVEL)
        if self._REAL_MACS_BAD:
            self.logger.warning('REAL_MACS: token non validi ignorati: %s', self._REAL_MACS_BAD)
        if self.REAL_MACS:
            self.logger.info('REAL_MACS: %d MAC caricati', len(self.REAL_MACS))
        else:
            self.logger.warning('REAL_MACS vuoto: le metriche benigno/malevolo del blocker non saranno affidabili')

        self.mac_to_port = {}
        self.datapaths = {}

        # Lease FTF per dpid: istante di scadenza (None = nessun attacco dichiarato).
        # Finche' e' attiva gira solo l'MFR, poi riprende la detection FTF.
        self._ftf_until = {}

        # Feature Extractor per dpid
        self.feature_ext = defaultdict(lambda: FeatureExt(window_size=self.W_PKTIN))

        # TempBERT models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ftf = load_tempbert(self.FTF_MODEL_DIR, self.device, FTF_ORDER)
        self.mfr = load_tempbert(self.MFR_MODEL_DIR, self.device, MFR_ORDER)
        self.logger.info(f"TempBERT su {self.device}: FTF window={self.ftf['window']} "
                         f"n_feat={self.ftf['n_feat']}, MFR window={self.mfr['window']} n_feat={self.mfr['n_feat']}")

        # W_pktin per FTF and MFR
        self.w_round_per_ft = defaultdict(lambda: deque(maxlen=self.ftf["window"]))
        self.w_round_per_fr = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.mfr["window"])))
        self._round_idx = defaultdict(int)
        self._fr_last_seen = defaultdict(dict)
        tpool.set_num_threads(1)
        self._detecting = set()                 # inferenza/mitigazione in corso sul dpid

        # Packet-In Blocker
        self.blocked = {}                       # (dpid, in_port, eth_src, eth_dst) -> lease time
        self.strikes = {}                       # (dpid, in_port, eth_src, eth_dst) -> (n_segnalazioni, ts ultima)
        self.n_blocked = defaultdict(int)       # Packet-In senza flow-mod per dpid
        self.n_pktin_total = defaultdict(int)
        self.n_pktin_real = defaultdict(int)
        self.n_blocked_real = defaultdict(int)
        self.n_pktin_atk = defaultdict(int)
        self.n_blocked_pktin_atk = defaultdict(int)

        self.real_seen = defaultdict(set)           # coppie MAC reali visti in Packet-In, per dpid
        self.real_blocked_ever = defaultdict(set)   # coppie MAC reali bloccati almeno una volta, per dpid

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
        """Polling: T = idle_timeout / K, K>1."""
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
        """OFPFC_DELETE_STRICT (match+priority) or OFPFC_DELETE (wildcard)."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        command = ofproto.OFPFC_DELETE_STRICT if strict else ofproto.OFPFC_DELETE
        mod = parser.OFPFlowMod(
            datapath=datapath, command=command,
            out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
            priority=priority, match=match)
        datapath.send_msg(mod)
        self.logger.debug('[EVICTION] dpid=%016x priority=%s match=%s', datapath.id, priority, match)

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

        ip = pkt.get_protocol(ipv4.ipv4) if eth.ethertype == ether_types.ETH_TYPE_IP else None

        # Packet-In Collector
        if ip is not None:
            payload = pkt.protocols[-1]
            if not isinstance(payload, (bytes, bytearray)):
                payload = b""
            self.feature_ext[dpid].add_pkt_in(payload=payload, src_ip=ip.src, dst_ip=ip.dst,
                                              src_mac=src, dst_mac=dst)

        src_real = src.lower() in self.REAL_MACS
        is_real = src_real and dst.lower() in self.REAL_MACS
        self.n_pktin_total[dpid] += 1
        if not src_real:
            self.n_pktin_atk[dpid] += 1
        if is_real:
            self.n_pktin_real[dpid] += 1
            self.real_seen[dpid].add((src.lower(), dst.lower()))

        # Packet-In Blocker: il pacchetto viene comunque inoltrato (PacketOut) ma
        # non si installa nessuna regola e non si apprende il MAC.
        bypass_flow_mod = False
        until = self.blocked.get((dpid, in_port, src, dst))
        if until is not None and time.time() < until:
            self.n_blocked[dpid] += 1
            if not src_real:
                self.n_blocked_pktin_atk[dpid] += 1
            if is_real:
                self.n_blocked_real[dpid] += 1
            bypass_flow_mod = True

        mac_table = self.mac_to_port.setdefault(dpid, {})
        if not bypass_flow_mod:
            mac_table[src] = in_port            # niente apprendimento da MAC bloccati/spoofati
        out_port = mac_table.get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if not bypass_flow_mod and out_port != ofproto.OFPP_FLOOD:
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

    def _update_blocker(self, dpid, mal_stats):
        """ Packet-In Blocker """
        now = time.time()
        for k in {(s.match['in_port'], s.match['eth_src'], s.match['eth_dst']) for s in mal_stats}:
            key = (dpid,) + k
            n, last = self.strikes.get(key, (0, 0.0))
            n = n + 1 if now - last <= self.STRIKE_TTL else 1
            self.strikes[key] = (n, now)
            if n >= self.BLK_MIN_ROUNDS:
                self.blocked[key] = now + self.BLK_TTL
                if k[1].lower() in self.REAL_MACS and k[2].lower() in self.REAL_MACS:
                    self.real_blocked_ever[dpid].add((k[1].lower(), k[2].lower()))
                self.logger.debug('[BLOCK] dpid=%016x port=%s mac=%s strikes=%d', dpid, k[0], k[1], n)
        # Clean the blocker e le strikes scadute (con MAC casuali crescerebbero senza limite)
        for k in [k for k, u in self.blocked.items() if u <= now]:
            del self.blocked[k]
        for k in [k for k, (n, t) in self.strikes.items() if now - t > self.STRIKE_TTL]:
            del self.strikes[k]

    # ----------------------------------------------------------------- #
    # Flow stats -> round
    # ----------------------------------------------------------------- #
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        until = self._ftf_until.get(dpid)
        if until is not None and time.time() > until:
            self._ftf_until[dpid] = None
        stats = [s for s in ev.msg.body if s.priority == 1]
        if not stats:
            self._round_idx[dpid] += 1
            self._prune_fr_windows(dpid, self._round_idx[dpid])
            return
        hub.spawn(self._update_windows_and_maybe_detect, dpid, stats)

    def _prune_fr_windows(self, dpid, rnd):
        """Delete the window of the FR that is not in stats."""
        seen, wins = self._fr_last_seen[dpid], self.w_round_per_fr[dpid]
        for h in [h for h, r in seen.items() if rnd - r > self.FR_PRUNE_GRACE]:
            seen.pop(h, None)
            wins.pop(h, None)

    def _update_windows_and_maybe_detect(self, dpid, stats):
        tm = StageTimer()
        ts0, t0 = time.time(), time.perf_counter()
        self._round_idx[dpid] += 1
        rnd = self._round_idx[dpid]
        fe = self.feature_ext[dpid]

        ftf_win = fr_snap = None
        with tm.stage("prep"):
            self.w_round_per_ft[dpid].append(self._extract_ftf_vector(fe, stats))

            mfr_rows = self._scale(self.mfr, [fe.get_flow_feature(s) for s in stats])
            seen, hashes = self._fr_last_seen[dpid], []
            for stat, row in zip(stats, mfr_rows):
                h = md5(str(stat.match).encode()).hexdigest()
                self.w_round_per_fr[dpid][h].append(row)
                seen[h] = rnd
                hashes.append(h)
            self._prune_fr_windows(dpid, rnd)

            occupancy = len(stats) / float(self.FLOW_TABLE_SIZE)
            ready = fe.has_enough_data() and occupancy >= self.FLOW_TABLE_THRESHOLD
            busy = dpid in self._detecting                      # inferenza/mitigazione precedente in corso
            ftf_active = self._ftf_until.get(dpid) is not None

            run_ftf = ready and not busy and not ftf_active     # serve la detection FTF
            run_mit = ftf_active and not busy                   # attacco gia' dichiarato: solo MFR
            skipped = busy and (ready or ftf_active)

            # Snapshot prima di cedere il controllo: servono sia a run_ftf (la mitigazione
            # parte subito se la FTF risulta positiva) sia a run_mit.
            if run_ftf:
                ftf_win = pad_window(self.w_round_per_ft[dpid], self.ftf["window"], self.ftf["n_feat"])
            if run_ftf or run_mit:
                fr_snap = [tuple(self.w_round_per_fr[dpid][h]) for h in hashes]

        res = None
        ftf_run = ftf_pred = mfr_run = 0
        if fr_snap is not None:
            self._detecting.add(dpid)
            try:
                if run_ftf:
                    pred = self._detect(dpid, stats, ftf_win, tm)
                    ftf_run = int(pred is not None)
                    ftf_pred = int(bool(pred))
                    self._ftf_until[dpid] = (time.time() + self.FTF_TTL) if pred else None
                if self._ftf_until.get(dpid) is not None:
                    # MFR detection + eviction + packet-in blocker
                    res = self._mitigate(dpid, stats, fr_snap, tm)
                    mfr_run = int(res is not None)
            finally:
                self._detecting.discard(dpid)
        res = res or {}

        n_wc, n_strict = res.get("n_eviction_msgs_wildcard", 0), res.get("n_eviction_msgs_strict", 0)
        self._write_report(
            timestamp=round(ts0, 3), dpid=f"{dpid:016x}", round=rnd,
            detection_run=int(bool(ftf_run or mfr_run)), skipped=int(skipped),
            flow_table_occupancy=round(occupancy, 4),
            ftf_detected=int(self._ftf_until.get(dpid) is not None),
            ftf_run=ftf_run, ftf_pred=ftf_pred, mfr_run=mfr_run,
            n_malicious_rules=res.get("n_malicious_rules", 0),
            n_malicious_real=res.get("n_malicious_real", 0),
            n_legit_rules=res.get("n_legit_rules", 0),
            n_fr_windows=len(self.w_round_per_fr[dpid]),
            n_rules_wildcard=res.get("n_rules_wildcard", 0),
            n_eviction_msgs_wildcard=n_wc, n_eviction_msgs_strict=n_strict,
            n_eviction_msgs_total=n_wc + n_strict,
            round_wall_s=round(time.perf_counter() - t0, 4),
            prep_wall_s=round(tm.wall["prep"], 4),   prep_cpu_s=round(tm.cpu["prep"], 4),
            infer_wall_s=round(tm.wall["infer"], 4), infer_cpu_s=round(tm.cpu["infer"], 4),
            mitig_wall_s=round(tm.wall["mitig"], 4), mitig_cpu_s=round(tm.cpu["mitig"], 4),
            n_blocked_pktin=self.n_blocked[dpid],
            n_blocked_pairs=sum(1 for k, u in self.blocked.items() if k[0] == dpid and u > ts0),
            n_pktin_total=self.n_pktin_total[dpid],
            n_pktin_real=self.n_pktin_real[dpid],
            n_blocked_pktin_real=self.n_blocked_real[dpid],
            n_real_pairs_seen=len(self.real_seen[dpid]),
            n_real_pairs_blocked_cum=len(self.real_blocked_ever[dpid]),
            n_blocked_pairs_real=sum(1 for k, u in self.blocked.items() if k[0] == dpid and u > ts0 and k[2].lower() in self.REAL_MACS and k[3].lower() in self.REAL_MACS),
            n_pktin_atk=self.n_pktin_atk[dpid],
            n_blocked_pktin_atk=self.n_blocked_pktin_atk[dpid]
        )

    # ----------------------------------------------------------------- #
    # Feature -> Matrix (n, F) and only one transform
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
    # Inference
    # ----------------------------------------------------------------- #
    @torch.inference_mode()
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
    # Detection FTF, poi MFR + eviction durante la lease
    # ----------------------------------------------------------------- #
    def _detect(self, dpid, stats, ftf_win, tm):
        """Ritorna 0/1, oppure None se lo switch e' sparito (inferenza non eseguita)."""
        if self.datapaths.get(dpid) is None:
            return None

        with tm.stage("infer", cpu=False):
            (ftf_pred,), _, c = self._infer(self.ftf["model"], [ftf_win])
        tm.cpu["infer"] += c

        if ftf_pred:
            self.logger.info(f"[!] FTF Attack Detected on DPID={dpid:016x}")
        return ftf_pred

    def _mitigate(self, dpid, stats, fr_snap, tm):
        if self.datapaths.get(dpid) is None:
            return None

        n_malicious = n_legit = n_rules_wildcard = n_wc_msgs = n_strict = 0
        n_mal_real = 0

        # --- parte 1: finestre per l'MFR
        with tm.stage("mitig"):
            windows = [pad_window(snap, self.mfr["window"], self.mfr["n_feat"])
                       for snap in fr_snap]

        # --- inferenza MFR
        with tm.stage("infer", cpu=False):
            mfr_preds, _, c = self._infer(self.mfr["model"], windows)
        tm.cpu["infer"] += c

        # --- parte 2: classificazione + invio delle delete
        with tm.stage("mitig"):
            malicious_stats = [s for s, p in zip(stats, mfr_preds) if p]
            legit_stats = [s for s, p in zip(stats, mfr_preds) if not p]
            n_malicious, n_legit = len(malicious_stats), len(legit_stats)
            n_mal_real = sum(1 for s in malicious_stats
                             if s.match['eth_src'].lower() in self.REAL_MACS)
            datapath = self.datapaths.get(dpid)   # dopo l'attesa lo switch potrebbe essere sparito
            if datapath is not None and malicious_stats:
                self.logger.info(f"[!] {n_malicious} regole malevole su DPID={dpid:016x}")
                evicted, remaining, n_wc_msgs = self._evict_by_port(datapath, malicious_stats, legit_stats)
                n_rules_wildcard = len(evicted)
                for stat in remaining:
                    self.delete_flow(datapath, stat.match, priority=stat.priority, strict=True)
                    n_strict += 1
                self.logger.info('[EVICTION] dpid=%016x wildcard_msgs=%d covered=%d strict_msgs=%d',
                                 dpid, n_wc_msgs, n_rules_wildcard, n_strict)

                # The Packet-In Blocker collects malicious stats to block
                self._update_blocker(dpid, malicious_stats)

        return {
            "n_malicious_rules": n_malicious, "n_legit_rules": n_legit,
            "n_rules_wildcard": n_rules_wildcard,
            "n_eviction_msgs_wildcard": n_wc_msgs,
            "n_eviction_msgs_strict": n_strict,
            "n_malicious_real": n_mal_real
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