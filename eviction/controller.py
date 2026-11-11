# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Solo le classi nn.Module (nessuna dipendenza dall'ambiente di training:
# niente optuna, utils.py, config.py). Il file di training deve importare le
# stesse classi da qui, cosi' l'architettura ha un'unica sorgente.
from mitigation.tempbert_arch import TempBert


# --------------------------------------------------------------------------- #
# Caricamento modello TempBERT (FTF o MFR) gia' addestrato
# --------------------------------------------------------------------------- #
def load_tempbert(model_dir, model_cls, device):
    """
    model_dir: cartella con il sottodirectory 'temp_bert/' contenente
    model.pt e meta.joblib (prodotti da train_tempBERT).
    """
    temp_bert_dir = os.path.join(model_dir, "temp_bert")
    meta = joblib.load(os.path.join(temp_bert_dir, "meta.joblib"))
    params = meta["params"]
    features = meta["features"]
    scaler = meta["scaler"]
    n_feat = len(features)
    window = params["window"]        # <-- dimensione di finestra del modello, non fissata a priori

    model = model_cls(n_feat, window, params)
    state_dict = torch.load(os.path.join(temp_bert_dir, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model, "scaler": scaler, "features": features,
        "window": window, "n_feat": n_feat,
    }


# Una riga per ogni round di polling (anche quando la detection non parte),
# cosi' il costo del round e' misurabile in tutte le fasi del test.
#   round_wall_s : durata totale del round (include l'attesa dell'inferenza)
#   prep_cpu_s   : CPU nel thread del controller per estrazione feature,
#                  aggiornamento/potatura finestre e snapshot
#   infer_wall_s : tempo reale speso nel thread nativo di inferenza
#   infer_cpu_s  : CPU del thread di inferenza (thread_time: i thread
#                  helper interni di torch NON sono inclusi)
#   mitig_cpu_s  : CPU nel thread del controller per build finestre MFR + eviction
REPORT_COLUMNS = [
    "timestamp", "dpid", "round", "detection_run", "skipped",
    "flow_table_occupancy", "ftf_detected",
    "n_malicious_rules", "n_legit_rules", "n_fr_windows",
    "n_rules_wildcard",
    "n_eviction_msgs_wildcard", "n_eviction_msgs_strict", "n_eviction_msgs_total",
    "round_wall_s", "prep_cpu_s", "infer_wall_s", "infer_cpu_s", "mitig_cpu_s",
]


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

    # Quanti round una regola puo' mancare dalle stats prima che la sua
    # finestra MFR venga eliminata. 0 = si elimina appena sparisce.
    # Se la stessa regola (stesso match) riappare dopo la scadenza, con 0
    # la sua storia riparte da zero.
    FR_PRUNE_GRACE = int(os.environ.get('FR_PRUNE_GRACE', '0'))

    FTF_MODEL_DIR = os.environ.get('FTF_MODEL_DIR', root + '/models/ftf_detector')
    MFR_MODEL_DIR = os.environ.get('MFR_MODEL_DIR', root + '/models/mfr_detector')

    FLOW_TABLE_THRESHOLD = 0.8

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.logger.setLevel(self.LOG_LEVEL)
        self.mac_to_port = {}
        self.datapaths = {}

        # defaultdict con factory corretta (bug nello scheletro originale:
        # FeatureExt richiede window_size, e un defaultdict() senza factory
        # si comporta come un dict normale e solleva KeyError).
        self.feature_ext = defaultdict(lambda: FeatureExt(window_size=self.W_PKTIN))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"TempBERT inference device: {self.device}")

        ftf = load_tempbert(self.FTF_MODEL_DIR, TempBert, self.device)
        self.ftf_model = ftf["model"]
        self.ftf_scaler = ftf["scaler"]
        self.ftf_features = ftf["features"]
        self.ftf_window_size = ftf["window"]
        self.ftf_n_feat = ftf["n_feat"]
        self.logger.info(f"FTF TempBERT caricato: window={self.ftf_window_size}, "
                          f"n_feat={self.ftf_n_feat}, da {self.FTF_MODEL_DIR}")

        mfr = load_tempbert(self.MFR_MODEL_DIR, TempBert, self.device)
        self.mfr_model = mfr["model"]
        self.mfr_scaler = mfr["scaler"]
        self.mfr_features = mfr["features"]
        self.mfr_window_size = mfr["window"]
        self.mfr_n_feat = mfr["n_feat"]
        self.logger.info(f"MFR TempBERT caricato: window={self.mfr_window_size}, "
                          f"n_feat={self.mfr_n_feat}, da {self.MFR_MODEL_DIR}")

        # Finestre W_round come code: una per DPID (FT), una per ogni
        # coppia (DPID, match_hash) di flow rule (FR). maxlen = la window del
        # rispettivo modello: una volta piena, append+eviction del piu'
        # vecchio e' gestito automaticamente da deque, cosi' come il
        # progressivo riempimento da "tutto padding" a "finestra piena"
        # corrisponde esattamente al comportamento di build_windows in
        # training (padding a sinistra finche' la storia reale e' < window).
        self.w_round_per_ft = defaultdict(lambda: deque(maxlen=self.ftf_window_size))
        self.w_round_per_fr = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.mfr_window_size)))

        # Punto 1: ultimo round in cui ogni regola e' comparsa nelle stats,
        # serve a potare le finestre FR delle regole scadute/evictate.
        self._round_idx = defaultdict(int)
        self._fr_last_seen = defaultdict(dict)

        # Punto 3: l'inferenza gira in UN thread nativo (tpool), cosi' l'event
        # loop di Ryu continua a servire Packet-In ed echo mentre torch calcola.
        # Un solo thread = inferenze serializzate, niente CPU concorrente
        # che falserebbe le misure.
        tpool.set_num_threads(1)
        # DPID con una detection in corso: evita che round successivi ne
        # lancino una seconda mentre la prima non e' ancora tornata.
        self._detecting = set()

        self.T = float("{:.2f}".format(self.IDLE_TIMEOUT / self.K_FACTOR))
        self.logger.debug(f"K: {self.K_FACTOR}, T: {self.T}s")
        self.logger.debug(f"W_pktin: {self.W_PKTIN}")

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
        """
        T_max = min { idle, hard } tempo massimo per intercettare una flow rule
        prima che venga rimossa

        Per il teorema del campionamento
        - fs > 2*fM
        - 1/ts > 2*1/tM (=>) 2*ts <= tM

        (=>) T <= T_max/2

        Per catturare piu'campioni si puo' considerare un fattore k>=2

        (=>) T <= T_max/k

        K piu' grande => T_k piu' piccolo => polling piu' frequente per quel K.
        """
        while True:
            for datapath in list(self.datapaths.values()):
                self._request_stats(datapath)
            hub.sleep(self.T)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    """
    Per gli switch
    """
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Switch registrato: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Switch disconnesso: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                              actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                 match=match, instructions=inst,
                                 idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    def delete_flow(self, datapath, match, priority=1, strict=True):
        """
        Send an OFPFlowMod:
        - OFPFC_DELETE_STRICT (strict=True) deletes flow rules by match+priority
        - OFPFC_DELETE (strict=False) deletes flow rules by wildcard
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        command = ofproto.OFPFC_DELETE_STRICT if strict else ofproto.OFPFC_DELETE
        mod = parser.OFPFlowMod(
            datapath=datapath, command=command,
            out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
            priority=priority, match=match)
        datapath.send_msg(mod)
        self.logger.info(
            '[EVICTION] dpid=%016x priority=%s match=%s',
            datapath.id, priority, match
        )

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

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        # DEBUG: a livello INFO, sotto flooding, il logging domina la CPU
        self.logger.debug("packet in %s %s %s %s", dpid, src, dst, in_port)

        self.mac_to_port[dpid][src] = in_port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        ip = None
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip = pkt.get_protocol(ipv4.ipv4)
            if ip is None:
                self.logger.warning(
                    "Pacchetto dichiarato IP ma header ipv4 non estraibile, dpid=%s in_port=%s",
                    dpid, in_port
                )

        if out_port != ofproto.OFPP_FLOOD:
            if eth.ethertype == ether_types.ETH_TYPE_IP and ip is not None:
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    eth_dst=dst, eth_src=src,
                    ipv4_dst=ip.dst, ipv4_src=ip.src
                )
            else:
                match = parser.OFPMatch(
                    in_port=in_port, eth_dst=dst, eth_src=src
                )
            self.add_flow(
                datapath, 1, match, actions,
                idle_timeout=self.IDLE_TIMEOUT,
                hard_timeout=self.HARD_TIMEOUT
            )

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

        """
        Packet-In Collector
        """
        if eth.ethertype != ether_types.ETH_TYPE_IP or ip is None:
            return

        payload = pkt.protocols[-1]
        if not isinstance(payload, (bytes, bytearray)):
            payload = b""

        self.feature_ext[dpid].add_pkt_in(
            payload=payload,
            src_ip=ip.src,
            dst_ip=ip.dst,
            src_mac=src,
            dst_mac=dst
        )

    # ----------------------------------------------------------------- #
    # Flow Stats Handler
    # ----------------------------------------------------------------- #
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        stats = [stat for stat in body if stat.priority == 1]
        if not stats:
            # Tabella vuota (es. dopo un'eviction): nessuna finestra da
            # aggiornare, ma quelle delle regole sparite vanno comunque potate.
            self._round_idx[dpid] += 1
            self._prune_fr_windows(dpid, self._round_idx[dpid])
            return
        hub.spawn(self._update_windows_and_maybe_detect, dpid, stats)

    # ----------------------------------------------------------------- #
    # Punto 1: potatura delle finestre per-regola
    # ----------------------------------------------------------------- #
    def _prune_fr_windows(self, dpid, rnd):
        """Elimina le finestre FR delle regole non piu' presenti nelle stats."""
        seen = self._fr_last_seen[dpid]
        wins = self.w_round_per_fr[dpid]
        stale = [h for h, r in seen.items() if rnd - r > self.FR_PRUNE_GRACE]
        for h in stale:
            seen.pop(h, None)
            wins.pop(h, None)

    # ----------------------------------------------------------------- #
    # Round completo: parte sincrona (feature + finestre) e, se serve,
    # detection. La parte sincrona NON cede mai il controllo, quindi i suoi
    # tempi CPU sono attribuibili al solo round.
    # ----------------------------------------------------------------- #
    def _update_windows_and_maybe_detect(self, dpid, stats):
        ts0 = time.time()
        t0 = time.perf_counter()
        c0 = time.thread_time()

        self._round_idx[dpid] += 1
        rnd = self._round_idx[dpid]

        # Aggiorna SEMPRE la finestra FT, indipendentemente dalla soglia di
        # occupazione: se aspettassimo la soglia per cominciare a popolare la
        # storia, il modello vedrebbe una finestra quasi tutta di padding
        # proprio nel momento in cui servirebbe davvero.
        ftf_vec = self._extract_ftf_vector(dpid, stats)
        self.w_round_per_ft[dpid].append(ftf_vec)

        # Anche le finestre per-regola vanno SEMPRE aggiornate, per non
        # perdere storia nei round in cui la detection non parte.
        seen = self._fr_last_seen[dpid]
        hashes = []
        for stat in stats:
            h = sha256(str(stat.match).encode()).hexdigest()
            self.w_round_per_fr[dpid][h].append(self._extract_mfr_vector(dpid, stat))
            seen[h] = rnd
            hashes.append(h)
        self._prune_fr_windows(dpid, rnd)

        occupancy = len(stats) / float(self.FLOW_TABLE_SIZE)
        ready = self.feature_ext[dpid].has_enough_data() and occupancy >= self.FLOW_TABLE_THRESHOLD
        skipped = ready and dpid in self._detecting   # inferenza precedente ancora in corso
        detect = ready and not skipped

        # Snapshot prima di cedere il controllo: round successivi possono
        # modificare le deque mentre questa detection e' in attesa.
        ftf_win = fr_snap = None
        if detect:
            self.logger.debug(f"DPID={dpid:016x} flow table {occupancy:.1%} full")
            ftf_win = self._build_window(
                self.ftf_window_size, self.ftf_n_feat, self.w_round_per_ft[dpid])
            fr_snap = [tuple(self.w_round_per_fr[dpid][h]) for h in hashes]
        prep_cpu = time.thread_time() - c0

        res = None
        if detect:
            self._detecting.add(dpid)
            try:
                res = self._detect_and_mitigate(dpid, stats, ftf_win, fr_snap)
            finally:
                self._detecting.discard(dpid)
        res = res or {}

        n_wc_msgs = res.get("n_eviction_msgs_wildcard", 0)
        n_strict = res.get("n_eviction_msgs_strict", 0)
        self._write_report(
            timestamp=round(ts0, 3), dpid=f"{dpid:016x}", round=rnd,
            detection_run=int(bool(res)), skipped=int(skipped),
            flow_table_occupancy=round(occupancy, 4),
            ftf_detected=res.get("ftf_detected", 0),
            n_malicious_rules=res.get("n_malicious_rules", 0),
            n_legit_rules=res.get("n_legit_rules", 0),
            n_fr_windows=len(self.w_round_per_fr[dpid]),
            n_rules_wildcard=res.get("n_rules_wildcard", 0),
            n_eviction_msgs_wildcard=n_wc_msgs,
            n_eviction_msgs_strict=n_strict,
            n_eviction_msgs_total=n_wc_msgs + n_strict,
            round_wall_s=round(time.perf_counter() - t0, 4),
            prep_cpu_s=round(prep_cpu, 4),
            infer_wall_s=round(res.get("infer_wall_s", 0.0), 4),
            infer_cpu_s=round(res.get("infer_cpu_s", 0.0), 4),
            mitig_cpu_s=round(res.get("mitig_cpu_s", 0.0), 4),
        )

    # ----------------------------------------------------------------- #
    # Estrazione feature -> vettore ordinato come il modello si aspetta,
    # poi scalato con lo stesso QuantileTransformer fittato in training.
    # ----------------------------------------------------------------- #
    def _extract_ftf_vector(self, dpid, stats):
        ((payload_entropy_mean, payload_entropy_cv), payload_size_entropy,
         src_ip_entropy, dst_ip_entropy,
         src_mac_entropy, dst_mac_entropy) = self.feature_ext[dpid].get_pkt_in_features()

        (c_used, paf, ft_psi,
         durations_entropy, pkts_entropy, bytes_entropy,
         (mean_duration, cvd),
         (mean_pkts, cvp),
         (mean_bytes, cvb)) = self.feature_ext[dpid].get_ft_features(stats, self.FLOW_TABLE_SIZE)

        raw = {
            "payload_entropy_mean": payload_entropy_mean,
            "payload_entropy_cv": payload_entropy_cv,
            "payload_size_entropy": payload_size_entropy,
            "src_ip_entropy": src_ip_entropy,
            "dst_ip_entropy": dst_ip_entropy,
            "src_mac_entropy": src_mac_entropy,
            "dst_mac_entropy": dst_mac_entropy,
            "c_used": c_used,
            "paf": paf,
            "ft_psi": ft_psi,
            "durations_entropy": durations_entropy,
            "pkts_entropy": pkts_entropy,
            "bytes_entropy": bytes_entropy,
            "mean_duration": mean_duration,
            "cvd": cvd,
            "mean_pkts": mean_pkts,
            "cvp": cvp,
            "mean_bytes": mean_bytes,
            "cvb": cvb,
        }
        ordered = np.array([[raw[f] for f in self.ftf_features]], dtype=np.float64)
        return self.ftf_scaler.transform(ordered).astype(np.float32)[0]

    def _extract_mfr_vector(self, dpid, stat):
        duration, packet_count, byte_count, pkt_rate, byte_rate, psi = \
            self.feature_ext[dpid].get_flow_feature(stat)
        raw = {
            "duration": duration,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "packets_per_seconds": pkt_rate,
            "bytes_per_seconds": byte_rate,
            "psi": psi,
        }
        ordered = np.array([[raw[f] for f in self.mfr_features]], dtype=np.float64)
        return self.mfr_scaler.transform(ordered).astype(np.float32)[0]

    # ----------------------------------------------------------------- #
    # Predizione TempBERT su una finestra
    # ----------------------------------------------------------------- #
    @staticmethod
    def _build_window(window_size, n_feat, snapshot):
        """
        Replica esattamente la logica di padding di build_windows: i round
        reali vanno allineati a destra (il piu' recente e' sempre l'ultimo
        token), gli slot iniziali mancanti restano a zero con valid=False.
        `snapshot` e' una deque o una tupla; np.stack copia i dati, quindi
        la finestra restituita e' indipendente dalle strutture condivise.
        """
        k = len(snapshot)
        seq = np.zeros((window_size, n_feat), dtype=np.float32)
        valid = np.zeros((window_size,), dtype=bool)
        if k > 0:
            real = np.stack(list(snapshot), axis=0)   # (k, n_feat), k <= window_size (maxlen)
            seq[window_size - k:] = real
            valid[window_size - k:] = True
        return seq, valid

    @torch.no_grad()
    def _forward_many(self, model, windows, max_batch=512):
        """
        Gira in un thread nativo (tpool): tocca solo gli array ricevuti e il
        modello, mai le strutture condivise del controller. Le finestre vengono
        impilate e processate in batch (a blocchi di max_batch per limitare la
        memoria con molte regole). Restituisce (predizioni, wall, cpu).
        """
        t_wall = time.perf_counter()
        t_cpu = time.thread_time()
    
        if not windows:
            return [], time.perf_counter() - t_wall, time.thread_time() - t_cpu
    
        seqs = np.stack([seq for seq, _ in windows])        # (N, W, F)
        valids = np.stack([valid for _, valid in windows])  # (N, W)
    
        preds = []
        for i in range(0, len(windows), max_batch):
            seq_t = torch.from_numpy(seqs[i:i + max_batch]).to(self.device)      # (B, W, F)
            valid_t = torch.from_numpy(valids[i:i + max_batch]).to(self.device)  # (B, W)
            logits = model(seq_t, valid_t)                                       # (B, 2)
            preds.extend(logits.argmax(-1).tolist())
    
        return preds, time.perf_counter() - t_wall, time.thread_time() - t_cpu

    def _infer(self, model, windows):
        """Sospende solo questo greenlet: l'event loop di Ryu resta libero."""
        return tpool.execute(self._forward_many, model, windows)

    # ----------------------------------------------------------------- #
    # Detection (FTF) + classificazione per-regola (MFR) + eviction
    # ----------------------------------------------------------------- #
    def _detect_and_mitigate(self, dpid, stats, ftf_win, fr_snap):
        if self.datapaths.get(dpid) is None:
            return None  # switch disconnesso nel frattempo

        infer_wall = infer_cpu = mitig_cpu = 0.0
        n_malicious = n_legit = n_rules_wildcard = n_wc_msgs = n_strict = 0

        (ftf_pred,), w, c = self._infer(self.ftf_model, [ftf_win])
        infer_wall += w
        infer_cpu += c

        if ftf_pred:
            c0 = time.thread_time()
            windows = [
                self._build_window(self.mfr_window_size, self.mfr_n_feat, snap)
                for snap in fr_snap
            ]
            mitig_cpu += time.thread_time() - c0

            mfr_preds, w, c = self._infer(self.mfr_model, windows)
            infer_wall += w
            infer_cpu += c

            c0 = time.thread_time()
            malicious_stats = [s for s, p in zip(stats, mfr_preds) if p]
            legit_stats = [s for s, p in zip(stats, mfr_preds) if not p]
            n_malicious, n_legit = len(malicious_stats), len(legit_stats)

            self.logger.info(f"[!] FTF Attack Detected on DPID={dpid:016x}")
            # Dopo l'attesa lo switch potrebbe essere sparito
            datapath = self.datapaths.get(dpid)
            if datapath is not None and malicious_stats:
                self.logger.info(f"[!] {n_malicious} regole malevole su DPID={dpid:016x}")
                evicted, remaining, n_wc_msgs = self._evict_by_port(
                    datapath, malicious_stats, legit_stats)
                n_rules_wildcard = len(evicted)
                for stat in remaining:
                    self.delete_flow(datapath, stat.match, priority=stat.priority, strict=True)
                    n_strict += 1
            elif not malicious_stats:
                self.logger.debug(f"[-] Nessuna regola malevola su DPID={dpid:016x}")
            mitig_cpu += time.thread_time() - c0
        else:
            self.logger.debug(f"[-] Nessun attacco FTF rilevato su DPID={dpid:016x}")

        return {
            "ftf_detected": int(bool(ftf_pred)),
            "n_malicious_rules": n_malicious, "n_legit_rules": n_legit,
            "n_rules_wildcard": n_rules_wildcard,
            "n_eviction_msgs_wildcard": n_wc_msgs,
            "n_eviction_msgs_strict": n_strict,
            "infer_wall_s": infer_wall, "infer_cpu_s": infer_cpu,
            "mitig_cpu_s": mitig_cpu,
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

        evicted, remaining = [], []
        n_wc_msgs = 0
        for in_port, group in by_port.items():
            collateral = legit_by_port.get(in_port, 0)

            if len(group) < 2:
                # una sola regola malevola sulla porta: non conviene un
                # wildcard, il delete strict e' gia' ottimo (1 msg, 0 collateral)
                remaining.extend(group)
                continue

            if collateral <= self.MAX_COLLATERAL_PORT:
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(in_port=in_port)
                self.delete_flow(datapath, match, priority=1, strict=False)
                n_wc_msgs += 1

                self.logger.info(
                    '[EVICTION-L0] Dpid=%016x discarded in_port=%s covered=%d collateral=%d',
                    datapath.id, in_port, len(group), collateral
                )
                evicted.extend(group)
            else:
                self.logger.debug(
                    '[EVICTION-L0] in_port=%s scartata: collateral=%d > soglia=%d, fallback',
                    in_port, collateral, self.MAX_COLLATERAL_PORT
                )
                remaining.extend(group)

        return evicted, remaining, n_wc_msgs
