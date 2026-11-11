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

from netaddr import IPAddress, IPNetwork
from collections import defaultdict
import csv
import heapq
import threading
import time
from hashlib import sha256

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
from mitigation.feature_ext import FeatureExt

class FTFMController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    DATASET_FTF_COLUMNS = [
        "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
        "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
        "c_used", "paf", "ft_psi", "durations_entropy", "pkts_entropy", "bytes_entropy",
        "mean_duration", "cvd", "mean_pkts", "cvp", "mean_bytes", "cvb", 
        "round", "scenario_id", "label"
    ]
    DATASET_MFR_COLUMNS = [
        "match_hash", "duration", "packet_count", "byte_count",
        "packets_per_seconds", "bytes_per_seconds", "psi",
        "round", "scenario_id", "label"
    ]
    SCENARIO_ID = int(os.environ.get('SCENARIO_ID', '0'))
    FTF_LABEL = int(os.environ.get('FTF_LABEL', '0'))  # 0 buono, 1 cattivo
    DATASET_FTF_PATH = os.environ.get('DATASET_FTF_PATH', '/tmp/dataset_ftf')
    DATASET_MFR_PATH = os.environ.get('DATASET_MFR_PATH', '/tmp/dataset_mfr')
    FLOW_TABLE_SIZE = int(os.environ.get('FLOW_TABLE_SIZE', '50'))
    IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '10'))
    HARD_TIMEOUT = int(os.environ.get('HARD_TIMEOUT', '20'))
    DPID = int(os.environ.get('DPID', '0000000000000001'), 16)
    VALID_MACS = [
        '00:00:00:00:00:01', '00:00:00:00:00:02', '00:00:00:00:00:03',
        '00:00:00:00:00:04', '00:00:00:00:00:05', '00:00:00:00:00:06',
        '00:00:00:00:00:07', '00:00:00:00:00:08', '00:00:00:00:00:09',
        '00:00:00:00:00:0a', '00:00:00:00:00:0b', '00:00:00:00:00:0c',
        '00:00:00:00:00:0d', '00:00:00:00:00:0e'
    ]

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}

        self.WINDOW_SIZE = [10, 100, 1000]
        self.K_FACTOR = [2, 3, 4]
        self.feature_ext = [FeatureExt(window_size=ws) for ws in self.WINDOW_SIZE]

        self._init_csv()
        self._csv_lock = threading.Lock()
        self._rounds_ftf = {(k, ws): 0 for k in self.K_FACTOR for ws in self.WINDOW_SIZE}
        self._rounds_mfr = {(k, ws): 0 for k in self.K_FACTOR for ws in self.WINDOW_SIZE}
        self._pending_requests = {}
        self.PENDING_TTL = 2 * self.IDLE_TIMEOUT

        self.collector = hub.spawn(self._collector)

    def _init_csv(self):
        """
        Crea 9 file per FTF e 9 per MFR: uno per ogni combinazione
        (K_FACTOR, WINDOW_SIZE). Nome file: <DATASET_..._PATH>_<K>_<W>.csv
        """
        self._csv_file_ftf = {}
        self._csv_writer_ftf = {}
        self._csv_file_mfr = {}
        self._csv_writer_mfr = {}

        for k in self.K_FACTOR:
            for ws in self.WINDOW_SIZE:
                path_ftf = f"{self.DATASET_FTF_PATH}_{k}_{ws}.csv"
                write_header_ftf = not os.path.exists(path_ftf)
                f_ftf = open(path_ftf, "a", newline="")
                w_ftf = csv.writer(f_ftf)
                if write_header_ftf:
                    w_ftf.writerow(self.DATASET_FTF_COLUMNS)
                    f_ftf.flush()
                self._csv_file_ftf[(k, ws)] = f_ftf
                self._csv_writer_ftf[(k, ws)] = w_ftf

                path_mfr = f"{self.DATASET_MFR_PATH}_{k}_{ws}.csv"
                write_header_mfr = not os.path.exists(path_mfr)
                f_mfr = open(path_mfr, "a", newline="")
                w_mfr = csv.writer(f_mfr)
                if write_header_mfr:
                    w_mfr.writerow(self.DATASET_MFR_COLUMNS)
                    f_mfr.flush()
                self._csv_file_mfr[(k, ws)] = f_mfr
                self._csv_writer_mfr[(k, ws)] = w_mfr

    def close(self):
        """Chiude ordinatamente tutti i file CSV allo spegnimento dell'app."""
        super(FTFMController, self).close()
        for f in list(self._csv_file_ftf.values()) + list(self._csv_file_mfr.values()):
            try:
                f.close()
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

        Scheduler multi-rate con scadenze assolute: per ogni K si tiene la
        prossima scadenza teorica in un min-heap. 
        Il loop dorme fino alla scadenza piu' vicina; dopo la scadenza, la successiva
        e' calcolata a partire dalla scadenza TEORICA.
        Se piu' K scadono insieme si invia una sola FlowStatsRequest.
        """
        self.T = {k: self.IDLE_TIMEOUT / k for k in self.K_FACTOR}

        for k in sorted(self.K_FACTOR):
            self.logger.debug(f"[GAP] K={k}: T={self.T[k]:.3f}s")

        t0 = time.monotonic()
        heap = [(t0 + self.T[k], k) for k in self.K_FACTOR]
        heapq.heapify(heap)

        while True:
            delay = heap[0][0] - time.monotonic()
            if delay > 0:
                hub.sleep(delay)

            now = time.monotonic()
            due_ks = []
            while heap and heap[0][0] <= now:
                deadline, k = heapq.heappop(heap)
                due_ks.append(k)
                nxt = deadline + self.T[k]
                while nxt <= now:
                    nxt += self.T[k]
                heapq.heappush(heap, (nxt, k))

            self._prune_pending(now)

            datapath = next(
                (dp for dp in self.datapaths.values() if dp.id == self.DPID),
                None,
            )
            if datapath is None:
                continue

            self._request_stats(datapath, sorted(due_ks), now)

    def _prune_pending(self, now):
        """Elimina le richieste che non hanno mai ricevuto risposta."""
        stale = [key for key, p in self._pending_requests.items()
                 if now - p["sent"] > self.PENDING_TTL]
        for key in stale:
            self.logger.warning("Richiesta flow stats %s senza risposta; scarto.", key)
            del self._pending_requests[key]

    def _request_stats(self, datapath, due_ks, now):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.set_xid(req)
        self._pending_requests[(datapath.id, req.xid)] = {
            "due_ks": due_ks,
            "sent": now,
            "stats": [],
        }
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
            for key in [key for key in self._pending_requests if key[0] == datapath.id]:
                del self._pending_requests[key]

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
            # ignore lldp packet
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
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

        # install a flow to avoid packet_in next time
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

        if dpid == self.DPID:
            # Ogni finestra W accumula sempre, indipendentemente da K:
            # K riguarda solo la frequenza di polling delle flow stats.
            for i in range(len(self.WINDOW_SIZE)):
                self.feature_ext[i].add_pkt_in(
                    payload=payload,
                    src_ip=ip.src,
                    dst_ip=ip.dst,
                    src_mac=src,
                    dst_mac=dst
                )

    """
    Flow Stats Handler
    """
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        key = (datapath.id, msg.xid)

        pending = self._pending_requests.get(key)
        if pending is None:
            self.logger.warning(
                "Flow stats reply (dpid=%s xid=%s) senza richiesta pendente "
                "corrispondente; scarto.", datapath.id, msg.xid
            )
            return

        pending["stats"].extend(msg.body)
        if msg.flags & ofproto.OFPMPF_REPLY_MORE:
            return

        del self._pending_requests[key]
        due_ks = pending["due_ks"]
        stats = [stat for stat in pending["stats"] if stat.priority == 1]

        if not stats:
            return

        for k in due_ks:
            for w_idx in range(len(self.WINDOW_SIZE)):
                if self.feature_ext[w_idx].has_enough_data(n_pktin=2):
                    hub.spawn(self._ftf_stats, k, w_idx, stats)
                    if self.FTF_LABEL:  # attack mode
                        hub.spawn(self._mfr_stats, k, w_idx, stats)

    def _ftf_stats(self, k, w_idx, stats):
        ws = self.WINDOW_SIZE[w_idx]

        # Get Pkt In Feature
        ((payload_entropy_mean, payload_entropy_cv), payload_size_entropy,
         src_ip_entropy, dst_ip_entropy,
         src_mac_entropy, dst_mac_entropy) = self.feature_ext[w_idx].get_pkt_in_features()

        # Get Flow Table Feature
        (c_used, paf, ft_psi,
         durations_entropy, pkts_entropy, bytes_entropy,
         (mean_duration, cvd),
         (mean_pkts, cvp),
         (mean_bytes, cvb)) = self.feature_ext[w_idx].get_ft_features(stats, self.FLOW_TABLE_SIZE)

        # Feature (senza round): il round si assegna solo alla scrittura.
        features_ftf = [
            payload_entropy_mean, payload_entropy_cv, payload_size_entropy,
            src_ip_entropy, dst_ip_entropy, src_mac_entropy, dst_mac_entropy,
            c_used, paf, ft_psi, durations_entropy, pkts_entropy, bytes_entropy,
            mean_duration, cvd, mean_pkts, cvp, mean_bytes, cvb,
        ]

        with self._csv_lock:
            rnd = self._rounds_ftf[(k, ws)]
            row_ftf = features_ftf + [rnd, self.SCENARIO_ID, self.FTF_LABEL]
            self._csv_writer_ftf[(k, ws)].writerow(row_ftf)
            self._csv_file_ftf[(k, ws)].flush()
            self._rounds_ftf[(k, ws)] = rnd + 1

        self.logger.debug(f"[FTF] K={k} W={ws} round={rnd}")

    def _mfr_stats(self, k, w_idx, stats):
        ws = self.WINDOW_SIZE[w_idx]

        # Get Flow Rules Features (senza round)
        flows_mfr = []
        for stat in stats:
            match_hash = sha256(str(stat.match).encode()).hexdigest()
            duration, packet_count, byte_count, pkt_rate, byte_rate, psi = self.feature_ext[w_idx].get_flow_feature(stat)
            label = int(stat.match.get('eth_src') not in self.VALID_MACS or
                        stat.match.get('eth_dst') not in self.VALID_MACS)
            flows_mfr.append((
                [match_hash, duration, packet_count, byte_count, pkt_rate, byte_rate, psi],
                label
            ))

        with self._csv_lock:
            rnd = self._rounds_mfr[(k, ws)]
            rows_mfr = [feat + [rnd, self.SCENARIO_ID, label] for feat, label in flows_mfr]
            self._csv_writer_mfr[(k, ws)].writerows(rows_mfr)
            self._csv_file_mfr[(k, ws)].flush()
            self._rounds_mfr[(k, ws)] = rnd + 1

        self.logger.debug(f"[MFR] K={k} W={ws} round={rnd}")