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

from time import time
from collections import defaultdict 

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
from mitigation.feature_ext import FeatureExt

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from transformers import BertForSequenceClassification

class FTFMController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    REPORT_PATH = os.environ.get('REPORT_PATH', current + '/logs/report.txt')
    FLOW_TABLE_SIZE = int(os.environ.get('FLOW_TABLE_SIZE', '50'))
    IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '10'))
    HARD_TIMEOUT = int(os.environ.get('HARD_TIMEOUT', '20'))
    K_FACTOR = int(os.environ.get('K_FACTOR', '3'))
    FLOW_TABLE_THRESHOLD = 0.8
    MODEL = os.environ.get('MODEL', 'Catboost')

    CATBOOST_FTF_DETECTOR = root + '/models/ftf_detector/catboost_model.cbm' 
    CATBOOST_MFR_DETECTOR = root + '/models/malicious_fr_detector/catboost_model.cbm'
    XGBOOST_FTF_DETECTOR = root + '/models/ftf_detector/xgboost_model.json' 
    XGBOOST_MFR_DETECTOR = root + '/models/malicious_fr_detector/xgboost_model.json'
    BERT_FTF_DETECTOR = root + '/models/ftf_detector/hf_bert_model'
    BERT_MFR_DETECTOR = root + '/models/malicious_fr_detector/hf_bert_model'

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.collector = hub.spawn(self._collector)
        self.feature_ext = defaultdict(FeatureExt)
        self.T = float("{:.2f}".format(min(self.IDLE_TIMEOUT, self.HARD_TIMEOUT) / self.K_FACTOR))
        self.logger.debug(f"[GAP] Intervallo di tempo per la cattura delle features: {self.T}s")
        self.collector = hub.spawn(self._collector)

        if self.MODEL == "Catboost":
            self.ftf_detector = CatBoostClassifier()
            self.mfr_detector = CatBoostClassifier()
            self.ftf_detector.load_model(self.CATBOOST_FTF_DETECTOR)
            self.mfr_detector.load_model(self.CATBOOST_MFR_DETECTOR)
        elif self.MODEL == "XGBoost":
            self.ftf_detector = XGBClassifier()
            self.mfr_detector = XGBClassifier()
            self.ftf_detector.load_model(self.XGBOOST_FTF_DETECTOR)
            self.mfr_detector.load_model(self.XGBOOST_MFR_DETECTOR)
        elif self.MODEL == "HFBert":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.ftf_bert = BertForSequenceClassification.from_pretrained(self.BERT_FTF_DETECTOR)
            self.ftf_bert.to(self.device).eval()
            self.ftf_bert_discretizer = joblib.load(os.path.join(self.BERT_FTF_DETECTOR, 'discretizer.joblib'))
            self.ftf_bert_n_bins = int(self.ftf_bert_discretizer.n_bins_[0])
            self.mfr_bert = BertForSequenceClassification.from_pretrained(self.BERT_MFR_DETECTOR)
            self.mfr_bert.to(self.device).eval()
            self.mfr_bert_discretizer = joblib.load(os.path.join(self.BERT_MFR_DETECTOR, 'discretizer.joblib'))
            self.mfr_bert_n_bins = int(self.mfr_bert_discretizer.n_bins_[0])
        else:
            self.ftf_detector = CatBoostClassifier()
            self.mfr_detector = CatBoostClassifier()
            self.ftf_detector.load_model(self.CATBOOST_FTF_DETECTOR)
            self.mfr_detector.load_model(self.CATBOOST_MFR_DETECTOR)

        with open(self.REPORT_PATH, "w", newline="") as f:
            f.write(f"""
            {self.MODEL} FTF DETECTOR & MFR DETECTOR
            
            Configurations:
            - FLOW_TABLE_SIZE: {int(os.environ.get('FLOW_TABLE_SIZE', '50'))}
            - FLOW_TABLE_THRESHOLD: 0.8
            - IDLE_TIMEOUT: {int(os.environ.get('IDLE_TIMEOUT', '10'))}
            - HARD_TIMEOUT: {int(os.environ.get('HARD_TIMEOUT', '20'))}
            - K_FACTOR: {int(os.environ.get('K_FACTOR', '3'))}
            - T: {self.T}
            """)

    def _bert_tokenize(self, discretizer, n_bins, features):
        """Applica lo stesso discretizer (fit in training) a un singolo
        vettore di feature e lo trasforma in una sequenza di token id,
        replicando esattamente _tokenize_features() dello script di training."""
        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
        bins = discretizer.transform(X).astype(np.int64)
        offsets = (np.arange(X.shape[1]) * n_bins).astype(np.int64)
        return bins + offsets

    def _bert_predict(self, model, discretizer, n_bins, features):
        """Ritorna la classe predetta (0/1) dal modello BERT per un flow."""
        input_ids = self._bert_tokenize(discretizer, n_bins, features)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids_t)
        with torch.no_grad():
            outputs = model(input_ids=input_ids_t, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return pred

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

        Il valore k=2 e' minimo, k=3 o k=4 e' il piu' sicuro dipende da quanto 
        vogliamo raccogliere in un periodo.
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

        # install table-miss flow entry
        #
        # We specify NO BUFFER to max_len of the output action due to
        # OVS bug. At this moment, if we specify a lesser number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and
        # truncated packet data. In that case, we cannot output packets
        # correctly.  The bug has been fixed in OVS v2.1.0.
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

        self.feature_ext[dpid].add_pkt_in(
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
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        stats = [stat for stat in body if stat.priority == 1]
        if not stats:
            return
        # Check occupancy
        occupancy = len(stats) / float(self.FLOW_TABLE_SIZE)
        if occupancy < self.FLOW_TABLE_THRESHOLD:
            pass
        hub.spawn(self._process_stats, dpid, stats) 

    def _process_stats(self, dpid, stats):
        self.logger.debug(f"[GAP] DPID: {dpid:016x}")

        # Get Pkt In Feature
        ((payload_entropy_mean, payload_entropy_cv), payload_size_entropy,
         src_ip_entropy, dst_ip_entropy,
         src_mac_entropy, dst_mac_entropy) = self.feature_ext[dpid].get_pkt_in_features()

        # Get Flow Table Feature
        (c_used, paf, ft_psi,
         durations_entropy, pkts_entropy, bytes_entropy,
         (mean_duration, cvd),
         (mean_pkts, cvp),
         (mean_bytes, cvb)) = self.feature_ext[dpid].get_ft_features(stats, self.FLOW_TABLE_SIZE)

        ft_features = [
            payload_entropy_mean, payload_entropy_cv, payload_size_entropy,
            src_ip_entropy, dst_ip_entropy, src_mac_entropy, dst_mac_entropy,
            c_used, paf, ft_psi, durations_entropy, pkts_entropy, bytes_entropy,
            mean_duration, cvd, mean_pkts, cvp, mean_bytes, cvb
        ]

        if self.MODEL in ("Catboost", "XGBoost"):
            ftf_pred = self.ftf_detector.predict([ft_features])
        else:
            ftf_pred = self._bert_predict(
                self.ftf_bert, self.ftf_bert_discretizer, self.ftf_bert_n_bins, ft_features
            )

        self.logger.debug('[FTF] dpid=%s %s=%s', dpid, self.MODEL, ftf_pred)

        if ftf_pred:
            self.logger.info('[ANOMALY] Detected FTF attack for %s', dpid)
            with open(self.REPORT_PATH, "a", newline="") as f:
                f.write(f"""
                ANOMALY@{dpid}, timestamp {time()}
                    FT FEATURES: {ft_features}
                """)
        else:
            return

        # Get Flow Rule Features
        i = 0; n = len(stats)
        for stat in stats:
            duration, packet_count, byte_count, pkt_rate, byte_rate, psi = self.feature_ext[dpid].get_flow_feature(stat)
            fr_features = [
                duration, packet_count, byte_count, pkt_rate, byte_rate, psi
            ]

            if self.MODEL in ("Catboost", "XGBoost"):
                fr_pred = self.mfr_detector.predict([fr_features])
            else:
                fr_pred = self._bert_predict(
                    self.mfr_bert, self.mfr_bert_discretizer, self.mfr_bert_n_bins, fr_features
                )

            if fr_pred:
                with open(self.REPORT_PATH, "a", newline="") as f:
                    f.write(f"""
                        FR FEATURES: {fr_features}
                        FLOW RULE {i}/{n}: {stat}
                    
                    """)
            i+=1
