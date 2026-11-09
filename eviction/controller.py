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

import time
from collections import defaultdict
from hashlib import sha256

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)
from mitigation.feature_ext import FeatureExt

class FTSnapshot():
    def __init__(self, payload_entropy_mean, payload_entropy_cv, payload_size_entropy,
        src_ip_entropy, dst_ip_entropy, src_mac_entropy, dst_mac_entropy,
        c_used, paf, ft_psi, durations_entropy, pkts_entropy, bytes_entropy,
        mean_duration, cvd, mean_pkts, cvp, mean_bytes, cvb): 

        self.payload_entropy_mean = payload_entropy_mean
        self.payload_entropy_cv = ...

class FRSnapshot():
    def __init__(self, duration, packet_count, byte_count,
        packets_per_seconds, bytes_per_seconds, psi):
        self.duration = ...

class FTFMController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    FLOW_TABLE_SIZE = int(os.environ.get('FLOW_TABLE_SIZE', '50'))
    IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '10'))
    HARD_TIMEOUT = int(os.environ.get('HARD_TIMEOUT', '20'))
    K_FACTOR = int(os.environ.get('K_FACTOR', '2'))
    W_PKTIN = int(os.environ.get('W_PKTIN', '10'))
    MAX_COLLATERAL_PORT = int(os.environ.get('MAX_COLLATERAL_PORT', '0'))
    REPORT_PATH = os.environ.get('REPORT_PATH', '/tmp/report.txt')

    FLOW_TABLE_THRESHOLD = 0.8

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}

        self.feature_ext = defaultdict(FeatureExt)
        self.w_round_per_ft = defaultdict() # { DPID: deque(FTSnapshot) }
        self.w_round_per_fr = defaultdict() # { DPID: { match_hash: deque(FRSnapshot) } }

        self.T = float("{:.2f}".format(self.IDLE_TIMEOUT / self.K_FACTOR))

        self.logger.debug(f"K: {self.K_FACTOR}, T: {self.T}s")
        self.logger.debug(f"W_pktin: {self.W_PKTIN}")

        self.collector = hub.spawn(self._collector)

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
        if not stats or not self.feature_ext[dpid].has_enough_data():
            return
        occupancy = len(stats) / float(self.FLOW_TABLE_SIZE)
        if occupancy < self.FLOW_TABLE_THRESHOLD:
            return
        self.logger.debug(f"DPID={dpid}'s flow table {occupancy}% full")
        hub.spawn(self._detection, dpid, stats)
    
    def _mitigation(self, dpid, stats):
        """
        Detection
        """
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

        features_ftf = [
            payload_entropy_mean, payload_entropy_cv, payload_size_entropy,
            src_ip_entropy, dst_ip_entropy, src_mac_entropy, dst_mac_entropy,
            c_used, paf, ft_psi, durations_entropy, pkts_entropy, bytes_entropy,
            mean_duration, cvd, mean_pkts, cvp, mean_bytes, cvb,
        ]

        # FTF prediction
        ftf_pred = 

        if ftf_pred:
            self.logger.debug(f"[!] FTF Attack Detected on DPID={dpid}")
        else:
            self.logger.debug(f"[-] No FTF Attack Detected on DPID={dpid}")
            return

        """
        Eviction
        """
        # Get Flow Rule Features
        malicious_stats = []
        legit_stats = []

    def _eviction(self, dpid, stats):
        # Get Flow Rules Features
        flows_mfr = []
        for stat in stats:
            match_hash = sha256(str(stat.match).encode()).hexdigest()
            duration, packet_count, byte_count, pkt_rate, byte_rate, psi = self.feature_ext[dpid].get_flow_feature(stat)
            flows_mfr.append((
                [match_hash, duration, packet_count, byte_count, pkt_rate, byte_rate, psi],
            ))
            # FTF prediction
            mfr_pred = 

            if mfr_pred:
                malicious_stats.append(stat)
            else:
                legit_stats.append(stat)

        if malicious_stats:
            self.logger.debug(f"[!] Detected {len(malicious_stats)} malicious flow rules on DPID={dpid}")
        else:
            self.logger.debug(f"[-] No malicious flow rules on DPID={dpid}")
            return
        
        # ---- Eviction Tier 0: group by in_port ----
        evicted_via_wildcard, remaining = self._evict_by_port(
            datapath, malicious_stats, legit_stats
        )

        # ---- Fallback Tier 1: delete strict per flow rules ----
        for stat in remaining:
            self.delete_flow(datapath, stat.match, priority=stat.priority, strict=True)

    def _evict_by_port(self, datapath, malicious_stats, legit_stats):
        by_port = defaultdict(list)
        for stat in malicious_stats:
            by_port[stat.match['in_port']].append(stat)

        legit_by_port = defaultdict(int)
        for stat in legit_stats:
            legit_by_port[stat.match['in_port']] += 1

        evicted, remaining = [], []
        for in_port, group in by_port.items():
            collateral = legit_by_port.get(in_port, 0)

            if len(group) < 2:
                # una sola regola malevola sulla porta: non conviene un
                # wildcard, il delete strict e' gia' ottimo (1 msg, 0 collateral)
                remaining.extend(group)
                continue

            if collateral <= self.MAX_COLLATERAL_PORT:
                ofproto = datapath.ofproto
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(in_port=in_port)
                self.delete_flow(datapath, match, priority=1, strict=False)

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

        return evicted, remaining
