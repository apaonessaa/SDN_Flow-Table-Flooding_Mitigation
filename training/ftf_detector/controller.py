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

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
root = os.path.dirname(parent)
sys.path.append(parent)
sys.path.append(root)
from mitigation.feature_ext import FeatureExt

class FTFMController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    DATASET_FTF_COLUMNS = [
        "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
        "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
        "c_used", "paf", "ft_psi", "durations_entropy", "pkts_entropy", "bytes_entropy",
        "mean_duration", "cvd", "mean_pkts", "cvp", "mean_bytes", "cvb", 
        "group_id", "label"
    ]
    DATASET_MFR_COLUMNS = [
        "duration", "packet_count", "byte_count",
        "packets_per_seconds", "bytes_per_seconds", "psi",
        "group_id", "label"
    ]
    GROUP_ID = os.environ.get('GROUP_ID', 0)
    FTF_LABEL = os.environ.get('FTF_LABEL', 0) #0 buono, 1 cattivo
    DATASET_FTF_PATH = os.environ.get('DATASET_FTF_PATH', '/tmp/dataset_ftf.csv')
    DATASET_MFR_PATH = os.environ.get('DATASET_MFR_PATH', '/tmp/dataset_mfr.csv')
    FLOW_TABLE_SIZE = int(os.environ.get('FLOW_TABLE_SIZE', '50'))
    IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '10'))
    HARD_TIMEOUT = int(os.environ.get('HARD_TIMEOUT', '20'))
    K_FACTOR = int(os.environ.get('K_FACTOR', '3'))
    DPID = int(os.environ.get('DPID', '0000000000000001'), 16)
    VALID_IPS = [
        '10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4', '10.0.0.5',
        '10.0.0.6', '10.0.0.7', '10.0.0.8', '10.0.0.9', '10.0.0.10',
        '10.0.0.11', '10.0.0.12', '10.0.0.13', '10.0.0.14', #'10.0.0.15'
    ]

    def __init__(self, *args, **kwargs):
        super(FTFMController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.collector = hub.spawn(self._collector)
        self.feature_ext = defaultdict(FeatureExt)
        self._init_csv()
        self.T = float("{:.2f}".format(min(self.IDLE_TIMEOUT, self.HARD_TIMEOUT) / self.K_FACTOR))
        self.logger.debug(f"[GAP] Intervallo di tempo per la cattura delle features: {self.T}s")
        self.collector = hub.spawn(self._collector)

    def _init_csv(self):
        write_header_ftf = not os.path.exists(self.DATASET_FTF_PATH)
        self._csv_file_ftf = open(self.DATASET_FTF_PATH, "a", newline="")
        self._csv_writer_ftf = csv.writer(self._csv_file_ftf)
        if write_header_ftf:
            self._csv_writer_ftf.writerow(self.DATASET_FTF_COLUMNS)
            self._csv_file_ftf.flush()

        write_header_mfr = not os.path.exists(self.DATASET_MFR_PATH)
        self._csv_file_mfr = open(self.DATASET_MFR_PATH, "a", newline="")
        self._csv_writer_mfr = csv.writer(self._csv_file_mfr)
        if write_header_mfr:
            self._csv_writer_mfr.writerow(self.DATASET_MFR_COLUMNS)
            self._csv_file_mfr.flush()

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
                if datapath.id == self.DPID:
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

        if dpid == self.DPID:
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
        if not self.feature_ext[dpid].has_enough_data(occupancy=0.2):
            return
        hub.spawn(self._process_stats, dpid, stats) 

    def _process_stats(self, dpid, stats):
        self.logger.debug(f"[GAP] DPID: {dpid:016x}")

        # Get Pkt In Feature
        ((payload_entropy_mean, payload_entropy_cv), payload_size_entropy,
         src_ip_entropy, dst_ip_entropy,
         src_mac_entropy, dst_mac_entropy) = self.feature_ext[dpid].get_pkt_in_features()
        """
        self.logger.debug(
            f"[GAP] > Payload Entropy Mean: {payload_entropy_mean}, Payload Entropy CV: {payload_entropy_cv}, "
            f"Payload Size Entropy: {payload_size_entropy}, "
            f"Src IP Entropy: {src_ip_entropy}, Dst IP Entropy: {dst_ip_entropy}, "
            f"Src MAC Entropy: {src_mac_entropy}, Dst MAC Entropy: {dst_mac_entropy}"
        )
        """

        # Get Flow Table Feature
        (c_used, paf, ft_psi,
         durations_entropy, pkts_entropy, bytes_entropy,
         (mean_duration, cvd),
         (mean_pkts, cvp),
         (mean_bytes, cvb)) = self.feature_ext[dpid].get_ft_features(stats, self.FLOW_TABLE_SIZE)
        """
        self.logger.debug(
            f"[GAP] > C_used: {c_used}, PAF: {paf}, PSI: {ft_psi}, Duration Entropy: {durations_entropy}, "
            f"Duration entropy: {durations_entropy}, Packet entropy: {pkts_entropy}, Bytes_entropy: {bytes_entropy}, "
            f"Mean Duration: {mean_duration}, CVD: {cvd}, "
            f"Mean Packet: {mean_pkts}, CVP: {cvp}, "
            f"Mean Bytes: {mean_bytes}, CVB: {cvb}"
        )
        """

        row_ftf = [
            payload_entropy_mean, payload_entropy_cv, payload_size_entropy,
            src_ip_entropy, dst_ip_entropy, src_mac_entropy, dst_mac_entropy,
            c_used, paf, ft_psi, durations_entropy, pkts_entropy, bytes_entropy,
            mean_duration, cvd, mean_pkts, cvp, mean_bytes, cvb, 
            self.GROUP_ID, self.FTF_LABEL
        ]
        self._csv_writer_ftf.writerow(row_ftf)
        self._csv_file_ftf.flush()

        # Get Features
        if self.FTF_LABEL: # attack mode
            for stat in stats:
                duration, packet_count, byte_count, pkt_rate, byte_rate, psi = self.feature_ext[dpid].get_flow_feature(stat)
                """
                self.logger.debug(
                    f"[GAP] >> Duration: {duration}, Packet count: {packet_count}, "
                    f"Byte count: {byte_count}, Packet Rate: {pkt_rate}, "
                    f"Byte Rate: {byte_rate}, PSI: {psi}"
                )
                """
                row_mfr = [
                    duration, packet_count, byte_count, pkt_rate, byte_rate, psi,
                    self.GROUP_ID, int(stat.match.get('ipv4_src', None) not in self.VALID_IPS or stat.match.get('ipv4_dst', None) not in self.VALID_IPS)
                ]
                self._csv_writer_mfr.writerow(row_mfr)
                self._csv_file_mfr.flush()
            
