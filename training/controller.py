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

from collections import namedtuple, deque, defaultdict, Counter 
from netaddr import IPAddress, IPNetwork
import numpy as np
import os
import csv

class FeatureExt():
    VALID_IP_RANGES = ['10.0.0.0/24']
    WINDOW_SIZE = 500
    
    PktInData = namedtuple('PktInData', ['payload', 'size', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac'])

    def __init__(self):
        self.window = deque(maxlen=self.WINDOW_SIZE)

    def add_pkt_in(self, payload, src_ip, dst_ip, src_mac, dst_mac):
        self.window.append(self.PktInData(payload, len(payload), src_ip, dst_ip, src_mac, dst_mac))

    """
    - Mean and CV payloads entropy
    - Source and Destination IP entropy
    - Source and Destination MAC entropy
    """
    def get_pkt_in_features(self):
        return (
            FeatureExt.calculate_mean_and_cv([FeatureExt.shannon_entropy(pktin.payload) for pktin in self.window]),
            FeatureExt.shannon_entropy([pktin.size for pktin in self.window]),
            FeatureExt.shannon_entropy([pktin.src_ip for pktin in self.window]),
            FeatureExt.shannon_entropy([pktin.dst_ip for pktin in self.window]),
            FeatureExt.shannon_entropy([pktin.src_mac for pktin in self.window]),
            FeatureExt.shannon_entropy([pktin.dst_mac for pktin in self.window]))
    
    """
    Feature Per Flow Table
        - Capacity used (%)
        - Packet Arrival Frequency
        - Cumulative Possibility Spoofed Ip
        - Duration entropy
        - Packet Count entropy
        - Byte Count entropy
        - Mean Duration and CVD
        - Mean Packets and CVP
        - Mean Bytes and CVB
    """
    def get_ft_features(self, flows, flow_table_size: int):
        durations = [FeatureExt.effective_duration(flow) for flow in flows]
        pkt_counts = [flow.packet_count for flow in flows]
        byte_counts = [flow.byte_count for flow in flows]
        return (
            (len(flows)*100)/flow_table_size,
            FeatureExt.calculate_paf(flows),
            FeatureExt.calculate_ft_psi(flows),
            FeatureExt.shannon_entropy(durations),
            FeatureExt.shannon_entropy(pkt_counts),
            FeatureExt.shannon_entropy(byte_counts),
            FeatureExt.calculate_mean_and_cv(durations),
            FeatureExt.calculate_mean_and_cv(pkt_counts),
            FeatureExt.calculate_mean_and_cv(byte_counts))

    """
    Feature Per Flow Rule
        - Duration
        - Packet Count
        - Byte Count
        - Packet Per Seconds
        - Bytes Per Seconds
        - PSI
    """
    def get_flow_feature(self, flow):
        flow_duration = FeatureExt.effective_duration(flow)
        return (
            flow_duration,
            flow.packet_count,
            flow.byte_count,
            flow.packet_count / flow_duration if flow_duration else 0,
            flow.byte_count / flow_duration if flow_duration else 0,
            FeatureExt.calculate_psi(flow))

    @staticmethod
    def calculate_paf(flows):
        if len(flows) < 2:
            return 0
        t = [FeatureExt.effective_duration(flow) for flow in flows]
        delta_t = np.diff(sorted(t))
        mean_delta_t = np.mean(delta_t)
        return 1 / mean_delta_t if mean_delta_t != 0 else 0

    @staticmethod
    def calculate_ft_psi(flows):
        total_psi = 0
        for flow in flows:
            total_psi += FeatureExt.calculate_psi(flow)
        return total_psi

    @staticmethod
    def calculate_mean_and_cv(values):
        if len(values) == 0:
            return 0.0, 0.0
        mean = np.mean(values)
        cv = np.std(values) / mean if mean != 0 else 0
        return mean, cv

    @staticmethod
    def effective_duration(flow):
        return flow.duration_sec + flow.duration_nsec / 1e9 if flow else 0

    @staticmethod
    def calculate_psi(flow):
        if flow is None:
            return 0
        src_ip = flow.match.get('ipv4_src', None)
        if src_ip is None:
            return 0
        for valid_range in FeatureExt.VALID_IP_RANGES:
            if IPAddress(src_ip) in IPNetwork(valid_range):
                return 0
        return 1

    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        n = len(data)
        if n == 0:
            return 0.0
        counts = Counter(data)
        pbs = [nb / n for nb in counts.values()]
        return -np.sum([pb * np.log2(pb) for pb in pbs])

class SimpleSwitch14(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    DATASET_COLUMNS = [
        "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
        "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
        "c_used", "paf", "ft_psi", "durations_entropy", "pkts_entropy", "bytes_entropy",
        "mean_duration", "cvd", "mean_pkts", "cvp", "mean_bytes", "cvb", 
        "group_id", "label"
    ]
    GROUP_ID = os.environ.get('GROUP_ID', 0)
    LABEL = os.environ.get('LABEL', 0) #0 buono, 1 cattivo
    DATASET_PATH = os.environ.get('DATASET', '/tmp/dataset.csv')
    FLOW_TABLE_SIZE = int(os.environ.get('FLOW_TABLE_SIZE', '50'))
    IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '10'))
    HARD_TIMEOUT = int(os.environ.get('HARD_TIMEOUT', '20'))

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch14, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.collector = hub.spawn(self._collector)
        self.feature_ext = defaultdict(FeatureExt)
        self._init_csv()

    def _init_csv(self):
        write_header = not os.path.exists(self.DATASET_PATH)
        self._csv_file = open(self.DATASET_PATH, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow(self.DATASET_COLUMNS)
            self._csv_file.flush()

    def _collector(self):
        while True:
            for datapath in list(self.datapaths.values()):
                self._request_stats(datapath)
            hub.sleep(1)

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
        stats = [stat for stat in body if stat.priority == 1]
        dpid = ev.msg.datapath.id

        self.logger.debug(f"[GAP] DPID: {dpid:016x}")

        # Get Pkt In Feature
        ((payload_entropy_mean, payload_entropy_cv), payload_size_entropy,
         src_ip_entropy, dst_ip_entropy,
         src_mac_entropy, dst_mac_entropy) = self.feature_ext[dpid].get_pkt_in_features()
        self.logger.debug(
            f"[GAP] > Payload Entropy Mean: {payload_entropy_mean}, Payload Entropy CV: {payload_entropy_cv}, "
            f"Payload Size Entropy: {payload_size_entropy}, "
            f"Src IP Entropy: {src_ip_entropy}, Dst IP Entropy: {dst_ip_entropy}, "
            f"Src MAC Entropy: {src_mac_entropy}, Dst MAC Entropy: {dst_mac_entropy}"
        )

        # Get Flow Table Feature
        (c_used, paf, ft_psi,
         durations_entropy, pkts_entropy, bytes_entropy,
         (mean_duration, cvd),
         (mean_pkts, cvp),
         (mean_bytes, cvb)) = self.feature_ext[dpid].get_ft_features(stats, self.FLOW_TABLE_SIZE)

        self.logger.debug(
            f"[GAP] > C_used: {c_used}, PAF: {paf}, PSI: {ft_psi}, Duration Entropy: {durations_entropy}, "
            f"Duration entropy: {durations_entropy}, Packet entropy: {pkts_entropy}, Bytes_entropy: {bytes_entropy}, "
            f"Mean Duration: {mean_duration}, CVD: {cvd}, "
            f"Mean Packet: {mean_pkts}, CVP: {cvp}, "
            f"Mean Bytes: {mean_bytes}, CVB: {cvb}"
        )

        # Get Features
        for stat in stats:
            duration, packet_count, byte_count, pkt_rate, byte_rate, psi = self.feature_ext[dpid].get_flow_feature(stat)
            self.logger.debug(
                f"[GAP] >> Duration: {duration}, Packet count: {packet_count}, "
                f"Byte count: {byte_count}, Packet Rate: {pkt_rate}, "
                f"Byte Rate: {byte_rate}, PSI: {psi}"
            )

        row = [
            payload_entropy_mean, payload_entropy_cv, payload_size_entropy,
            src_ip_entropy, dst_ip_entropy, src_mac_entropy, dst_mac_entropy,
            c_used, paf, ft_psi, durations_entropy, pkts_entropy, bytes_entropy,
            mean_duration, cvd, mean_pkts, cvp, mean_bytes, cvb, 
            self.GROUP_ID, self.LABEL
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()