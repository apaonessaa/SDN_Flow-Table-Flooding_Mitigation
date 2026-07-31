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
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib import hub

from collections import defaultdict
from netaddr import IPAddress, IPNetwork
import numpy as np


class SimpleSwitch14(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]
    WINDOW = 10
    FLOW_TABLE_CAPACITY = 50
    FLOW_TABLE_THRESHOLD = 0.8

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch14, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_stats = defaultdict(list)
        self.datapaths = {}
        self.collector = hub.spawn(self._collector)
        self.valid_ip_ranges = ['10.0.0.0/8']

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

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions, idle_timeout=200, hard_timeout=200)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    """
    Flow Stats Handler
    """
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body

        stats = [stat for stat in body if stat.priority == 1]

        dpid = ev.msg.datapath.id

        # Check occupancy
        occupancy = len(stats) / float(self.FLOW_TABLE_CAPACITY)
        if occupancy < self.FLOW_TABLE_THRESHOLD:
            return

        self.logger.debug(f"[GAP] DPID: {dpid:016x}")

        # Get Flow Table Feature
        (paf, ft_psi,
         (mean_duration, cvd),
         (mean_pkts, cvp),
         (mean_bytes, cvb)) = self.get_ft_features(stats)

        self.logger.debug(
            f"[GAP] > PAF: {paf}, PSI: {ft_psi}, Mean Duration: {mean_duration}, "
            f"CVD: {cvd}, Mean Packet: {mean_pkts}, CVP: {cvp}, "
            f"Mean Bytes: {mean_bytes}, CVB: {cvb}"
        )

        # Get Features
        for stat in stats:
            duration, packet_count, byte_count, pkt_rate, byte_rate, psi = self.get_flow_feature(stat)
            self.logger.debug(
                f"[GAP] >> Duration: {duration}, Packet count: {packet_count}, "
                f"Byte count: {byte_count}, Packet Rate: {pkt_rate}, "
                f"Byte Rate: {byte_rate}, PSI: {psi}"
            )

    """
    Feature Per Flow Table
        - Packet Arrival Frequency
        - Cumulative Possibility Spoofed Ip
        - Mean Duration and CVD
        - Mean Packets and CVP
        - Mean Bytes and CVB
    """
    def get_ft_features(self, flows):
        return (
            self.calculate_paf(flows),
            self.calculate_ft_psi(flows),
            self.calculate_mean_and_cv([self.effective_duration(flow) for flow in flows]),
            self.calculate_mean_and_cv([flow.packet_count for flow in flows]),
            self.calculate_mean_and_cv([flow.byte_count for flow in flows]))

    def calculate_paf(self, flows):
        if len(flows) < 2:
            return 0
        t = [self.effective_duration(flow) for flow in flows]
        delta_t = np.diff(sorted(t))
        mean_delta_t = np.mean(delta_t)
        return 1 / mean_delta_t if mean_delta_t != 0 else 0

    def calculate_ft_psi(self, flows):
        total_psi = 0
        for flow in flows:
            total_psi += self.calculate_psi(flow)
        return total_psi

    """
    - Mean
    - Coeff var
    """
    def calculate_mean_and_cv(self, values):
        mean = np.mean(values)
        cv = np.std(values) / mean if mean != 0 else 0
        return mean, cv

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
        flow_duration = self.effective_duration(flow)
        return (
            flow_duration,
            flow.packet_count,
            flow.byte_count,
            flow.packet_count / flow_duration if flow_duration else 0,
            flow.byte_count / flow_duration if flow_duration else 0,
            self.calculate_psi(flow))

    def effective_duration(self, flow):
        return flow.duration_sec + flow.duration_nsec / 1e9

    def calculate_psi(self, flow):
        src_ip = flow.match.get('ipv4_src')
        if src_ip is None:
            return 0
        for valid_range in self.valid_ip_ranges:
            if IPAddress(src_ip) in IPNetwork(valid_range):
                return 0
        return 1
