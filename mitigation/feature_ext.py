from collections import namedtuple, deque, Counter
from netaddr import IPAddress, IPNetwork
import numpy as np

class FeatureExt():
    VALID_IP_RANGES = ['10.0.0.0/24']
    WINDOW_SIZE = 200
    
    PktInData = namedtuple('PktInData', ['payload', 'size', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac'])

    def __init__(self):
        self.window = deque(maxlen=self.WINDOW_SIZE)

    def has_enough_data(self, occupancy: float=0.8):
        """
        Check if pkt in window has enough data
        """
        return len(self.window) / float(self.WINDOW_SIZE) >= occupancy

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
            100.0*len(flows)/flow_table_size,
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
            return 0.0
        t = [FeatureExt.effective_duration(flow) for flow in flows]
        delta_t = np.diff(sorted(t))
        mean_delta_t = np.mean(delta_t)
        return 1.0 / mean_delta_t if mean_delta_t != 0 else 0.0

    @staticmethod
    def calculate_ft_psi(flows):
        total_psi = 0.0
        for flow in flows:
            total_psi += FeatureExt.calculate_psi(flow)
        return total_psi

    @staticmethod
    def calculate_mean_and_cv(values):
        if len(values) == 0:
            return 0.0, 0.0
        mean = np.mean(values)
        cv = np.std(values) / mean if mean != 0 else 0.0
        return mean, cv

    @staticmethod
    def effective_duration(flow):
        return flow.duration_sec + flow.duration_nsec / 1e9 if flow else 0.0

    @staticmethod
    def calculate_psi(flow):
        if flow is None:
            return 0.0
        src_ip = flow.match.get('ipv4_src', None)
        if src_ip is None:
            return 0.0
        for valid_range in FeatureExt.VALID_IP_RANGES:
            if IPAddress(src_ip) in IPNetwork(valid_range):
                return 0.0
        return 1.0

    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        n = len(data)
        if n == 0:
            return 0.0
        counts = Counter(data)
        pbs = [nb / n for nb in counts.values()]
        return -1.0*np.sum([pb * np.log2(pb) for pb in pbs])
