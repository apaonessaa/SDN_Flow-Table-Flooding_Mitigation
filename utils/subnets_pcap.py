from scapy.all import PcapReader
from collections import Counter

subnet_counter = Counter()
ip_set = set()

with PcapReader("/home/ubuntu/Downloads/test.pcap") as pcap:
    for pkt in pcap:
        if pkt.haslayer("IP"):
            src = pkt["IP"].src
            dst = pkt["IP"].dst
            ip_set.add(src)
            ip_set.add(dst)
            for ip in (src, dst):
                octets = ip.split(".")
                subnet = f"{octets[0]}.{octets[1]}.0.0/16"
                subnet_counter[subnet] += 1

print(f"Indirizzi IP univoci: {len(ip_set)}")
print("\nTop 10 subnet /16 per frequenza:")
for subnet, count in subnet_counter.most_common(10):
    print(f"  {subnet}: {count} occorrenze")