from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from collections import Counter, defaultdict

packets = rdpcap("/home/ubuntu/Desktop/progetto/training/pcaps/bottom.pcap")

# Counters for analysis
src_ips = Counter()
dst_ips = Counter()
protocols = Counter()
total_bytes = 0

for pkt in packets:
    if IP not in pkt:
        continue

    src_ips[pkt[IP].src] += 1
    dst_ips[pkt[IP].dst] += 1
    total_bytes += len(pkt)

    # Identify protocol by IP proto number
    proto = pkt[IP].proto
    if proto == 6:
        protocols["TCP"] += 1
    elif proto == 17:
        protocols["UDP"] += 1
    elif proto == 1:
        protocols["ICMP"] += 1
    else:
        protocols[f"Other({proto})"] += 1

print(f"\n=== Traffic Statistics ===")
print(f"Total IPv4 packets: {sum(protocols.values())}")
print(f"Total bytes: {total_bytes:,}")
print(f"\nProtocol breakdown: {dict(protocols)}")

print(f"\nTop source IPs:")
for ip, count in src_ips.most_common(10000000):
    print(f"  {ip}: {count} packets")

print(f"\nTop destination IPs:")
for ip, count in dst_ips.most_common(10000000):
    print(f"  {ip}: {count} packets")

