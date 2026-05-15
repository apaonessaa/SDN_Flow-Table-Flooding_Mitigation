#!/usr/bin/python
import time
from mininet.log import setLogLevel, info
from mininet.node import Host, OVSSwitch
import re

# ---------------------------------------------------------------------------
# Host stats
# ---------------------------------------------------------------------------

def h_stats(src: Host, dst: Host, nping: int = 100) -> None:
    raw = src.cmd(f'ping -c {nping} {dst.IP()}')
    matches = re.findall(r'time=([\d.]+)', raw)
    if matches:
        avg = sum(float(t) for t in matches) / len(matches)
        rtt = f"{avg:.2f}"
    else:
        rtt = "timeout"
    print(f"{'Source':>10} {'Destination':>10} {'AVG RTT':>16} on {nping} ping")
    print(f"{src.name:>10} {dst.name:>10} {rtt:>16}")


# ---------------------------------------------------------------------------
# Switch stats
# ---------------------------------------------------------------------------

def s_stats(sw: OVSSwitch) -> None:
    raw = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-flows {sw.name}')
    flow_count = sum(1 for line in raw.splitlines() if 'cookie=' in line)
    print(f"Switch {sw.name}: {flow_count} flow(s)")
