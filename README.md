# **README**

Ruy-manager (v4.34) issue:

```text
ImportError: cannot import name 'ALREADY_HANDLED' from 'eventlet.wsgi' (/usr/local/lib/python3.8/dist-packages/eventlet/wsgi.py)
```

Downgrade:

```bash
$ pip install eventlet==0.30.2
```

## Network

```bash
# Controller
$ ryu-manager --observe-links ryu.app.simple_switch_14

# Topology
$ python3 network.py 
```

Monitoring:

```text
mininet> py h_stats(h1, h2)
mininet> py s_stats(s1)

mininet> py monitor_sdn(net=net, attack=True)
```

## Flow Table Flooding

```text
mininet> py FloodAttack(switch=s2, iface=s2.intfList()[2], npacket=100, threads=4, bsize=50, bdelay=1).start()
```

---
