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
$ ryu-manager --observe-links ryu.app.simple_switch_stp_13

# Topology
$ python3 network.py 
```

Monitoring:

```text
mininet> py h_stats(h1, h2)
mininet> py s_stats(s1)
```

## Flow Table Flooding

```bash
(venv)(root) $ python3 flow_table_flood_v2.py --iface s2-eth1 --seed 500 --threads 8 --expand-pool --expand-interval 5 --expand-batch 50
```

---
