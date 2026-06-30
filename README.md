# **README**

Ruy-manager (v4.34) issue:

```text
ImportError: cannot import name 'ALREADY_HANDLED' from 'eventlet.wsgi' (/usr/local/lib/python3.8/dist-packages/eventlet/wsgi.py)
```

Downgrade:

```bash
$ pip install eventlet==0.30.2
```

## Network, Monitoring and Flow Table Flooding

```bash
# Controller
$ ryu-manager --observe-links simple_switch_14_mod.py 

# Topology
$ python3 network.py
```

## Flow Table Flooding

```text
mininet> py FloodAttack(switch=s2, iface=s2.intfList()[2], npacket=100, threads=4, bsize=50, bdelay=1).start()
```

---=
