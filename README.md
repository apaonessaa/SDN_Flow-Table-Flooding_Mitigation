# **README**

Ruy-manager (v4.34) issue:

```text
ImportError: cannot import name 'ALREADY_HANDLED' from 'eventlet.wsgi' (/usr/local/lib/python3.8/dist-packages/eventlet/wsgi.py)
```

Downgrade:

```bash
$ pip install eventlet==0.30.2
```

## Controller

```txt
/usr/local/lib/python3.8/dist-packages/ryu/app/simple_switch_14.py
```

- [simple_switch_14](https://github.com/faucetsdn/ryu/blob/master/ryu/app/simple_switch_14.py)

## Network, Monitoring and Flow Table Flooding

```bash
# Controller
$ ryu-manager --verbose --observe-links mycontroller.py 

# Topology
$ python3 network.py
```

## Flow Table Flooding

```text
mininet> py FloodAttack(switch=s2, iface=s2.intfList()[2], npacket=100, threads=4, bsize=50, bdelay=1).start()
```

```python
def estimate_timeout_inference(idle_timeout: int, hard_timeout: int):
    res = 0
    s = 0
    for t in range(1,idle_timeout+1):
        s += t
        if s >= hard_timeout:
            res += (s+t)
            s = 0
    return res + hard_timeout
```

---

