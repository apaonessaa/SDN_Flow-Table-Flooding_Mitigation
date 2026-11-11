import os
import sys
import json
import subprocess
import socket
from time import time, sleep
from mininet.node import Host
from mininet.log import info

def load_metadata(prefix):
    with open(f"{prefix}traffic_metadata.json") as f:
        return json.load(f)

def start_controller(env_vars: dict, log_path: str = "controller.log",
                     app: str = "controller.py") -> subprocess.Popen:
    """app: file dell'app Ryu da lanciare (relativo alla cwd o assoluto)."""
    log_file = open(log_path, "a")
    full_env = os.environ.copy()
    full_env.update(env_vars)
    proc = subprocess.Popen(
        ['ryu-manager','--verbose','--observe-links', app],
        env=full_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,)
    return proc

def wait_for_controller(host: str = "127.0.0.1", port: int = 6653, timeout: float = 15.0) -> bool:
    deadline = time() + timeout
    while time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            sleep(0.5)
    return False

def stop_controller(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def run_script(path: str, env_vars: dict) -> None:
    info(f"- {path}\n")
    full_env = os.environ.copy()
    full_env.update(env_vars)
    result = subprocess.run(
        ['bash', path],
        env=full_env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        info(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Error code: {result.returncode}")

def wait_for_network(src: Host, dst: Host, timeout: int = 60) -> None:
    info(f"*** Verifica connettività {src.IP()}->{dst.IP()}\n")
    for attempt in range(timeout):
        result = src.cmd(f'ping -c 1 -W 1 {dst.IP()}')
        if '1 received' in result:
            info(f"*** Rete pronta (tentativo {attempt + 1})\n")
            return
        sleep(1)
    info("*** ATTENZIONE: rete non pronta dopo timeout\n")

def replay_worker(host, pcap_file, launch_time, duration, offset, proc_holder, stop_event):
    sleep_duration = (launch_time + offset) - time()
    if sleep_duration > 0:
        sleep(sleep_duration)
    if stop_event.is_set():
        return
    iface = host.defaultIntf().name
    if duration > 0:
        cmd = f"tcpreplay --intf1={iface} --loop=0 --duration={duration} {pcap_file}"
    else:
        cmd = f"tcpreplay --intf1={iface} --loop=0 {pcap_file}"
    print(f"[*] {host.name}: avvio a t+{offset:.3f}s -> {cmd}")
    proc = host.popen(cmd, shell=True)
    proc_holder[host.name] = proc
    proc.wait()
    print(f"[*] {host.name}: replay terminato")

def kill_replay_traffic(procs, stop_event):
    stop_event.set()
    for name, proc in procs.items():
        if proc.poll() is None:
            print(f"[*] Termino tcpreplay su {name} (pid={proc.pid})")
            proc.terminate()
    sleep(1)
    for name, proc in procs.items():
        if proc.poll() is None:
            proc.kill()
