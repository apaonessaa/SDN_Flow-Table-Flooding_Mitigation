#!/usr/bin/env python3

"""
python3 split_pcap_sessions.py \
    --input Monday-WorkingHours.pcap \
    --outdir ./scenarios \
    --n-segments 12
"""

import argparse
import subprocess
import shutil
import sys
import os
import re
from datetime import datetime, timedelta

def get_pcap_time_bounds(pcap_path):
    result = subprocess.run(
        ["capinfos", "-a", "-e", "-S", pcap_path],
        capture_output=True, text=True, check=True
    )
    out = result.stdout
    start_match = re.search(r"First packet time.*?(\d+\.\d+)", out) or \
                  re.search(r"First packet.*?:\s*(\d+\.\d+)", out)
    end_match = re.search(r"Last packet time.*?(\d+\.\d+)", out) or \
                re.search(r"Last packet.*?:\s*(\d+\.\d+)", out)
    start_ts = float(start_match.group(1))
    end_ts = float(end_match.group(1))
    return start_ts, end_ts

def split_by_time(pcap_path, outdir, start_ts, end_ts, n_segments=None, prefix="scenario"):
    os.makedirs(outdir, exist_ok=True)

    total_duration = end_ts - start_ts
    if total_duration <= 0:
        sys.exit("Errore: durata del pcap non valida (<= 0 secondi).")

    if n_segments:
        segment_seconds = total_duration / n_segments
    else:
        sys.exit("Specifica --n-segments")

    print(f"Durata totale pcap: {total_duration:.1f}s "
          f"({total_duration/60:.1f} minuti)")
    print(f"Numero di segmenti: {n_segments}, "
          f"durata per segmento: {segment_seconds:.1f}s "
          f"({segment_seconds/60:.1f} minuti)")

    manifest = []

    for i in range(n_segments):
        seg_start = start_ts + i * segment_seconds
        seg_end = min(start_ts + (i + 1) * segment_seconds, end_ts)
        if seg_start >= end_ts:
            break

        out_file = os.path.join(outdir, f"{prefix}_{i:03d}.pcap")

        # editcap -A / -B accettano timestamp assoluti in formato
        # "YYYY-MM-DD HH:MM:SS" oppure epoch con -a/-b a seconda della versione.
        # Usiamo il formato con secondi relativi tramite -r (record range) NON
        # è supportato per tempo; usiamo quindi i timestamp assoluti leggibili.
        start_str = datetime.utcfromtimestamp(seg_start).strftime("%Y-%m-%d %H:%M:%S")
        end_str = datetime.utcfromtimestamp(seg_end).strftime("%Y-%m-%d %H:%M:%S")

        cmd = [
            "editcap",
            "-A", start_str,
            "-B", end_str,
            pcap_path,
            out_file
        ]
        print(f"[{i+1}/{n_segments}] {start_str} -> {end_str}  =>  {out_file}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ATTENZIONE: editcap ha fallito per il segmento {i}: {e.stderr}")
            continue

        if os.path.exists(out_file) and os.path.getsize(out_file) > 24:
            # verifica numero pacchetti nel segmento
            info = subprocess.run(
                ["capinfos", "-c", out_file], capture_output=True, text=True
            )
            pkt_count = None
            m = re.search(r"Number of packets.*?(\d+)", info.stdout)
            if m:
                pkt_count = int(m.group(1))
            manifest.append({
                "group_id": i,
                "file": out_file,
                "start": start_str,
                "end": end_str,
                "packet_count": pkt_count
            })
            print(f"  OK - {pkt_count} pacchetti")
        else:
            print(f"  Segmento vuoto o non generato, salto.")
    return manifest

def write_manifest(manifest, outdir):
    import csv
    manifest_path = os.path.join(outdir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "file", "start", "end", "packet_count"])
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)
    print(f"\nManifest scritto in: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path al pcap Monday")
    parser.add_argument("--outdir", required=True, help="Directory di output per i segmenti")
    parser.add_argument("--n-segments", type=int, default=None,
                         help="Numero di segmenti da generare (durata calcolata automaticamente)")
    parser.add_argument("--prefix", default="scenario", help="Prefisso dei file generati")
    args = parser.parse_args()

    if not args.n_segments:
        sys.exit("Devi specificare --n-segments")

    if not os.path.exists(args.input):
        sys.exit(f"File non trovato: {args.input}")

    print(f"Analisi timestamp di {args.input}...")
    start_ts, end_ts = get_pcap_time_bounds(args.input)
    print(f"Primo pacchetto: {datetime.utcfromtimestamp(start_ts)} UTC")
    print(f"Ultimo pacchetto: {datetime.utcfromtimestamp(end_ts)} UTC\n")

    manifest = split_by_time(
        args.input, args.outdir, start_ts, end_ts,
        n_segments=args.n_segments,
        prefix=args.prefix
    )

    write_manifest(manifest, args.outdir)
    print(f"\nCompletato: {len(manifest)} segmenti generati in {args.outdir}")

if __name__ == "__main__":
    main()
