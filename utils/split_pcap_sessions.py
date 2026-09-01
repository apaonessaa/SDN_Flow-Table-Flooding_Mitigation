#!/usr/bin/env python3

"""
$ capinfos -o Monday-WorkingHours.pcap
File name:           Monday-WorkingHours.pcap
Strict time order:   False

$ reordercap Monday-WorkingHours.pcap Monday-WorkingHours-sorted.pcap

$ python3 ../utils/split_pcap_sessions.py --input ./pcaps/Monday-WorkingHours-sorted.pcap --outdir ./pcaps/scenarios --segment-seconds 3600
"""

import argparse
import subprocess
import shutil
import sys
import os
import re
from datetime import datetime
import dpkt

PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": "pcap",   # little-endian, usec
    b"\xa1\xb2\xc3\xd4": "pcap",   # big-endian, usec
    b"\x4d\x3c\xb2\xa1": "pcap",   # little-endian, nsec
    b"\xa1\xb2\x3c\x4d": "pcap",   # big-endian, nsec
    b"\x0a\x0d\x0d\x0a": "pcapng",
}

def detect_format(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    fmt = PCAP_MAGICS.get(magic)
    if fmt is None:
        sys.exit(f"Errore: formato file non riconosciuto (magic={magic!r}). "
                  f"Atteso pcap classico o pcapng.")
    return fmt


def open_dpkt_reader(fileobj, fmt):
    if fmt == "pcap":
        return dpkt.pcap.Reader(fileobj)
    else:
        return dpkt.pcapng.Reader(fileobj)

def get_pcap_time_bounds(pcap_path, fmt, use_capinfos=True):
    try:
        result = subprocess.run(
            ["capinfos", "-a", "-e", "-S", pcap_path],
            capture_output=True, text=True, check=True
        )
        out = result.stdout
        start_match = re.search(r"First packet time.*?(\d+\.\d+)", out) or \
                    re.search(r"First packet.*?:\s*(\d+\.\d+)", out) or \
                    re.search(r"Earliest packet time:\s*(\d+\.\d+)", out)
        end_match = re.search(r"Last packet time.*?(\d+\.\d+)", out) or \
                    re.search(r"Last packet.*?:\s*(\d+\.\d+)", out) or \
                    re.search(r"Latest packet time:\s*(\d+\.\d+)", out)
        if start_match and end_match:
            return float(start_match.group(1)), float(end_match.group(1))
        print("Attenzione: output capinfos inatteso.")
    except subprocess.CalledProcessError:
        print("Attenzione: capinfos ha fallito.")

def split_by_time_single_pass(pcap_path, fmt, outdir, start_ts, end_ts,
                               n_segments=None, segment_seconds=None,
                               prefix="scenario"):
    os.makedirs(outdir, exist_ok=True)

    total_duration = end_ts - start_ts
    if total_duration <= 0:
        sys.exit("Errore: durata del pcap non valida (<= 0 secondi).")

    if n_segments:
        segment_seconds = total_duration / n_segments
    elif segment_seconds:
        n_segments = int(total_duration // segment_seconds) + 1
    else:
        sys.exit("Specifica --n-segments oppure --segment-seconds")

    print(f"Durata totale pcap: {total_duration:.1f}s "
          f"({total_duration/60:.1f} minuti)")
    print(f"Numero di segmenti: {n_segments}, "
          f"durata per segmento: {segment_seconds:.1f}s "
          f"({segment_seconds/60:.1f} minuti)")

    # Prepariamo i confini e i file/writer per ogni segmento, tutti aperti
    # contemporaneamente: un solo passaggio sul file di input basta per
    # smistare ogni pacchetto nel segmento giusto.
    seg_bounds = []
    seg_files = []
    seg_writers = []
    seg_counts = [0] * n_segments

    for i in range(n_segments):
        seg_start = start_ts + i * segment_seconds
        seg_end = min(start_ts + (i + 1) * segment_seconds, end_ts)
        seg_bounds.append((seg_start, seg_end))
        out_file = os.path.join(outdir, f"{prefix}_{i:03d}.pcap")
        fh = open(out_file, "wb")
        writer = dpkt.pcap.Writer(fh)
        seg_files.append(out_file)
        seg_writers.append((fh, writer))

    print(f"Lettura single-pass di {pcap_path}...")
    n_read = 0
    n_out_of_range = 0
    with open(pcap_path, "rb") as f:
        reader = open_dpkt_reader(f, fmt)
        for ts, buf in reader:
            n_read += 1
            idx = int((ts - start_ts) // segment_seconds)
            if idx < 0:
                idx = 0
            elif idx >= n_segments:
                # ultimo segmento include il bordo superiore per non perdere
                # pacchetti per arrotondamento
                if ts <= end_ts + 1e-6:
                    idx = n_segments - 1
                else:
                    n_out_of_range += 1
                    continue
            _, writer = seg_writers[idx]
            writer.writepkt(buf, ts=ts)
            seg_counts[idx] += 1

            if n_read % 1_000_000 == 0:
                print(f"  ...{n_read} pacchetti letti finora")

    for fh, _ in seg_writers:
        fh.close()

    if n_out_of_range:
        print(f"Attenzione: {n_out_of_range} pacchetti con timestamp fuori dai bound, scartati.")

    manifest = []
    for i in range(n_segments):
        seg_start, seg_end = seg_bounds[i]
        start_str = datetime.utcfromtimestamp(seg_start).strftime("%Y-%m-%d %H:%M:%S")
        end_str = datetime.utcfromtimestamp(seg_end).strftime("%Y-%m-%d %H:%M:%S")
        manifest.append({
            "group_id": i,
            "file": seg_files[i],
            "start": start_str,
            "end": end_str,
            "packet_count": seg_counts[i],
        })
        print(f"[{i+1}/{n_segments}] {start_str} -> {end_str}  =>  "
              f"{seg_files[i]}  ({seg_counts[i]} pacchetti)")

    print(f"\nTotale pacchetti letti dal file di input: {n_read}")
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

    total = sum(row["packet_count"] or 0 for row in manifest)
    print(f"Totale pacchetti nei segmenti generati: {total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path al pcap Monday")
    parser.add_argument("--outdir", required=True, help="Directory di output per i segmenti")
    parser.add_argument("--n-segments", type=int, default=None,
                         help="Numero di segmenti da generare (durata calcolata automaticamente)")
    parser.add_argument("--segment-seconds", type=float, default=None,
                         help="Durata fissa per segmento in secondi (alternativa a --n-segments)")
    parser.add_argument("--prefix", default="scenario", help="Prefisso dei file generati")
    parser.add_argument("--no-capinfos", action="store_true",
                         help="Non usare capinfos per i bound temporali, usa sempre il pre-scan dpkt")
    args = parser.parse_args()

    if not args.n_segments and not args.segment_seconds:
        sys.exit("Devi specificare --n-segments oppure --segment-seconds")

    if not os.path.exists(args.input):
        sys.exit(f"File non trovato: {args.input}")

    fmt = detect_format(args.input)
    print(f"Formato rilevato: {fmt}")

    print(f"Analisi timestamp di {args.input}...")
    start_ts, end_ts = get_pcap_time_bounds(args.input, fmt, use_capinfos=not args.no_capinfos)
    print(f"Primo pacchetto: {datetime.utcfromtimestamp(start_ts)} UTC")
    print(f"Ultimo pacchetto: {datetime.utcfromtimestamp(end_ts)} UTC\n")

    manifest = split_by_time_single_pass(
        args.input, fmt, args.outdir, start_ts, end_ts,
        n_segments=args.n_segments,
        segment_seconds=args.segment_seconds,
        prefix=args.prefix
    )

    write_manifest(manifest, args.outdir)
    print(f"\nCompletato: {len(manifest)} segmenti generati in {args.outdir}")


if __name__ == "__main__":
    main()
