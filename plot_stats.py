#!/usr/bin/env python3
"""
plot_stats.py
=============
Legge il CSV generato da NetworkMonitor e produce un report grafico
in un unico file PNG con pannelli separati per ogni metrica.

Utilizzo
--------
    python plot_stats.py                        # legge mininet_stats.csv
    python plot_stats.py --csv mio_file.csv
    python plot_stats.py --csv stats.csv --out report.png --dpi 150

Grafici generati
----------------
  1. Flow Table — flow count per switch nel tempo
  2. Flow Table — active / lookup / matched count per switch (se disponibili)
  3. Ping       — RTT medio per coppia nel tempo
  4. Ping       — Packet loss % per coppia nel tempo
  5. iPerf      — Banda (Mbps) per coppia nel tempo
  6. iPerf      — Jitter (ms) per coppia nel tempo
  7. iPerf      — Packet loss % per coppia nel tempo
  8. Riepilogo  — box-plot banda e RTT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np

# ---------------------------------------------------------------------------
# Palette e stile
# ---------------------------------------------------------------------------
PALETTE = [
    '#00C8FF', '#FF5F5F', '#4BDE97', '#FFB347',
    '#B48EFF', '#FF85C8', '#5FDDFF', '#FFE066',
]

BG_DARK   = '#0D1117'
BG_PANEL  = '#161B22'
BG_AXES   = '#1C2333'
GRID_COL  = '#2A3040'
TEXT_COL  = '#C9D1D9'
TITLE_COL = '#E6EDF3'
ACCENT    = '#00C8FF'


def apply_style():
    plt.rcParams.update({
        'figure.facecolor':  BG_DARK,
        'axes.facecolor':    BG_AXES,
        'axes.edgecolor':    GRID_COL,
        'axes.labelcolor':   TEXT_COL,
        'axes.titlecolor':   TITLE_COL,
        'axes.titlesize':    10,
        'axes.labelsize':    8,
        'axes.grid':         True,
        'grid.color':        GRID_COL,
        'grid.linewidth':    0.6,
        'xtick.color':       TEXT_COL,
        'ytick.color':       TEXT_COL,
        'xtick.labelsize':   7,
        'ytick.labelsize':   7,
        'legend.facecolor':  BG_PANEL,
        'legend.edgecolor':  GRID_COL,
        'legend.labelcolor': TEXT_COL,
        'legend.fontsize':   7,
        'lines.linewidth':   1.6,
        'lines.markersize':  4,
        'font.family':       'monospace',
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _fmt_xaxis(ax, timestamps):
    """Formatta l'asse X con date/ora leggibili."""
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    else:
        # Fallback: indici numerici
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))


def _no_data(ax, msg='No data'):
    ax.text(0.5, 0.5, msg,
            transform=ax.transAxes,
            ha='center', va='center',
            color=TEXT_COL, fontsize=9, alpha=0.6)
    ax.set_visible(True)


def _plot_lines(ax, df, x_col, y_col, group_col, ylabel, title, marker='o'):
    """Traccia una linea per ogni gruppo in group_col."""
    if df.empty or y_col not in df.columns:
        _no_data(ax, f'No {y_col} data')
        ax.set_title(title)
        return
    groups = df[group_col].unique()
    for i, g in enumerate(groups):
        sub = df[df[group_col] == g].sort_values(x_col)
        if sub[y_col].dropna().empty:
            continue
        ax.plot(sub[x_col], sub[y_col],
                color=_color(i), marker=marker,
                label=str(g), markevery=max(1, len(sub)//20))
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=TITLE_COL)
    ax.legend(loc='upper right')
    _fmt_xaxis(ax, df[x_col])


# ---------------------------------------------------------------------------
# Sezioni di plot
# ---------------------------------------------------------------------------

def plot_flow_count(ax, df_ft):
    _plot_lines(ax, df_ft, 'timestamp', 'flow_count', 'node',
                'Flow Count', 'Flow Table — active flows per switch')


def plot_flow_table_detail(ax, df_ft):
    """Mostra active/lookup/matched per il primo switch che li ha."""
    cols = ['active_count', 'lookup_count', 'matched_count']
    available = [c for c in cols if c in df_ft.columns and df_ft[c].notna().any()]
    if not available:
        _no_data(ax, 'No table detail data')
        ax.set_title('Flow Table — table details')
        return

    switches = df_ft['node'].unique()
    for i, sw in enumerate(switches):
        sub = df_ft[df_ft['node'] == sw].sort_values('timestamp')
        for j, col in enumerate(available):
            if sub[col].dropna().empty:
                continue
            label = f'{sw} / {col.replace("_count","")}'
            ax.plot(sub['timestamp'], sub[col],
                    color=_color(i * len(available) + j),
                    label=label, linestyle=['solid','dashed','dotted'][j % 3])
    ax.set_ylabel('Count')
    ax.set_title('Flow Table — per-table detail')
    ax.legend(loc='upper right')
    _fmt_xaxis(ax, df_ft['timestamp'])


def plot_rtt(ax, df_ping):
    _plot_lines(ax, df_ping, 'timestamp', 'rtt_avg_ms', '_pair',
                'RTT avg (ms)', 'Ping — RTT medio per coppia')


def plot_ping_loss(ax, df_ping):
    _plot_lines(ax, df_ping, 'timestamp', 'packet_loss_pct', '_pair',
                'Loss (%)', 'Ping — Packet loss %')


def plot_bandwidth(ax, df_iperf):
    _plot_lines(ax, df_iperf, 'timestamp', 'bandwidth_mbps', '_pair',
                'Mbps', 'iPerf — Banda per coppia')


def plot_jitter(ax, df_iperf):
    _plot_lines(ax, df_iperf, 'timestamp', 'jitter_ms', '_pair',
                'Jitter (ms)', 'iPerf — Jitter per coppia')


def plot_iperf_loss(ax, df_iperf):
    _plot_lines(ax, df_iperf, 'timestamp', 'loss_pct', '_pair',
                'Loss (%)', 'iPerf — Packet loss %')


def plot_boxplots(ax_bw, ax_rtt, df_iperf, df_ping):
    """Box-plot riassuntivi per banda e RTT."""
    # Banda
    if not df_iperf.empty and 'bandwidth_mbps' in df_iperf.columns:
        pairs = df_iperf['_pair'].unique()
        data_bw = [df_iperf[df_iperf['_pair'] == p]['bandwidth_mbps'].dropna().values
                   for p in pairs]
        bp = ax_bw.boxplot(data_bw, labels=pairs, patch_artist=True,
                           medianprops=dict(color=ACCENT, linewidth=2))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_bw.set_ylabel('Mbps')
        ax_bw.set_title('Banda — distribuzione per coppia')
        plt.setp(ax_bw.xaxis.get_majorticklabels(), rotation=15, ha='right')
    else:
        _no_data(ax_bw, 'No bandwidth data')
        ax_bw.set_title('Banda — distribuzione')

    # RTT
    if not df_ping.empty and 'rtt_avg_ms' in df_ping.columns:
        pairs = df_ping['_pair'].unique()
        data_rtt = [df_ping[df_ping['_pair'] == p]['rtt_avg_ms'].dropna().values
                    for p in pairs]
        bp = ax_rtt.boxplot(data_rtt, labels=pairs, patch_artist=True,
                            medianprops=dict(color='#4BDE97', linewidth=2))
        for patch, color in zip(bp['boxes'], PALETTE[2:]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_rtt.set_ylabel('RTT (ms)')
        ax_rtt.set_title('RTT — distribuzione per coppia')
        plt.setp(ax_rtt.xaxis.get_majorticklabels(), rotation=15, ha='right')
    else:
        _no_data(ax_rtt, 'No RTT data')
        ax_rtt.set_title('RTT — distribuzione')


# ---------------------------------------------------------------------------
# Caricamento CSV
# ---------------------------------------------------------------------------

def load_csv(path: str) -> tuple:
    """
    Legge il CSV e restituisce tre DataFrame separati:
    (df_flow_table, df_ping, df_iperf).

    Il monitor scrive una colonna generica 'value' come campo principale
    di ogni riga. Questa funzione la mappa alle colonne semantiche attese
    (flow_count, rtt_avg_ms, bandwidth_mbps) quando le colonne dedicate
    sono assenti o tutte NaN.
    """
    df = pd.read_csv(path, dtype=str)

    # Converti timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Converti colonna value (campo principale del monitor)
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Converti tutte le colonne numeriche specifiche
    num_cols = [
        'table_id', 'active_count', 'lookup_count', 'matched_count',
        'rtt_avg_ms', 'packet_loss_pct',
        'interval_s', 'bandwidth_mbps', 'jitter_ms', 'loss_pct',
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_ft    = df[df['type'] == 'flow_table'].copy()
    df_ping  = df[df['type'] == 'ping'].copy()
    df_iperf = df[df['type'] == 'iperf'].copy()

    # Flow table: flow_count viene scritto nel campo 'value'
    if not df_ft.empty:
        df_ft['flow_count'] = df_ft['value']

    # Ping: rtt_avg_ms può stare in 'value' se la colonna dedicata è vuota
    if not df_ping.empty:
        if 'rtt_avg_ms' not in df_ping.columns or df_ping['rtt_avg_ms'].isna().all():
            df_ping['rtt_avg_ms'] = df_ping['value']
        df_ping['_pair'] = df_ping['src'].fillna('?') + '→' + df_ping['dst'].fillna('?')

    # iPerf: bandwidth_mbps può stare in 'value' se la colonna dedicata è vuota
    if not df_iperf.empty:
        if 'bandwidth_mbps' not in df_iperf.columns or df_iperf['bandwidth_mbps'].isna().all():
            df_iperf['bandwidth_mbps'] = df_iperf['value']
        df_iperf['_pair'] = df_iperf['src'].fillna('?') + '→' + df_iperf['dst'].fillna('?')

    return df_ft, df_ping, df_iperf


# ---------------------------------------------------------------------------
# Layout e assemblaggio
# ---------------------------------------------------------------------------

def build_figure(df_ft, df_ping, df_iperf) -> plt.Figure:
    apply_style()

    fig = plt.figure(figsize=(18, 22), facecolor=BG_DARK)
    fig.suptitle(
        'Mininet Network Monitor — Statistics Report',
        fontsize=15, color=TITLE_COL, fontweight='bold', y=0.995,
    )

    gs = GridSpec(
        nrows=5, ncols=2,
        figure=fig,
        hspace=0.52,
        wspace=0.30,
        top=0.975, bottom=0.04,
        left=0.07, right=0.97,
    )

    # Riga 0 — Flow Table
    ax_fc  = fig.add_subplot(gs[0, 0])
    ax_ftd = fig.add_subplot(gs[0, 1])
    plot_flow_count(ax_fc, df_ft)
    plot_flow_table_detail(ax_ftd, df_ft)

    # Riga 1 — Ping
    ax_rtt  = fig.add_subplot(gs[1, 0])
    ax_ploss = fig.add_subplot(gs[1, 1])
    plot_rtt(ax_rtt, df_ping)
    plot_ping_loss(ax_ploss, df_ping)

    # Riga 2 — iPerf banda + jitter
    ax_bw  = fig.add_subplot(gs[2, 0])
    ax_jit = fig.add_subplot(gs[2, 1])
    plot_bandwidth(ax_bw, df_iperf)
    plot_jitter(ax_jit, df_iperf)

    # Riga 3 — iPerf loss (colonna sinistra) + vuota (colonna destra)
    ax_iloss = fig.add_subplot(gs[3, 0])
    ax_empty  = fig.add_subplot(gs[3, 1])
    plot_iperf_loss(ax_iloss, df_iperf)
    ax_empty.set_visible(False)

    # Riga 4 — Box-plot riassuntivi
    ax_box_bw  = fig.add_subplot(gs[4, 0])
    ax_box_rtt = fig.add_subplot(gs[4, 1])
    plot_boxplots(ax_box_bw, ax_box_rtt, df_iperf, df_ping)

    # Cornice decorativa per ogni axes
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    return fig


# ---------------------------------------------------------------------------
# Statistiche testuali a console
# ---------------------------------------------------------------------------

def print_summary(df_ft, df_ping, df_iperf):
    sep = '─' * 56
    print(f'\n{sep}')
    print(' Mininet Stats — Summary')
    print(sep)

    # Flow table
    if not df_ft.empty:
        print('\n[Flow Table]')
        for sw, grp in df_ft.groupby('node'):
            fc = grp['flow_count'].dropna()
            if not fc.empty:
                print(f'  {sw:6s}  min={fc.min():.0f}  avg={fc.mean():.1f}  max={fc.max():.0f}')

    # Ping
    if not df_ping.empty:
        print('\n[Ping]')
        for pair, grp in df_ping.groupby('_pair'):
            rtt = grp['rtt_avg_ms'].dropna()
            loss = grp['packet_loss_pct'].dropna()
            rtt_str  = f'rtt avg={rtt.mean():.2f} ms' if not rtt.empty else 'no rtt data'
            loss_str = f'loss avg={loss.mean():.1f}%' if not loss.empty else ''
            print(f'  {pair:14s}  {rtt_str}  {loss_str}')

    # iPerf
    if not df_iperf.empty:
        print('\n[iPerf UDP]')
        for pair, grp in df_iperf.groupby('_pair'):
            bw   = grp['bandwidth_mbps'].dropna()
            jit  = grp['jitter_ms'].dropna()
            loss = grp['loss_pct'].dropna()
            bw_str   = f'bw avg={bw.mean():.2f} Mbps' if not bw.empty else 'no bw data'
            jit_str  = f'jitter avg={jit.mean():.2f} ms' if not jit.empty else ''
            loss_str = f'loss avg={loss.mean():.1f}%' if not loss.empty else ''
            print(f'  {pair:14s}  {bw_str}  {jit_str}  {loss_str}')

    print(f'\n{sep}\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Genera grafici dal CSV di NetworkMonitor.'
    )
    p.add_argument(
        '--csv', default='mininet_stats.csv',
        help='Percorso del file CSV (default: mininet_stats.csv)',
    )
    p.add_argument(
        '--out', default='mininet_report.png',
        help='File PNG di output (default: mininet_report.png)',
    )
    p.add_argument(
        '--dpi', type=int, default=120,
        help='Risoluzione del PNG (default: 120)',
    )
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'[ERROR] File non trovato: {csv_path}', file=sys.stderr)
        sys.exit(1)

    print(f'[*] Lettura {csv_path} ...')
    df_ft, df_ping, df_iperf = load_csv(str(csv_path))
    print(f'    flow_table rows : {len(df_ft)}')
    print(f'    ping rows       : {len(df_ping)}')
    print(f'    iperf rows      : {len(df_iperf)}')

    print_summary(df_ft, df_ping, df_iperf)

    print(f'[*] Generazione grafici ...')
    fig = build_figure(df_ft, df_ping, df_iperf)

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f'[✓] Report salvato in: {out_path}')


if __name__ == '__main__':
    main()
