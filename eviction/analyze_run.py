#!/usr/bin/env python3
"""
analyze_run.py - analisi per fase delle run online (baseline o FTFM).

Uso:
    python analyze_run.py logs/ftfm:1794756082 logs/baseline:1794755441
        [--thr 0.8] [--gap 5] [--skip 30] [--slow-ms 10] [--old-ping-ts]
        [--out analysis.csv]

Ogni argomento e' "<cartella>:<run_id>". Legge (se presenti) questi file:
    meta_<id>.json, events_<id>.csv, network_<id>_flows.csv,
    network_<id>_latency.csv, resources_<id>.csv, report_<id>.txt (solo FTFM),
    ftf_<id>.csv (stage dell'attacco).

FASI (tutte ricavate da events_<id>.csv e, se c'e', da ftf_<id>.csv):
    idle    : idle_start      -> benign_start  (nessun traffico)
    benign  : benign_start+skip -> attack_start (solo traffico benigno; si scartano
              i primi --skip secondi, dove c'e' il picco di avvio dei replay)
    prelude : attack_start    -> inizio stage 2 (stima dei timeout, nessun flooding)
    flood   : inizio stage 2  -> attack_end (la tabella si sta riempiendo)
    tail    : attack_end      -> traffic_end
Senza ftf_<id>.csv "prelude" e' vuota e "flood" parte da attack_start.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

NAN = float("nan")

# --- topologia: tenere allineato a TestOnlineTopology -----------------------
HOST_SWITCH = {}
for _i in range(1, 7):
    HOST_SWITCH[f"h{_i}"] = 1      # h1-h6   su s1
for _i in range(7, 12):
    HOST_SWITCH[f"h{_i}"] = 3      # h7-h11  su s3
for _i in range(12, 16):
    HOST_SWITCH[f"h{_i}"] = 2      # h12-h15 su s2
VICTIM_SWITCH = 2                  # switch attaccato (dpid = numero dello switch)

LAT_PHASES = ("benign", "flood", "tail")
GROUP_PHASES = ("benign", "flood")


# ---------------------------------------------------------------------------
# Lettura file
# ---------------------------------------------------------------------------
def _read(path, required, **kw):
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    if not ok:
        if required:
            raise FileNotFoundError(f"file mancante o vuoto: {path}")
        return None
    try:
        df = pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        if required:
            raise
        return None
    if df.empty:
        if required:
            raise ValueError(f"nessuna riga in {path}")
        return None
    return df


def load_run(d, rid):
    p = lambda name: os.path.join(d, name)
    with open(p(f"meta_{rid}.json")) as f:
        meta = json.load(f)
    flows = _read(p(f"network_{rid}_flows.csv"), True)
    ev = _read(p(f"events_{rid}.csv"), True)
    lat = _read(p(f"network_{rid}_latency.csv"), False)
    res = _read(p(f"resources_{rid}.csv"), False)
    rep = _read(p(f"report_{rid}.txt"), False, dtype={"dpid": str})
    atk = _read(p(f"ftf_{rid}.csv"), False)

    for c in ("ts", "flows_total", "flows_valid"):
        flows[c] = pd.to_numeric(flows[c], errors="coerce")
    ev["ts"] = pd.to_numeric(ev["ts"], errors="coerce")
    if lat is not None:
        lat["ts"] = pd.to_numeric(lat["ts"], errors="coerce")
        lat["rtt_ms"] = pd.to_numeric(lat["rtt_ms"], errors="coerce")
        lat["loss_pct"] = pd.to_numeric(lat["loss_pct"], errors="coerce")
    if res is not None:
        res["ts"] = pd.to_numeric(res["ts"], errors="coerce")
        res["cpu_pct"] = pd.to_numeric(res["cpu_pct"], errors="coerce")
    if rep is not None and "timestamp" in rep.columns:
        rep["timestamp"] = pd.to_numeric(rep["timestamp"], errors="coerce")
    return meta, flows, lat, ev, res, rep, atk


# ---------------------------------------------------------------------------
# Fasi
# ---------------------------------------------------------------------------
def attack_stages(atk):
    """{stage: (ts_start, ts_end)} dal CSV scritto dal worker dell'attacco."""
    stages = {}
    if atk is None or "stage" not in atk.columns:
        return stages
    for _, r in atk.iterrows():
        try:
            stages[int(r["stage"])] = (float(r["ts_start"]), float(r["ts_end"]))
        except (ValueError, KeyError):
            pass
    return stages


def get_phases(ev, t_last, stages, skip):
    m, rc = {}, None
    for e, t in zip(ev["event"].astype(str), ev["ts"]):
        if e.startswith("attack_worker_rc="):
            try:
                rc = int(e.split("=", 1)[1])
            except ValueError:
                rc = None
        else:
            m[e] = float(t)
    for k in ("idle_start", "benign_start", "attack_start"):
        if k not in m:
            raise ValueError(f"evento mancante: {k}")
    complete = "attack_end" in m and "traffic_end" in m
    a0 = m["attack_start"]
    a1 = m.get("attack_end", t_last)
    end = m.get("traffic_end", t_last)
    ref = a0
    if 2 in stages and a0 <= stages[2][0] < a1:
        ref = stages[2][0]                       # inizio reale del flooding
    b0 = min(m["benign_start"] + skip, a0)
    ph = {"idle": (m["idle_start"], m["benign_start"]),
          "benign": (b0, a0),
          "prelude": (a0, ref),
          "flood": (ref, a1),
          "tail": (a1, end)}
    return ph, complete, rc


# ---------------------------------------------------------------------------
# Occupazione della tabella (switch vittima)
# ---------------------------------------------------------------------------
def occupancy(flows, sw, size):
    """occ = regole (senza table-miss) / dimensione tabella;
    occ_atk = regole con MAC sorgente NON reale / dimensione tabella."""
    d = flows[(flows["switch"] == sw) & (flows["flows_total"] >= 0)].sort_values("ts")
    tot = np.clip(d["flows_total"].values - 1, 0, None) / size
    atk = np.clip(d["flows_total"].values - d["flows_valid"].values, 0, None) / size
    return pd.DataFrame({"ts": d["ts"].values, "occ": tot, "occ_atk": atk})


def hold_intervals(df, col, t0, t1):
    """Ogni campione vale fino al successivo (sample-and-hold), ritagliato a [t0, t1].
    Ritorna [(inizio, fine, valore), ...]."""
    if df.empty or t1 <= t0:
        return []
    ts, v = df["ts"].values, df[col].values
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    ends = np.append(ts[1:], ts[-1] + dt)
    out = []
    for s, e, x in zip(ts, ends, v):
        s2, e2 = max(s, t0), min(e, t1)
        if e2 > s2:
            out.append((float(s2), float(e2), float(x)))
    return out


def episodes(iv, thr, gap):
    """Episodi sopra soglia; due episodi separati da <= gap secondi sono uno solo.
    Ritorna ([[inizio, fine], ...], aperto_a_fine_finestra)."""
    eps = []
    for s, e, x in iv:
        if x < thr:
            continue
        if eps and s - eps[-1][1] <= gap:
            eps[-1][1] = e
        else:
            eps.append([s, e])
    open_end = bool(iv) and iv[-1][2] >= thr
    return eps, open_end


def clear_time(eps, open_end, t_det):
    """(secondi dalla detection al rientro sotto soglia, rientrata?, detection dentro un episodio?)"""
    for i, (s, e) in enumerate(eps):
        if s <= t_det <= e:
            still_open = open_end and i == len(eps) - 1
            return (NAN if still_open else e - t_det), (not still_open), True
    return NAN, NAN, False


def occupancy_metrics(occ, args, ph, out, warns):
    f0, f1 = ph["flood"]
    iv = hold_intervals(occ, "occ", f0, f1)
    dur = f1 - f0
    if not iv or dur <= 0:
        warns.append("nessun campione di occupazione nella fase flood")
        for k in ("time_above_s", "frac_above", "occ_max", "occ_atk_max",
                  "t_to_saturation_s", "n_episodes", "longest_episode_s"):
            out[k] = NAN
        return [], False
    time_above = sum(e - s for s, e, x in iv if x >= args.thr)
    eps, open_end = episodes(iv, args.thr, args.gap)
    first = next((s for s, e, x in iv if x >= args.thr), None)
    ivt = hold_intervals(occ, "occ_atk", f0, f1)

    out["time_above_s"] = time_above
    out["frac_above"] = time_above / dur
    out["occ_max"] = max(x for _, _, x in iv)
    out["occ_atk_max"] = max((x for _, _, x in ivt), default=NAN)
    out["t_to_saturation_s"] = (first - f0) if first is not None else NAN
    out["n_episodes"] = len(eps)
    out["longest_episode_s"] = max((e - s for s, e in eps), default=0.0)
    out["above_at_flood_end"] = open_end

    if iv[0][2] >= args.thr:
        warns.append("la tabella e' gia' sopra soglia all'inizio del flood")
    d = occ[(occ.ts >= f0) & (occ.ts < f1)]["ts"].values
    gap_max = float(np.max(np.diff(d))) if len(d) > 1 else NAN
    out["flow_sample_gap_max_s"] = gap_max
    if gap_max > 5:
        warns.append(f"buco di {gap_max:.0f} s nei campioni delle flow rule")
    return eps, open_end


# ---------------------------------------------------------------------------
# Report del controller FTFM
# ---------------------------------------------------------------------------
def report_metrics(rep, ph, out, warns):
    """Ritorna il timestamp assoluto della prima detection sullo switch vittima (o NaN)."""
    need = {"timestamp", "dpid", "ftf_detected", "detection_run"}
    if not need <= set(rep.columns):
        warns.append(f"report senza colonne {sorted(need - set(rep.columns))}")
        return NAN
    vd = f"{VICTIM_SWITCH:016x}"
    f0, f1 = ph["flood"]
    pre = rep[rep.timestamp < f0]
    fl = rep[(rep.timestamp >= f0) & (rep.timestamp < f1)]
    v = fl[fl.dpid == vd]
    det = v[v.ftf_detected == 1]

    out["n_rounds_victim"] = len(v)
    out["n_detection_runs_victim"] = int(v.detection_run.sum())
    out["n_detected_victim"] = len(det)
    out["t_first_detection_s"] = (det.timestamp.min() - f0) if len(det) else NAN
    out["n_detected_other_dpid"] = int(((fl.ftf_detected == 1) & (fl.dpid != vd)).sum())
    out["n_detected_before_flood"] = int((pre.ftf_detected == 1).sum())
    for c, name in (("n_eviction_msgs_total", "evict_msgs"), ("n_malicious_rules", "rules_flagged")):
        if c in rep.columns:
            out[f"{name}_flood"] = int(fl[c].sum())
            out[f"{name}_before_flood"] = int(pre[c].sum())
    if "skipped" in rep.columns:
        out["n_skipped_victim"] = int(v.skipped.sum())
    if "flow_table_occupancy" in rep.columns and len(v):
        out["rep_occ_max_victim"] = float(v.flow_table_occupancy.max())

    # tempi per round (solo righe dello switch vittima nella fase flood)
    for c in ("round_wall_s", "prep_wall_s", "prep_cpu_s"):
        if c in v.columns and len(v):
            out[f"rep_{c}_p50"] = float(v[c].quantile(.5))
            out[f"rep_{c}_max"] = float(v[c].max())
    runs = v[v.detection_run == 1]
    for c in ("infer_wall_s", "infer_cpu_s", "mitig_wall_s", "mitig_cpu_s"):
        if c in runs.columns and len(runs):
            out[f"rep_{c}_mean_det"] = float(runs[c].mean())
            out[f"rep_{c}_max_det"] = float(runs[c].max())

    # regolarita' del polling (dovrebbe essere T = idle/k, es. 5 s)
    b0 = ph["benign"][0]
    w = rep[(rep.dpid == vd) & (rep.timestamp >= b0) & (rep.timestamp < f1)]
    dts = np.diff(np.sort(w.timestamp.values))
    if len(dts):
        out["poll_dt_p50_s"] = float(np.median(dts))
        out["poll_dt_max_s"] = float(np.max(dts))
    if len(rep) == 0:
        warns.append("report vuoto")
    return float(det.timestamp.min()) if len(det) else NAN


# ---------------------------------------------------------------------------
# Latenza
# ---------------------------------------------------------------------------
def touches_victim(src, dst):
    a, b = HOST_SWITCH.get(src), HOST_SWITCH.get(dst)
    if a is None or b is None:
        return None
    # topologia a linea s1-s2-s3: ogni coppia su switch diversi passa da s2
    return a != b or a == VICTIM_SWITCH


def quantile_inf(values, q):
    """Quantile 'higher' senza aritmetica: funziona anche con inf (pacchetti persi)."""
    a = np.sort(np.asarray(values, dtype=float))
    if a.size == 0:
        return NAN
    return float(a[int(math.ceil(q * (a.size - 1)))])


def lat_stats(d, slow_ms):
    n = len(d)
    if n == 0:
        return {"n": 0, "loss": NAN, "p50_rx": NAN, "p95": NAN, "p99": NAN, "slow": NAN}
    rtt = d["rtt_ms"].astype(float).fillna(np.inf).values      # pacchetto perso = RTT infinito
    rx = rtt[np.isfinite(rtt)]
    return {"n": n,
            "loss": 100.0 * float(np.mean(~np.isfinite(rtt))),
            "p50_rx": float(np.median(rx)) if len(rx) else NAN,
            "p95": quantile_inf(rtt, .95),
            "p99": quantile_inf(rtt, .99),
            "slow": 100.0 * float(np.mean(rtt > slow_ms))}     # persi + RTT > soglia


def latency_metrics(lat, ph, args, out, warns):
    lat = lat[lat["src"] != lat["dst"]].copy()                 # niente ping (h,h): loopback
    if args.old_ping_ts:
        lat["ts"] = lat["ts"] - lat["rtt_ms"].fillna(0.0) / 1000.0   # ricezione -> invio
    if lat.empty:
        warns.append("nessun dato di latenza")
        return
    pairs = {(s, d) for s, d in zip(lat["src"], lat["dst"])}
    grp = {}
    for s, d in pairs:
        t = touches_victim(s, d)
        grp[(s, d)] = None if t is None else ("victim" if t else "control")
    lat["grp"] = [grp[(s, d)] for s, d in zip(lat["src"], lat["dst"])]

    win = lambda df, ph_name: df[(df.ts >= ph[ph_name][0]) & (df.ts < ph[ph_name][1])]
    for name in LAT_PHASES:
        for k, val in lat_stats(win(lat, name), args.slow_ms).items():
            out[f"lat_{name}_{k}"] = val
    for g in ("victim", "control"):
        sub = lat[lat.grp == g]
        for name in GROUP_PHASES:
            st = lat_stats(win(sub, name), args.slow_ms)
            out[f"lat_{g}_{name}_n"] = st["n"]
            out[f"lat_{g}_{name}_loss"] = st["loss"]
            out[f"lat_{g}_{name}_slow"] = st["slow"]
    # variazione benign -> flood, e differenza tra coppie che passano da s2 e coppie di controllo
    for k in ("slow", "loss"):
        dv = out[f"lat_victim_flood_{k}"] - out[f"lat_victim_benign_{k}"]
        dc = out[f"lat_control_flood_{k}"] - out[f"lat_control_benign_{k}"]
        out[f"d_{k}_victim"] = dv
        out[f"d_{k}_control"] = dc
        out[f"did_{k}"] = dv - dc            # NaN se manca uno dei due gruppi
    if out["lat_benign_n"] < 30:
        warns.append("pochi pacchetti di latenza nella fase benigna: aumenta --warmup o abbassa --skip")
    if out["lat_victim_benign_n"] == 0 or out["lat_control_benign_n"] == 0:
        warns.append("manca un gruppo di coppie (victim/control): did_* non calcolabile")


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
def cpu_metrics(res, ph, out, warns):
    for proc in ("ryu", "ovs", "system"):
        for name in GROUP_PHASES:
            t0, t1 = ph[name]
            r = res[(res["name"] == proc) & (res.ts >= t0) & (res.ts < t1)]["cpu_pct"].dropna()
            out[f"cpu_{proc}_{name}_mean"] = float(r.mean()) if len(r) else NAN
            out[f"cpu_{proc}_{name}_p95"] = float(r.quantile(.95)) if len(r) else NAN
    t0, t1 = ph["flood"]
    s = res[(res["name"] == "system") & (res.ts >= t0) & (res.ts < t1)]["cpu_pct"].dropna()
    out["cpu_system_flood_ge95_pct"] = 100.0 * float((s >= 95).mean()) if len(s) else NAN
    if len(s) and out["cpu_system_flood_ge95_pct"] > 50:
        warns.append(f"VM satura: CPU di sistema >= 95% nel {out['cpu_system_flood_ge95_pct']:.0f}% "
                     "dei campioni del flood; le latenze non sono attribuibili al controller")


# ---------------------------------------------------------------------------
# Una run
# ---------------------------------------------------------------------------
def analyze(path, rid, args):
    meta, flows, lat, ev, res, rep, atk = load_run(path, rid)
    warns = []
    size = float(meta["flow_table_size"])
    stages = attack_stages(atk)
    ph, complete, rc = get_phases(ev, float(flows["ts"].max()), stages, args.skip)

    out = {"controller": meta.get("controller"), "run_id": rid, "complete": complete,
           "flow_table_size": int(size), "profile": meta.get("profile"),
           "warmup_s": meta.get("warmup"), "controller_cpus": meta.get("controller_cpus"),
           "attack_rc": rc}
    for name, (t0, t1) in ph.items():
        out[f"{name}_dur_s"] = round(max(t1 - t0, 0.0), 1)
    for k in (0, 1, 2):
        out[f"stage{k}_s"] = round(stages[k][1] - stages[k][0], 1) if k in stages else NAN

    if not complete:
        warns.append("run incompleta (manca attack_end o traffic_end)")
    if rc not in (None, 0):
        warns.append(f"il worker dell'attacco e' terminato con codice {rc}")
    if 2 not in stages:
        warns.append("manca lo stage 2 in ftf_<id>.csv: il flood parte da attack_start")
    if out["benign_dur_s"] < 20:
        warns.append("fase benigna < 20 s dopo lo scarto iniziale: confronto benign/flood debole")

    occ = occupancy(flows, f"s{VICTIM_SWITCH}", size)
    if occ.empty:
        raise ValueError(f"nessun campione valido per s{VICTIM_SWITCH}")
    eps, open_end = occupancy_metrics(occ, args, ph, out, warns)

    t_det = NAN
    if rep is not None:
        t_det = report_metrics(rep, ph, out, warns)
        if not math.isnan(t_det):
            t_clear, cleared, in_ep = clear_time(eps, open_end, t_det)
            out["t_det_to_clear_s"] = t_clear
            out["cleared_after_det"] = cleared
            out["det_in_episode"] = in_ep
            sat, tdet_rel = out.get("t_to_saturation_s", NAN), out["t_first_detection_s"]
            out["detect_delay_after_sat_s"] = tdet_rel - sat if not math.isnan(sat) else NAN

    if lat is not None:
        latency_metrics(lat, ph, args, out, warns)
    else:
        warns.append("manca network_<id>_latency.csv")
    if res is not None:
        cpu_metrics(res, ph, out, warns)
    else:
        warns.append("manca resources_<id>.csv")

    out["avvisi"] = " | ".join(warns) if warns else "-"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<cartella>:<run_id>")
    ap.add_argument("--thr", type=float, default=0.8, help="soglia di occupazione (default 0.8)")
    ap.add_argument("--gap", type=float, default=5.0,
                    help="episodi sopra soglia separati da <= GAP secondi sono uniti (default 5)")
    ap.add_argument("--skip", type=float, default=30.0,
                    help="secondi iniziali della fase benigna da scartare (default 30)")
    ap.add_argument("--slow-ms", type=float, default=10.0,
                    help="un pacchetto con RTT > SLOW_MS (o perso) e' 'lento' (default 10)")
    ap.add_argument("--old-ping-ts", action="store_true",
                    help="run vecchie: ts = istante di RICEZIONE; lo converte in istante di invio")
    ap.add_argument("--out", default="analysis.csv")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        d, rid = spec.rsplit(":", 1)
        try:
            rows.append(analyze(d, rid, args))
        except Exception as e:           # una run rotta non blocca le altre
            print(f"[ERRORE] {spec}: {type(e).__name__}: {e}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["label"] = df["controller"].astype(str) + "_" + df["run_id"].astype(str)
    df = df.set_index("label")
    df.to_csv(args.out)

    pd.set_option("display.width", 250, "display.max_rows", None, "display.max_colwidth", 120)
    print(df.drop(columns=["avvisi"]).T.round(3).to_string())
    print("\nAVVISI")
    for label, w in df["avvisi"].items():
        print(f"  {label}: {w}")
    if df.groupby("controller").size().max() > 1:
        print("\nMEDIANA PER CONTROLLER (numeri)")
        print(df.groupby("controller").median(numeric_only=True).T.round(3).to_string())


if __name__ == "__main__":
    main()