#!/usr/bin/env python3
"""python analyze_run.py logs/ftfm:1794575326 logs/baseline:1794570000"""
import json, os, sys
import numpy as np
import pandas as pd

OCC_THR = 0.8     # come FLOW_TABLE_THRESHOLD del controller
HOLD_S = 10       # secondi consecutivi sotto soglia per dire "recuperata"
VICTIM = "s2"     # = topo.victim_dpid


def load(d, rid):
    f = lambda n: os.path.join(d, n)
    meta = json.load(open(f(f"meta_{rid}.json")))
    flows = pd.read_csv(f(f"network_{rid}_flows.csv"))
    lat = pd.read_csv(f(f"network_{rid}_latency.csv"))
    ev = pd.read_csv(f(f"events_{rid}.csv"))
    res = pd.read_csv(f(f"resources_{rid}.csv"))
    rp = f(f"report_{rid}.txt")
    # dpid come stringa, altrimenti pandas toglie gli zeri iniziali
    rep = pd.read_csv(rp, dtype={"dpid": str}) if os.path.exists(rp) else None
    return meta, flows, lat, ev, res, rep


def get_phases(ev, t_last):
    m = dict(zip(ev["event"], ev["ts"]))
    end = m.get("traffic_end", t_last)
    a_end = m.get("attack_end", end)
    ph = {"idle":   (m["idle_start"], m["benign_start"]),
          "benign": (m["benign_start"], m["attack_start"]),
          "attack": (m["attack_start"], a_end),
          "tail":   (a_end, end)}
    return ph, "traffic_end" in m


def occupancy(flows, sw, size):
    d = flows[(flows.switch == sw) & (flows.flows_total >= 0)].sort_values("ts")
    # -1: toglie la table-miss, come il controller (che conta solo priority==1)
    return pd.DataFrame({"ts": d.ts.values, "occ": (d.flows_total.values - 1) / size})


def time_above(occ, thr, t0, t1):
    """Tempo sopra soglia, tenendo ogni campione costante fino al successivo."""
    d = occ[(occ.ts >= t0) & (occ.ts <= t1)]
    if len(d) < 2:
        return np.nan
    dt = np.diff(d.ts.values)
    return float(dt[d.occ.values[:-1] >= thr].sum())


def time_to_recover(occ, thr, t_from, t_attack_end, hold=HOLD_S):
    """Dalla prima saturazione dopo t_from fino a `hold` s consecutivi sotto soglia.
    Ritorna (secondi, recuperata_prima_di_attack_end)."""
    d = occ[occ.ts >= t_from]
    sat = d[d.occ >= thr]
    if sat.empty:
        return np.nan, False            # mai saturata
    t_sat = sat.ts.iloc[0]
    below = None
    for ts, o in zip(d.ts[d.ts >= t_sat], d.occ[d.ts >= t_sat]):
        if o < thr:
            below = below if below is not None else ts
            if ts - below >= hold:
                return below - t_sat, below <= t_attack_end
        else:
            below = None
    return np.nan, False                # non recuperata entro i dati


def lat_stats(lat, t0, t1):
    d = lat[(lat.ts >= t0) & (lat.ts < t1)]
    r = d.rtt_ms.dropna()
    return {"n": len(d), "loss": d.loss_pct.mean() if len(d) else np.nan,
            "p50": r.median() if len(r) else np.nan,
            "p95": r.quantile(.95) if len(r) else np.nan}


def analyze(d, rid):
    meta, flows, lat, ev, res, rep = load(d, rid)
    ph, complete = get_phases(ev, flows.ts.max())
    a0, a1 = ph["attack"]
    occ = occupancy(flows, VICTIM, meta["flow_table_size"])
    out = {"label": f"{meta['controller']}_{rid}", "complete": complete}

    for n, (t0, t1) in ph.items():
        out[f"{n}_dur_s"] = round(t1 - t0, 1)

    # --- occupazione della tabella vittima ---
    ta = time_above(occ, OCC_THR, a0, a1)
    out["time_above80_s"] = ta
    out["frac_above80"] = ta / (a1 - a0) if a1 > a0 else np.nan
    sat = occ[(occ.ts >= a0) & (occ.occ >= OCC_THR)]
    out["t_to_saturation_s"] = sat.ts.iloc[0] - a0 if len(sat) else np.nan
    out["t_recover_s"], out["recovered_in_attack"] = time_to_recover(occ, OCC_THR, a0, a1)

    # --- detection (solo FTFM: la baseline non ha il report) ---
    if rep is not None:
        vd = f"{int(VICTIM[1:]):016x}"
        att = rep[(rep.timestamp >= a0) & (rep.timestamp < a1)]
        det = att[att.ftf_detected == 1]
        vdet = det[det.dpid == vd]
        out["t_first_detection_s"] = vdet.timestamp.min() - a0 if len(vdet) else np.nan
        out["detect_delay_after_sat_s"] = (out["t_first_detection_s"] - out["t_to_saturation_s"]
                                           if len(vdet) else np.nan)
        out["n_detection_runs_victim"] = int(att[att.dpid == vd].detection_run.sum())
        out["n_detected_victim"] = len(vdet)
        out["n_detected_other_dpid"] = int((det.dpid != vd).sum())
        out["n_detected_before_attack"] = int(((rep.ftf_detected == 1) & (rep.timestamp < a0)).sum())
        out["evict_msgs_total"] = int(att.n_eviction_msgs_total.sum())

    # --- latenza per fase (senza le coppie (h,h) = loopback) ---
    lat = lat[lat.src != lat.dst]
    for n, (t0, t1) in ph.items():
        for k, v in lat_stats(lat, t0, t1).items():
            out[f"lat_{n}_{k}"] = v
    out["lat_p95_attack_minus_benign"] = out["lat_attack_p95"] - out["lat_benign_p95"]

    # --- CPU in fase di attacco ---
    for proc in ("ryu", "system"):
        r = res[(res.name == proc) & (res.ts >= a0) & (res.ts < a1)].cpu_pct
        out[f"cpu_{proc}_mean"], out[f"cpu_{proc}_p95"] = r.mean(), r.quantile(.95)
    return out


if __name__ == "__main__":
    rows = [analyze(*a.rsplit(":", 1)) for a in sys.argv[1:]]
    df = pd.DataFrame(rows).set_index("label")
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(df.T.round(3))
    df.to_csv("analysis.csv")