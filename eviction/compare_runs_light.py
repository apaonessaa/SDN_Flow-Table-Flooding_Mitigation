#!/usr/bin/env python3
"""
compare_runs_light.py - confronto sintetico baseline vs FTFM sulle run di test_online.py.

Versione ridotta di compare_runs.py: poche metriche invece di ~45, scelte per
raccontare in modo leggibile il punto di forza principale di FTFM (detection/eviction
e Packet-In Blocker) e il suo costo (latenza, CPU, RAM), senza la copertura
esaustiva ma anche senza il rumore statistico di troppe metriche correlate.

Riusa le utility di lettura/fasi di analyze_run.py.

Uso:
    python3 compare_runs_light.py --session logs/session_<id>.json
    python3 compare_runs_light.py logs/ftfm:<id> logs/baseline:<id> ...
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import analyze_run as ar

NAN = float("nan")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _opt(d, name):
    return ar._read(os.path.join(d, name), False)


def tw_mean(iv):
    """Media pesata per durata su una lista di intervalli (s, e, valore)."""
    dur = sum(e - s for s, e, _ in iv)
    return sum((e - s) * x for s, e, x in iv) / dur if dur > 0 else NAN


def cum_diff(df, col, t0, t1):
    """Incremento di un contatore cumulativo tra t0 e t1 (interpolazione lineare)."""
    if df is None or df.empty or t1 <= t0:
        return NAN
    d = df.sort_values("ts")
    ts, v = d["ts"].values, d[col].astype(float).values
    return float(np.interp(t1, ts, v) - np.interp(t0, ts, v))


# ---------------------------------------------------------------------------
# Blocchi di metriche (solo l'essenziale)
# ---------------------------------------------------------------------------
def occupancy_metrics(flows, victim, size, ph, a, out, warns):
    """Efficacia sulla tabella + sopravvivenza delle flow valide/non valide."""
    occ = ar.occupancy(flows, victim, size)  # 'occ' = regole / dimensione tabella
    if occ.empty:
        warns.append("nessun campione di flow per lo switch vittima")
        return
    f0, f1 = ph["flood"]
    iv = ar.hold_intervals(occ, "occ", f0, f1)
    if not iv:
        warns.append("nessun campione di occupazione nel flood")
        return

    t_sat = next((s for s, e, x in iv if x >= a.thr), None)
    if t_sat is None:
        out["occ_above80_frac_post_sat"] = 0.0
        out["occ_full_frac_post_sat"] = 0.0
        warns.append("la tabella non ha mai raggiunto la soglia di saturazione nel flood")
    else:
        post = [(max(s, t_sat), e, x) for s, e, x in iv if e > t_sat]
        pdur = sum(e - s for s, e, _ in post)
        out["occ_above80_frac_post_sat"] = (sum(e - s for s, e, x in post if x >= a.thr) / pdur
                                             if pdur else NAN)
        out["occ_full_frac_post_sat"] = (sum(e - s for s, e, x in post if x >= 0.99) / pdur
                                          if pdur else NAN)

    d = flows[(flows["switch"] == victim) & (flows["flows_total"] >= 0)].sort_values("ts")
    val = pd.DataFrame({
        "ts": d["ts"].values,
        "v": d["flows_valid"].values.astype(float),
        "nv": (d["flows_total"].values - d["flows_valid"].values).astype(float),
    })
    ivv_pre = ar.hold_intervals(val, "v", *ph["pre"])
    ivv_fl = ar.hold_intervals(val, "v", f0, f1)
    ivn_fl = ar.hold_intervals(val, "nv", f0, f1)
    v_pre = tw_mean(ivv_pre) if ivv_pre else NAN
    v_fl = tw_mean(ivv_fl) if ivv_fl else NAN
    out["valid_flow_survival_ratio"] = (v_fl / v_pre) if v_pre and v_pre > 0 else NAN
    out["invalid_flow_mean_flood"] = tw_mean(ivn_fl) if ivn_fl else NAN


def latency_metrics(lat, ph, a, out, warns):
    """Latenza ping, vittima vs controllo, nel flood.
    p95 e mediana sono calcolati solo sui pacchetti ricevuti; le perdite sono una metrica a parte,
    cosi' una perdita non trasforma il p95 in inf e non fa sparire la coppia dal confronto."""
    if lat is None:
        warns.append("manca la latenza dei ping")
        return
    lat = lat[lat["src"] != lat["dst"]].copy()
    cache = {}

    def grp(s, d):
        if (s, d) not in cache:
            t = ar.touches_victim(s, d)
            cache[(s, d)] = None if t is None else ("victim" if t else "control")
        return cache[(s, d)]

    lat["grp"] = [grp(s, d) for s, d in zip(lat["src"], lat["dst"])]
    f0, f1 = ph["flood"]
    for g in ("victim", "control"):
        sub = lat[(lat.grp == g) & (lat.ts >= f0) & (lat.ts < f1)]
        st = ar.lat_stats(sub, a.slow_ms)
        rx = pd.to_numeric(sub["rtt_ms"], errors="coerce").dropna().values
        out[f"ping_{g}_n"] = st["n"]
        out[f"ping_{g}_p95_ms"] = ar.quantile_inf(rx, .95) if len(rx) else NAN
        out[f"ping_{g}_p50_ms"] = st["p50_rx"]
        out[f"ping_{g}_loss_pct"] = st["loss"]
        out[f"ping_{g}_slow_loss_pct"] = st["slow"]
        if st["n"] < 30:
            warns.append(f"pochi ping {g} nel flood (n={st['n']})")

    vp, cp = out.get("ping_victim_p95_ms", NAN), out.get("ping_control_p95_ms", NAN)
    out["ping_extra_p95_ms"] = (vp - cp) if not (math.isnan(vp) or math.isnan(cp)) else NAN
    vl, cl = out.get("ping_victim_loss_pct", NAN), out.get("ping_control_loss_pct", NAN)
    out["ping_extra_loss_pct"] = (vl - cl) if not (math.isnan(vl) or math.isnan(cl)) else NAN


def resource_metrics(res, acct, ph, out, warns):
    """CPU e RAM di Ryu nel flood."""
    f0, f1 = ph["flood"]
    if res is not None:
        r = res[(res["name"] == "ryu") & (res.ts >= f0) & (res.ts < f1)]["cpu_pct"].dropna()
        out["ryu_cpu_pct_mean_flood"] = float(r.mean()) if len(r) else NAN
        rss = _num(res[res["name"] == "ryu"].copy(), ["rss_mb"])
        out["ryu_ram_max_mb"] = float(rss["rss_mb"].max()) if len(rss) else NAN
    else:
        warns.append("manca resources_<id>.csv")

    if acct is not None:
        d = acct[acct["name"] == "ryu"]
        dur = f1 - f0
        cpu_s = cum_diff(d, "user_s", f0, f1) + cum_diff(d, "sys_s", f0, f1)
        out["ryu_cpu_cores_mean_flood"] = cpu_s / dur if dur > 0 else NAN
    else:
        warns.append("manca cpuacct_<id>.csv")


def detection_metrics(rep, ph, victim, out, warns):
    """Tasso di detection (benigna vs flood, per round) ed evizioni/s nel flood."""
    need = {"timestamp", "dpid", "ftf_detected"}
    if rep is None or not need <= set(rep.columns):
        warns.append("manca il report FTFM")
        return
    vd = f"{int(victim[1:]):016x}"
    for name in ("benign", "flood"):
        t0, t1 = ph[name]
        r = rep[(rep.timestamp >= t0) & (rep.timestamp < t1) & (rep.dpid == vd)]
        det = int((r.ftf_detected == 1).sum())
        out[f"detect_rate_{name}"] = det / len(r) if len(r) else NAN
    fb = out.get("detect_rate_benign", NAN)
    if fb and not math.isnan(fb) and fb > 0.1:
        warns.append(f"tasso di detection in fase benigna alto ({fb:.2f}): possibili falsi positivi")

    if "n_eviction_msgs_total" in rep.columns:
        t0, t1 = ph["flood"]
        r = rep[(rep.timestamp >= t0) & (rep.timestamp < t1)]
        dur = max(t1 - t0, 1e-9)
        out["eviction_rate_per_s_flood"] = float(r.n_eviction_msgs_total.sum()) / dur


def blocker_metrics(rep, ph, victim, out, warns):
    """Packet-In Blocker: packet-in scartati e MAC bloccati, solo sullo switch vittima.
    n_blocked_pktin e' un contatore cumulativo (si legge per differenza), n_blocked_macs
    e' una fotografia del numero di identita' attualmente bloccate a quel round."""
    need = {"timestamp", "dpid", "n_blocked_pktin", "n_blocked_macs"}
    if rep is None or not need <= set(rep.columns):
        return
    vd = f"{int(victim[1:]):016x}"
    v = rep[rep.dpid == vd].copy().sort_values("timestamp")
    v["n_blocked_pktin"] = pd.to_numeric(v["n_blocked_pktin"], errors="coerce").fillna(0)
    v["n_blocked_macs"] = pd.to_numeric(v["n_blocked_macs"], errors="coerce").fillna(0)
    if v.empty:
        return

    def cum_at(t):
        prev = v[v.timestamp < t]
        return float(prev["n_blocked_pktin"].iloc[-1]) if len(prev) else 0.0

    b0 = ph["benign"][0]
    f0, f1 = ph["flood"]
    blocked_benign = cum_at(f0) - cum_at(b0)
    blocked_flood = cum_at(f1) - cum_at(f0)
    dur_fl = max(f1 - f0, 1e-9)
    out["blocked_pktin_rate_flood"] = blocked_flood / dur_fl

    w = v[(v.timestamp >= f0) & (v.timestamp < f1)]
    out["blocked_macs_max_flood"] = float(w["n_blocked_macs"].max()) if len(w) else 0.0

    out["blocked_pktin_benign_count"] = blocked_benign
    if blocked_benign > 0:
        warns.append(f"{int(blocked_benign)} packet-in bloccati in fase benigna: possibili falsi blocchi")
    if blocked_flood == 0:
        warns.append("blocker mai attivo nel flood (nessun packet-in scartato)")

    o = rep[rep.dpid != vd]
    if len(o):
        other = float(pd.to_numeric(o["n_blocked_pktin"], errors="coerce").fillna(0)
                       .groupby(o["dpid"]).max().sum())
        if other > 0:
            warns.append(f"{int(other)} packet-in bloccati su dpid non vittima (collaterale)")


# ---------------------------------------------------------------------------
# Una run
# ---------------------------------------------------------------------------
def analyze_one(d, rid, a):
    meta, flows, lat, ev, res, rep, atk = ar.load_run(d, rid)
    victim = meta.get("victim_switch") or f"s{ar.VICTIM_SWITCH}"
    ar.VICTIM_SWITCH = int(victim[1:])
    if meta.get("host_switch"):
        ar.HOST_SWITCH = {h: int(s[1:]) for h, s in meta["host_switch"].items()}
    size = float(meta["flow_table_size"])
    warm = meta.get("warmup")
    skip = a.skip if warm is None else max(0.0, min(a.skip, float(warm) - 25.0))

    ph, complete, rc = ar.get_phases(ev, float(flows["ts"].max()), ar.attack_stages(atk), skip)
    ph["pre"] = (ph["benign"][0], ph["flood"][0])

    warns = []
    out = {"controller": meta.get("controller"), "run_id": rid, "complete": complete,
           "victim": victim}
    if not complete:
        warns.append("run incompleta")
    if rc not in (None, 0):
        warns.append(f"worker dell'attacco con codice {rc}")

    occupancy_metrics(flows, victim, size, ph, a, out, warns)
    latency_metrics(lat, ph, a, out, warns)

    acct = _opt(d, f"cpuacct_{rid}.csv")
    if acct is not None:
        acct = _num(acct, ["ts", "user_s", "sys_s"])
    resource_metrics(res, acct, ph, out, warns)
    detection_metrics(rep, ph, victim, out, warns)
    blocker_metrics(rep, ph, victim, out, warns)

    out["avvisi"] = " | ".join(dict.fromkeys(warns)) if warns else "-"
    return out


# ---------------------------------------------------------------------------
# Le metriche: (sezione, chiave, etichetta, direzione: +1 piu'=meglio, -1 meno=meglio)
# Ordine scelto per mettere avanti i punti di forza di FTFM (detection/eviction/blocker,
# efficacia sulla tabella) e mostrare per ultimo il costo (latenza, CPU, RAM).
# ---------------------------------------------------------------------------
METRICS = [
    ("PUNTI DI FORZA - detection, eviction e blocker", None, None, None),
    ("", "detect_rate_flood", "tasso di detection nel flood (vittima)", +1),
    ("", "detect_rate_benign", "tasso di detection in fase benigna (falsi positivi)", -1),
    ("", "eviction_rate_per_s_flood", "evizioni/s nel flood (reattivita' alla saturazione)", +1),
    ("", "blocked_pktin_rate_flood", "packet-in bloccati/s nel flood (Packet-In Blocker)", +1),
    ("", "blocked_macs_max_flood", "identita' bloccate in contemporanea, max nel flood", 0),
    ("", "blocked_pktin_benign_count", "packet-in bloccati in fase benigna (falsi blocchi)", -1),
    ("EFFICACIA - stato della tabella e traffico legittimo", None, None, None),
    ("", "occ_above80_frac_post_sat", "frazione di tempo >=80% dopo la 1a saturazione", -1),
    ("", "occ_full_frac_post_sat", "frazione di tempo con tabella piena (>=99%)", -1),
    ("", "valid_flow_survival_ratio", "flow valide: rapporto flood/pre (sopravvivenza)", +1),
    ("", "invalid_flow_mean_flood", "flow non valide medie in tabella nel flood", -1),
    ("COSTO - latenza, CPU e RAM", None, None, None),
    ("", "ping_victim_p95_ms", "ping via vittima, p95 [ms] (solo ricevuti)", -1),
    ("", "ping_victim_p50_ms", "ping via vittima, mediana [ms]", -1),
    ("", "ping_victim_loss_pct", "ping via vittima, perdita [%]", -1),
    ("", "ping_control_p95_ms", "ping via controllo, p95 [ms] (riferimento)", 0),
    ("", "ping_extra_p95_ms", "ping: extra p95 ms (vittima - controllo)", -1),
    ("", "ping_extra_loss_pct", "ping: extra perdita % (vittima - controllo)", -1),
    ("", "ryu_cpu_pct_mean_flood", "CPU% media di Ryu nel flood", -1),
    ("", "ryu_cpu_cores_mean_flood", "core medi di Ryu nel flood", -1),
    ("", "ryu_ram_max_mb", "RAM massima di Ryu [MB]", -1),
]


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return NAN, NAN
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = np.median(x[idx], axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)


def f(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, (float, np.floating)) and math.isinf(x):
        return "inf"
    return f"{x:.4g}"


def summarize(df, a):
    rows, lines = [], []
    for sec, key, label, direction in METRICS:
        if key is None:
            lines.append(("H", sec))
            continue
        if key not in df.columns:
            continue
        r = {"metric": key, "label": label, "direction": direction}
        for c in ("baseline", "ftfm"):
            s = df.loc[df.controller == c, key].astype(float).dropna()
            r[f"{c}_n"] = len(s)
            r[f"{c}_median"] = s.median() if len(s) else NAN
            r[f"{c}_q1"] = s.quantile(.25) if len(s) else NAN
            r[f"{c}_q3"] = s.quantile(.75) if len(s) else NAN
        p = df.pivot_table(index="pair", columns="controller", values=key, aggfunc="first")
        r.update(pairs=0, diff_median=NAN, ci_lo=NAN, ci_hi=NAN)
        if {"ftfm", "baseline"} <= set(p.columns):
            diff = (p["ftfm"] - p["baseline"]).replace([np.inf, -np.inf], np.nan).dropna()
            r["pairs"] = len(diff)
            if len(diff):
                r["diff_median"] = float(diff.median())
                r["ci_lo"], r["ci_hi"] = boot_ci(diff.values, a.boot)
        rows.append(r)
        lines.append(("R", r))
    return pd.DataFrame(rows), lines


def render(df, lines, a):
    out = []
    n_by = df.groupby("controller").size().to_dict()
    out.append(f"RUN: {n_by}   coppie complete: {df.pair.nunique()}")
    if min(n_by.get("ftfm", 0), n_by.get("baseline", 0)) < 5:
        out.append("ATTENZIONE: meno di 5 run per controller, i numeri sono indicativi.")
    out.append("")
    hdr = (f"{'metrica':52s} {'baseline med [IQR]':>24s} {'ftfm med [IQR]':>24s} "
           f"{'Δ appaiato med [CI95]':>40s}")
    for kind, x in lines:
        if kind == "H":
            out += ["", f"== {x} ==", hdr]
            continue
        b = f"{f(x['baseline_median'])} [{f(x['baseline_q1'])}-{f(x['baseline_q3'])}]"
        t = f"{f(x['ftfm_median'])} [{f(x['ftfm_q1'])}-{f(x['ftfm_q3'])}]"
        dlt = (f"{f(x['diff_median'])} [{f(x['ci_lo'])}, {f(x['ci_hi'])}] n={x['pairs']}" if x["pairs"] else "-")
        out.append(f"{x['label'][:52]:52s} {b:>24s} {t:>24s} {dlt:>40s}")

    out += ["", "== AVVISI PER RUN =="]
    for _, r in df.sort_values(["pair", "controller"]).iterrows():
        out.append(f"  pair {r['pair']} {r['controller']:8s} {r['run_id']}: {r['avvisi']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="<cartella>:<run_id>")
    ap.add_argument("--session", help="JSON scritto da test_online.py --pairs")
    ap.add_argument("--thr", type=float, default=0.8, help="soglia di saturazione (default 0.8)")
    ap.add_argument("--skip", type=float, default=30.0, help="scarto iniziale della fase benigna [s]")
    ap.add_argument("--slow-ms", type=float, default=10.0)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="compare_light")
    a = ap.parse_args(argv)

    specs = []
    if a.session:
        with open(a.session) as fh:
            specs += json.load(fh)
    for s in a.runs:
        d, rid = s.rsplit(":", 1)
        specs.append({"dir": d, "rid": rid, "pair": None})

    rows = []
    for sp in specs:
        try:
            r = analyze_one(sp["dir"], str(sp["rid"]), a)
            r["pair"] = sp.get("pair")
            rows.append(r)
        except Exception as e:
            print(f"[ERRORE] {sp['dir']}:{sp['rid']}: {type(e).__name__}: {e}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df["pair"].isna().any():
        df["pair"] = df.groupby("controller").cumcount()
    df.to_csv(f"{a.out}_runs.csv", index=False)

    summ, lines = summarize(df, a)
    summ.to_csv(f"{a.out}_summary.csv", index=False)
    text = render(df, lines, a)
    with open(f"{a.out}_report.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nFile: {a.out}_runs.csv  {a.out}_summary.csv  {a.out}_report.txt")


if __name__ == "__main__":
    main()