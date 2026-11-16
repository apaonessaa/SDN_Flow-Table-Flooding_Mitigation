#!/usr/bin/env python3
"""
compare_runs.py - confronto di precisione baseline vs FTFM sulle run di test_online.py.

Uso:
    python3 compare_runs.py --session logs/session_<id>.json          # campagna (coppie)
    python3 compare_runs.py logs/ftfm:<id> logs/baseline:<id> ...      # run singole

Riusa analyze_run.py (lettura file, fasi, metriche del report FTFM) e aggiunge:
  - fasi "pre" (benigno + prelude, fino all'inizio del flooding), "flood", "tail"
  - efficacia: occupazione dopo la 1a saturazione, tempo >=80% e >=99%, oscillazioni,
    sopravvivenza delle flow valide, recupero in tail
  - prestazioni: latenza di setup di flow NUOVE (probe), ping, gruppo victim vs control
    (differenza delle differenze), traffico benigno consegnato, drop
  - consumo: core medi e CPU-secondi (cumulativi), context switch, steal, RAM
  - confondenti: carico d'attacco offerto, saturazione della VM
  - statistica sulle coppie: mediana, IQR, differenza appaiata con CI bootstrap, sign test
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import analyze_run as ar

NAN = float("nan")
PHASES = ("pre", "flood", "tail")
CPU_PHASES = ("idle", "pre", "flood", "tail")
THRS = (0.3, 0.8, 0.99)


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


def cum_diff(df, col, t0, t1):
    """Incremento di un contatore cumulativo tra t0 e t1 (interpolazione lineare)."""
    if df is None or df.empty or t1 <= t0:
        return NAN
    d = df.sort_values("ts")
    ts, v = d["ts"].values, d[col].astype(float).values
    return float(np.interp(t1, ts, v) - np.interp(t0, ts, v))


def tw_mean(iv):
    dur = sum(e - s for s, e, _ in iv)
    return sum((e - s) * x for s, e, x in iv) / dur if dur > 0 else NAN


def recover_time(iv, thr, hold):
    """Primo istante da cui la serie resta < thr per almeno `hold` s consecutivi."""
    start = None
    for s, e, x in iv:
        if x < thr:
            start = s if start is None else start
            if e - start >= hold:
                return start
        else:
            start = None
    return NAN


# ---------------------------------------------------------------------------
# Blocchi di metriche
# ---------------------------------------------------------------------------
def occupancy_block(flows, victim, size, ph, ft_thr, a, out, warns):
    occ = ar.occupancy(flows, victim, size)
    if occ.empty:
        warns.append("nessun campione di flow per lo switch vittima")
        return
    f0, f1 = ph["flood"]
    dur = f1 - f0
    iv = ar.hold_intervals(occ, "occ", f0, f1)
    if not iv or dur <= 0:
        warns.append("nessun campione di occupazione nel flood")
        return
    out["flood_occ_max"] = max(x for _, _, x in iv)
    for thr in THRS:
        k = int(round(thr * 100))
        above = sum(e - s for s, e, x in iv if x >= thr)
        first = next((s for s, e, x in iv if x >= thr), None)
        out[f"flood_above{k}_s"] = above
        out[f"flood_above{k}_frac"] = above / dur
        out[f"flood_t_to{k}_s"] = (first - f0) if first is not None else NAN

    # comportamento DOPO la prima saturazione (80%): confronto non falsato da quando satura
    t_sat = next((s for s, e, x in iv if x >= a.thr), None)
    keys = ("post_sat_dur_s", "post_sat_above80_frac", "post_sat_full_frac",
            "post_sat_mean_occ", "post_sat_n_dips", "post_sat_t_first_dip_s")
    if t_sat is None:
        for k in keys:
            out[k] = NAN
    else:
        post = [(max(s, t_sat), e, x) for s, e, x in iv if e > t_sat]
        pdur = sum(e - s for s, e, _ in post)
        out["post_sat_dur_s"] = pdur
        out["post_sat_above80_frac"] = sum(e - s for s, e, x in post if x >= a.thr) / pdur
        out["post_sat_full_frac"] = sum(e - s for s, e, x in post if x >= 0.99) / pdur
        out["post_sat_mean_occ"] = tw_mean(post)
        dips, first_dip, prev = 0, None, True
        for s, e, x in post:
            now = x >= a.thr
            if prev and not now:
                dips += 1
                first_dip = (s - t_sat) if first_dip is None else first_dip
            prev = now
        out["post_sat_n_dips"] = dips
        out["post_sat_t_first_dip_s"] = first_dip if first_dip is not None else NAN

    # riferimento senza flooding: occupazione benigna vs soglia del controller
    ivp = ar.hold_intervals(occ, "occ", *ph["pre"])
    if ivp:
        pdur = sum(e - s for s, e, _ in ivp)
        out["pre_occ_max"] = max(x for _, _, x in ivp)
        out["pre_occ_mean"] = tw_mean(ivp)
        out["pre_above_ftthr_frac"] = sum(e - s for s, e, x in ivp if x >= ft_thr) / pdur
        if out["pre_occ_max"] >= ft_thr:
            warns.append(f"la tabella benigna supera la soglia del controller ({ft_thr}): "
                         "detection/evizioni possibili senza attacco")

    # recupero a fine attacco
    t0, t1 = ph["tail"]
    ivt = ar.hold_intervals(occ, "occ", t0, t1)
    rt = recover_time(ivt, a.thr, a.hold) if ivt else NAN
    out["tail_t_recover_s"] = (rt - t0) if not math.isnan(rt) else NAN

    # flow valide (traffico legittimo) sullo switch vittima
    d = flows[(flows["switch"] == victim) & (flows["flows_total"] >= 0)].sort_values("ts")
    val = pd.DataFrame({"ts": d["ts"].values, "v": d["flows_valid"].values.astype(float),
                        "r": d["flows_valid"].values / np.maximum(d["flows_total"].values, 1)})
    ivv_pre = ar.hold_intervals(val, "v", *ph["pre"])
    ivv_fl = ar.hold_intervals(val, "v", f0, f1)
    out["valid_pre_mean"] = tw_mean(ivv_pre) if ivv_pre else NAN
    out["valid_flood_mean"] = tw_mean(ivv_fl) if ivv_fl else NAN
    out["valid_flood_min"] = min((x for _, _, x in ivv_fl), default=NAN)
    out["valid_ratio_flood_vs_pre"] = (out["valid_flood_mean"] / out["valid_pre_mean"]
                                       if out["valid_pre_mean"] and out["valid_pre_mean"] > 0 else NAN)
    rp = val[(val.ts >= ph["pre"][0]) & (val.ts < ph["pre"][1])]["r"]
    out["valid_frac_of_total_pre"] = float(rp.median()) if len(rp) else NAN
    if len(rp) and out["valid_frac_of_total_pre"] < 0.8:
        warns.append("poche flow 'valide' in fase benigna: flows_valid non affidabile "
                     "(prova --valid-match any o controlla il regex)")
    gap = float(np.max(np.diff(d["ts"].values))) if len(d) > 1 else NAN
    out["flow_sample_gap_max_s"] = gap
    if gap > 3:
        warns.append(f"buco di {gap:.0f} s nei campioni delle flow rule")


def detection_block(rep, ph, victim, out, warns):
    """Metriche di detection a TASSO (per round), cosi' fase benigna e flood sono confrontabili
    anche se hanno durate diverse. Il potere discriminante e' detrate(flood) - detrate(benigna)."""
    need = {"timestamp", "dpid", "ftf_detected"}
    if rep is None or not need <= set(rep.columns):
        return
    vd = f"{int(victim[1:]):016x}"
    f0, f1 = ph["flood"]
    for name in ("benign", "prelude", "flood", "tail"):
        t0, t1 = ph[name]
        dur = max(t1 - t0, 1e-9)
        r = rep[(rep.timestamp >= t0) & (rep.timestamp < t1)]
        for grp, sub in (("victim", r[r.dpid == vd]), ("other", r[r.dpid != vd])):
            det = int((sub.ftf_detected == 1).sum())
            out[f"rounds_{name}_{grp}"] = len(sub)
            out[f"det_{name}_{grp}"] = det
            out[f"detrate_{name}_{grp}"] = det / len(sub) if len(sub) else NAN
        if "n_eviction_msgs_total" in r.columns:
            out[f"evict_{name}"] = int(r.n_eviction_msgs_total.sum())
            out[f"evict_per_s_{name}"] = out[f"evict_{name}"] / dur
        if "n_malicious_rules" in r.columns:
            out[f"flagged_{name}"] = int(r.n_malicious_rules.sum())

    nb = out["rounds_benign_victim"] + out["rounds_benign_other"]
    db = out["det_benign_victim"] + out["det_benign_other"]
    out["fp_rate_benign_all"] = db / nb if nb else NAN
    fb, ff = out["detrate_benign_victim"], out["detrate_flood_victim"]
    out["detrate_flood_minus_benign_victim"] = ff - fb
    eb, ef = out.get("evict_per_s_benign", NAN), out.get("evict_per_s_flood", NAN)
    out["evict_rate_ratio"] = ef / eb if eb and eb > 0 else NAN

    # prima detection SOLO dentro il flood (il prelude non conta come detection dell'attacco)
    vdet = rep[(rep.dpid == vd) & (rep.ftf_detected == 1) & (rep.timestamp >= f0) & (rep.timestamp < f1)]
    out["t_first_det_from_flood_s"] = (vdet.timestamp.min() - f0) if len(vdet) else NAN
    sat = out.get("flood_t_to80_s", NAN)
    out["t_first_det_vs_sat_s"] = out["t_first_det_from_flood_s"] - sat

    if db > 0:
        warns.append(f"falsi positivi: {db} detection su {nb} round in fase benigna")
    if out["det_flood_victim"] == 0:
        warns.append("nessuna detection sulla vittima durante il flood")
    elif not math.isnan(out["detrate_flood_minus_benign_victim"]) and out["detrate_flood_minus_benign_victim"] <= 0.05:
        warns.append("la detection non discrimina: tasso nel flood ~ tasso in fase benigna")
    tmp = {}
    ar.report_metrics(rep, ph, tmp, warns)         # tempi per round, skipped, polling...
    out.update(tmp)


def phase_stats(df, ph, slow_ms, prefix, out):
    """df: ts, rtt_ms (NaN = perso), grp. Statistiche per gruppo e fase."""
    for g in ("all", "victim", "control"):
        sub = df if g == "all" else df[df["grp"] == g]
        for name in PHASES:
            t0, t1 = ph[name]
            st = ar.lat_stats(sub[(sub.ts >= t0) & (sub.ts < t1)], slow_ms)
            for k in ("n", "loss", "p50_rx", "p95", "p99", "slow"):
                out[f"{prefix}_{g}_{name}_{k}"] = st[k]
    for k in ("p95", "slow"):
        dv = out[f"{prefix}_victim_flood_{k}"] - out[f"{prefix}_victim_pre_{k}"]
        dc = out[f"{prefix}_control_flood_{k}"] - out[f"{prefix}_control_pre_{k}"]
        out[f"{prefix}_did_{k}"] = dv - dc        # effetto "extra" sui percorsi via vittima


def latency_block(lat, setup, ph, a, out, warns):
    if lat is not None:
        lat = lat[lat["src"] != lat["dst"]].copy()
        cache = {}

        def grp(s, d):
            if (s, d) not in cache:
                t = ar.touches_victim(s, d)
                cache[(s, d)] = None if t is None else ("victim" if t else "control")
            return cache[(s, d)]
        lat["grp"] = [grp(s, d) for s, d in zip(lat["src"], lat["dst"])]
        phase_stats(lat, ph, a.slow_ms, "lat", out)
    else:
        warns.append("manca la latenza dei ping")
    if setup is not None:
        setup = setup.rename(columns={"group": "grp"})
        phase_stats(setup, ph, a.slow_ms, "setup", out)
        w = setup[(setup.ts >= ph["pre"][0]) & (setup.ts < ph["flood"][1])]
        ok = w["rtt_ms"].dropna()
        out["setup_cold_frac"] = float((ok > 1.0).mean()) if len(ok) else NAN   # coinvolgono il controller
        if len(ok) and out["setup_cold_frac"] < 0.5:
            warns.append("meno del 50% dei probe di setup passa dal controller (flow tenute calde dal replay)")
    else:
        warns.append("manca setup_<id>.csv")


def cpu_block(res, acct, ph, out, warns):
    if res is not None:
        for proc in ("ryu", "ovs", "system"):
            for name in CPU_PHASES:
                t0, t1 = ph[name]
                r = res[(res["name"] == proc) & (res.ts >= t0) & (res.ts < t1)]["cpu_pct"].dropna()
                out[f"cpu_{proc}_{name}_mean"] = float(r.mean()) if len(r) else NAN
                out[f"cpu_{proc}_{name}_p95"] = float(r.quantile(.95)) if len(r) else NAN
        f0, f1 = ph["flood"]
        s = res[(res["name"] == "system") & (res.ts >= f0) & (res.ts < f1)]["cpu_pct"].dropna()
        out["sys_ge95_flood_frac"] = float((s >= 95).mean()) if len(s) else NAN
        if len(s) and out["sys_ge95_flood_frac"] > 0.5:
            warns.append(f"VM satura: CPU >= 95% nel {100 * out['sys_ge95_flood_frac']:.0f}% del flood")
        r = _num(res[res["name"] == "ryu"].copy(), ["rss_mb", "uss_mb"])
        if len(r):
            pre = r[(r.ts >= ph["pre"][0]) & (r.ts < ph["pre"][1])]["rss_mb"].mean()
            fl = r[(r.ts >= f0) & (r.ts < f1)]["rss_mb"].max()
            out["ryu_rss_max_mb"] = float(r["rss_mb"].max())
            out["ryu_rss_growth_mb"] = float(fl - pre) if not (math.isnan(pre) or math.isnan(fl)) else NAN
    else:
        warns.append("manca resources_<id>.csv")

    if acct is None:
        warns.append("manca cpuacct_<id>.csv")
        return
    for proc in ("ryu", "ovs"):
        d = acct[acct["name"] == proc]
        for name in CPU_PHASES:
            t0, t1 = ph[name]
            dur = t1 - t0
            cpu_s = cum_diff(d, "user_s", t0, t1) + cum_diff(d, "sys_s", t0, t1)
            out[f"{proc}_cpu_s_{name}"] = cpu_s
            out[f"{proc}_cores_{name}"] = cpu_s / dur if dur > 0 else NAN
            out[f"{proc}_ctx_invol_per_s_{name}"] = cum_diff(d, "ctx_invol", t0, t1) / dur if dur > 0 else NAN
    d = acct[acct["name"] == "system"]
    for name in CPU_PHASES:
        t0, t1 = ph[name]
        u, s_, i, w, st = (cum_diff(d, c, t0, t1) for c in ("user_s", "sys_s", "idle_s", "iowait_s", "steal_s"))
        tot = u + s_ + i + w + st
        out[f"sys_busy_{name}"] = 1.0 - (i + w) / tot if tot and tot > 0 else NAN
        out[f"sys_steal_{name}"] = st / tot if tot and tot > 0 else NAN
    if out.get("sys_steal_flood", 0) and out["sys_steal_flood"] > 0.05:
        warns.append(f"steal della VM {100 * out['sys_steal_flood']:.0f}% nel flood: l'hypervisor falsa la CPU")


def port_block(ports, meta, ph, out, warns):
    if ports is None:
        warns.append("manca portstats_<id>.csv")
        return
    att, dst = meta.get("attacker"), meta.get("attack_dst")
    ports = _num(ports.copy(), ["ts", "rx_packets", "rx_bytes", "tx_packets", "tx_bytes",
                                "rx_dropped", "tx_dropped"])

    def rate(sub, col, t0, t1):
        tot = 0.0
        for _, g in sub.groupby("intf"):
            tot += cum_diff(g, col, t0, t1)
        return tot / (t1 - t0) if t1 > t0 else NAN

    f0, f1 = ph["flood"]
    a_sub = ports[ports["peer"] == att]
    out["attack_rx_pps_flood"] = rate(a_sub, "rx_packets", f0, f1) if len(a_sub) else NAN
    out["attack_rx_mbps_flood"] = rate(a_sub, "rx_bytes", f0, f1) * 8 / 1e6 if len(a_sub) else NAN
    ben = ports[ports["peer"].astype(str).str.startswith("h") & ~ports["peer"].isin([att, dst])]
    for name in ("pre", "flood"):
        t0, t1 = ph[name]
        out[f"benign_rx_pps_{name}"] = rate(ben, "rx_packets", t0, t1)    # offerto dagli host benigni
        out[f"benign_tx_pps_{name}"] = rate(ben, "tx_packets", t0, t1)    # consegnato agli host benigni
    out["benign_rx_ratio"] = out["benign_rx_pps_flood"] / out["benign_rx_pps_pre"] \
        if out["benign_rx_pps_pre"] else NAN
    out["benign_tx_ratio"] = out["benign_tx_pps_flood"] / out["benign_tx_pps_pre"] \
        if out["benign_tx_pps_pre"] else NAN
    out["drops_flood"] = sum(cum_diff(g, c, f0, f1) for _, g in ports.groupby("intf")
                             for c in ("rx_dropped", "tx_dropped"))


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
    ft_thr = float(meta.get("controller_envs", {}).get("FLOW_TABLE_THRESHOLD", a.thr))
    warm = meta.get("warmup")
    skip = a.skip if warm is None else max(0.0, min(a.skip, float(warm) - 25.0))

    ph, complete, rc = ar.get_phases(ev, float(flows["ts"].max()), ar.attack_stages(atk), skip)
    ph["pre"] = (ph["benign"][0], ph["flood"][0])
    warns = []
    out = {"controller": meta.get("controller"), "run_id": rid, "complete": complete,
           "attack_rc": rc, "ft_threshold": ft_thr, "victim": victim,
           "pinning": json.dumps(meta.get("pinning_requested"))}
    for n, (t0, t1) in ph.items():
        out[f"{n}_dur_s"] = round(max(t1 - t0, 0.0), 1)
    if not complete:
        warns.append("run incompleta")
    if rc not in (None, 0):
        warns.append(f"worker dell'attacco con codice {rc}")
    if out["pre_dur_s"] < 20:
        warns.append("fase pre < 20 s: confronto con il riferimento debole")

    occupancy_block(flows, victim, size, ph, ft_thr, a, out, warns)
    detection_block(rep, ph, victim, out, warns)

    if lat is not None:
        pass
    setup = _opt(d, f"setup_{rid}.csv")
    if setup is not None:
        setup = _num(setup, ["ts", "rtt_ms", "lost"])
    latency_block(lat, setup, ph, a, out, warns)

    acct = _opt(d, f"cpuacct_{rid}.csv")
    if acct is not None:
        acct = _num(acct, ["ts", "user_s", "sys_s", "idle_s", "iowait_s", "steal_s", "ctx_vol", "ctx_invol"])
    cpu_block(res, acct, ph, out, warns)
    port_block(_opt(d, f"portstats_{rid}.csv"), meta, ph, out, warns)

    out["avvisi"] = " | ".join(dict.fromkeys(warns)) if warns else "-"
    return out


# ---------------------------------------------------------------------------
# Statistica sulle coppie
# ---------------------------------------------------------------------------
# (sezione, chiave, etichetta, direzione: -1 meno e' meglio, +1 piu' e' meglio, 0 neutro)
METRICS = [
    ("EFFICACIA - stato della tabella dello switch vittima", None, None, None),
    ("", "post_sat_above80_frac", "frazione del tempo >=80% dopo la 1a saturazione", -1),
    ("", "post_sat_full_frac", "frazione del tempo >=99% (tabella piena) dopo la saturazione", -1),
    ("", "post_sat_mean_occ", "occupazione media dopo la saturazione", -1),
    ("", "flood_above99_s", "secondi con tabella piena (>=99%) nel flood", -1),
    ("", "flood_above80_s", "secondi >=80% nel flood", -1),
    ("", "flood_t_to80_s", "secondi dal flood alla saturazione 80%", +1),
    ("", "post_sat_n_dips", "n. rientri sotto soglia dopo la saturazione", 0),
    ("", "valid_ratio_flood_vs_pre", "flow valide in flood / pre (sopravvivenza legittimo)", +1),
    ("", "tail_t_recover_s", "secondi per rientrare sotto soglia dopo la fine", -1),
    ("PRESTAZIONI - traffico legittimo", None, None, None),
    ("", "setup_victim_flood_p95", "setup flow nuove, via vittima, p95 [ms]", -1),
    ("", "setup_victim_flood_slow", "setup flow nuove, via vittima, % lente/perse", -1),
    ("", "setup_did_slow", "setup: extra % lente (vittima - controllo)", -1),
    ("", "setup_did_p95", "setup: extra p95 ms (vittima - controllo)", -1),
    ("", "setup_control_flood_slow", "setup, percorsi di controllo, % lente/perse", -1),
    ("", "lat_victim_flood_p95", "ping via vittima p95 [ms]", -1),
    ("", "lat_victim_flood_slow", "ping via vittima % lenti/persi", -1),
    ("", "lat_control_flood_slow", "ping di controllo % lenti/persi", -1),
    ("", "drops_flood", "pacchetti scartati dalle porte nel flood", -1),
    ("CONSUMO", None, None, None),
    ("", "ryu_cores_flood", "core medi di Ryu nel flood", -1),
    ("", "ryu_cores_pre", "core medi di Ryu in fase pre", -1),
    ("", "ryu_cpu_s_flood", "CPU-secondi di Ryu nel flood", -1),
    ("", "cpu_ryu_flood_p95", "CPU% Ryu p95 nel flood", -1),
    ("", "ovs_cores_flood", "core medi di OVS nel flood", -1),
    ("", "ryu_ctx_invol_per_s_flood", "context switch involontari/s di Ryu (contesa)", -1),
    ("", "ryu_rss_max_mb", "RAM massima di Ryu [MB]", -1),
    ("", "ryu_rss_growth_mb", "crescita RAM di Ryu nel flood [MB]", -1),
    ("CONFONDENTI - devono essere uguali nei due controller", None, None, None),
    ("", "attack_rx_pps_flood", "carico offerto dall'attaccante [pkt/s]", 0),
    ("", "flood_dur_s", "durata del flood [s]", 0),
    ("", "benign_rx_pps_pre", "carico benigno offerto in pre [pkt/s]", 0),
    ("", "sys_busy_flood", "CPU di sistema occupata nel flood [0-1]", 0),
    ("", "sys_steal_flood", "steal della VM nel flood [0-1]", 0),
    ("", "sys_ge95_flood_frac", "frazione campioni con sistema >=95%", 0),
    ("DETECTION - solo FTFM (tassi per round)", None, None, None),
    ("", "fp_rate_benign_all", "tasso detection in fase benigna, tutti i dpid (falsi positivi)", 0),
    ("", "detrate_benign_victim", "tasso detection fase benigna (vittima)", 0),
    ("", "detrate_prelude_victim", "tasso detection nel prelude (vittima)", 0),
    ("", "detrate_flood_victim", "tasso detection nel flood (vittima)", 0),
    ("", "detrate_flood_other", "tasso detection nel flood (altri dpid)", 0),
    ("", "detrate_flood_minus_benign_victim", "potere discriminante: flood - benigna (vittima)", 0),
    ("", "t_first_det_from_flood_s", "prima detection DENTRO il flood [s dal suo inizio]", 0),
    ("", "t_first_det_vs_sat_s", "prima detection rispetto alla saturazione 80% [s]", 0),
    ("", "evict_per_s_benign", "evizioni/s in fase benigna", 0),
    ("", "evict_per_s_flood", "evizioni/s nel flood", 0),
    ("", "evict_rate_ratio", "evizioni/s flood / benigna", 0),
    ("", "flagged_flood", "regole segnalate nel flood", 0),
    ("", "pre_occ_max", "occupazione massima benigna (vs soglia controller)", 0),
]


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return NAN, NAN
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = np.median(x[idx], axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)


def sign_p(wins, n):
    if n == 0:
        return NAN
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


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
        r = {"metric": key, "label": label}
        for c in ("baseline", "ftfm"):
            s = df.loc[df.controller == c, key].astype(float).dropna()
            r[f"{c}_n"] = len(s)
            r[f"{c}_median"] = s.median() if len(s) else NAN
            r[f"{c}_q1"] = s.quantile(.25) if len(s) else NAN
            r[f"{c}_q3"] = s.quantile(.75) if len(s) else NAN
        p = df.pivot_table(index="pair", columns="controller", values=key, aggfunc="first")
        r.update(pairs=0, diff_median=NAN, ci_lo=NAN, ci_hi=NAN, ftfm_better=NAN, n_nonzero=0, p_sign=NAN)
        if {"ftfm", "baseline"} <= set(p.columns):
            diff = (p["ftfm"] - p["baseline"]).replace([np.inf, -np.inf], np.nan).dropna()
            r["pairs"] = len(diff)
            if len(diff):
                r["diff_median"] = float(diff.median())
                r["ci_lo"], r["ci_hi"] = boot_ci(diff.values, a.boot)
                nz = diff[diff != 0]
                r["n_nonzero"] = len(nz)
                if direction:
                    wins = int((nz * direction > 0).sum())
                    r["ftfm_better"] = wins
                    r["p_sign"] = sign_p(wins, len(nz))
        r["direction"] = direction
        rows.append(r)
        lines.append(("R", r))
    return pd.DataFrame(rows), lines


def render(df, lines, a):
    out = []
    n_by = df.groupby("controller").size().to_dict()
    out.append(f"RUN: {n_by}   coppie complete: {df.pair.nunique()}")
    if min(n_by.get("ftfm", 0), n_by.get("baseline", 0)) < 5:
        out.append("ATTENZIONE: meno di 5 run per controller: CI e sign test sono indicativi, non probanti.")
    out.append("")
    hdr = f"{'metrica':58s} {'baseline med [IQR]':>26s} {'ftfm med [IQR]':>26s} {'Δ appaiato med [CI95]':>30s} {'ftfm meglio':>12s} {'p':>6s}"
    for kind, x in lines:
        if kind == "H":
            out += ["", f"== {x} ==", hdr]
            continue
        b = f"{f(x['baseline_median'])} [{f(x['baseline_q1'])}-{f(x['baseline_q3'])}]"
        t = f"{f(x['ftfm_median'])} [{f(x['ftfm_q1'])}-{f(x['ftfm_q3'])}]"
        dlt = f"{f(x['diff_median'])} [{f(x['ci_lo'])}, {f(x['ci_hi'])}]" if x["pairs"] else "-"
        w = f"{f(x['ftfm_better'])}/{x['n_nonzero']}" if x["direction"] else "-"
        out.append(f"{x['label'][:58]:58s} {b:>26s} {t:>26s} {dlt:>30s} {w:>12s} {f(x['p_sign']):>6s}")

    # controlli di equita' del confronto
    out += ["", "== CONTROLLI DI EQUITA' =="]
    for k, tol in (("attack_rx_pps_flood", 0.10), ("flood_dur_s", 0.10), ("benign_rx_pps_pre", 0.10)):
        if k in df:
            mb = df.loc[df.controller == "baseline", k].median()
            mt = df.loc[df.controller == "ftfm", k].median()
            if mb and not math.isnan(mb) and not math.isnan(mt):
                rel = (mt - mb) / mb
                out.append(f"  {k}: baseline {f(mb)} vs ftfm {f(mt)} ({100 * rel:+.1f}%) "
                           f"{'OK' if abs(rel) <= tol else '-> DIVERSO: confronto non equo su questo fattore'}")
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
    ap.add_argument("--hold", type=float, default=5.0, help="secondi sotto soglia per dire 'recuperata'")
    ap.add_argument("--slow-ms", type=float, default=10.0)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="compare")
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
            r["pos"] = sp.get("pos")
            rows.append(r)
        except Exception as e:                     # una run rotta non blocca le altre
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