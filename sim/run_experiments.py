"""
Runs every simulation experiment for the paper and writes
  results/results.json   (all numbers quoted in the paper)
  ../figures/*.pdf       (all simulation figures)

Usage:  python run_experiments.py      (about 5-10 min on a laptop)
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import actuator_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)

P = dict(S.P); pv = S.param_vec(P); Ts = P["Ts"]
RES = {"params": {k: float(v) for k, v in P.items()}}
BIG = np.array([1e9, 0, 0, 1e9, 1e9, 1.0])

# model-error bounds used for calibration and Monte Carlo
EB = dict(R=0.20, lam=0.05, J=0.10, b=0.30, tc=0.30, L=0.10)


def draw_err(rng, scale=1.0):
    return np.array([1 + scale * rng.uniform(-EB[k], EB[k])
                     for k in ["R", "lam", "J", "b", "tc", "L"]])


def nominal_refs(rng, T):
    return [("sine", S.ref_sine(T, Ts, A=rng.uniform(0.5, 1.5), f=rng.uniform(0.3, 1.0))),
            ("chirp", S.ref_chirp(T, Ts, A=rng.uniform(0.2, 0.35), f1=rng.uniform(3, 4.5))),
            ("random", S.ref_random(T, Ts, rng))]


def debounced(x, nd):
    """min over a sliding window of nd samples (alarm needs nd in a row)."""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(x, nd).min(axis=1)
    return np.concatenate([np.full(nd - 1, -np.inf), w])


nd = int(P["nd"])
t0 = time.time()
# ===================================================== 1. calibration
print("calibration ...")
rng = np.random.default_rng(0)
cal = []
for i in range(6):
    e = draw_err(rng)
    for name, ref in nominal_refs(rng, 10.0):
        log, _ = S.simulate(pv, *ref, S.F_NONE, 0., 0., 0., e, BIG, S.R_NONE, 100 + i, 1)
        cal.append(log[int(0.1 / Ts):][:, [7, 18, 5, 9, 10]])
cal = np.concatenate(cal)
rv, we, iq, rm, fe = cal.T
c1 = 1.5 * EB["lam"] * P["lam"]
c2 = 1.5 * EB["R"] * P["R"]
c0 = 1.25 * np.max(rv - c1 * np.abs(we) - c2 * np.abs(iq))
thr_m = 1.25 * np.abs(rm).max()
thr_fe = 1.25 * fe.max()
THR = np.array([c0, c1, c2, thr_m, thr_fe, 1.0])
RES["thresholds"] = dict(c0=c0, c1=c1, c2=c2, thr_m=thr_m, thr_fe=thr_fe,
                         cal_minutes=6 * 3 * 10 / 60)
print("  ", RES["thresholds"])

# ===================================================== 2. false alarms (held out)
print("false alarms ...")
rng = np.random.default_rng(1)
neg_v, neg_m, neg_fe = [], [], []   # per-1s-window debounced peaks (normalized)
fa = dict(rv=0, rm=0, fe=0, any_proposed=0)
T_fa = 20.0; n_fa = 0; secs = 0.0
for i in range(60):
    e = draw_err(rng)
    for name, ref in nominal_refs(rng, T_fa):
        log, res = S.simulate(pv, *ref, S.F_NONE, 0., 0., 0., e, THR, S.R_NONE, 1000 + i, 1)
        n_fa += 1; secs += T_fa
        fa["rv"] += int(res[0] > 0); fa["rm"] += int(res[1] > 0); fa["fe"] += int(res[2] > 0)
        fa["any_proposed"] += int(res[0] > 0 or res[1] > 0)
        L = log[int(0.05 / Ts):]
        sv = debounced(L[:, 7] / L[:, 8], nd)
        sm = debounced(np.abs(L[:, 9]) / thr_m, nd)
        sf = debounced(L[:, 10] / thr_fe, nd)
        w = int(1.0 / Ts)
        for j in range(0, len(L) - w, w):
            neg_v.append(sv[j:j + w].max()); neg_m.append(sm[j:j + w].max())
            neg_fe.append(sf[j:j + w].max())
RES["false_alarms"] = dict(runs=n_fa, hours=secs / 3600, **fa)
print("  ", RES["false_alarms"])
neg_v = np.array(neg_v); neg_m = np.array(neg_m); neg_fe = np.array(neg_fe)

# ===================================================== 3. robustness to larger model error
print("robustness ...")
rob = []
for scale in [1.0, 1.5, 2.0, 3.0]:
    rng = np.random.default_rng(2)
    cnt_v = cnt_m = n = 0
    for i in range(10):
        e = draw_err(rng, scale)
        for name, ref in nominal_refs(rng, 10.0):
            _, res = S.simulate(pv, *ref, S.F_NONE, 0., 0., 0., e, THR, S.R_NONE, 2000 + i, 1)
            n += 1; cnt_v += int(res[0] > 0); cnt_m += int(res[1] > 0)
    rob.append(dict(scale=scale, runs=n, rv_fa_runs=cnt_v, rm_fa_runs=cnt_m))
    print("  ", rob[-1])
RES["robustness"] = rob

# ===================================================== 4. Monte Carlo fault trials
print("Monte Carlo fault trials ...")
FAULTS = [("freeze", S.F_FREEZE), ("offset", S.F_OFFSET), ("disconnect", S.F_DISC),
          ("phase_open", S.F_PHASE), ("collision", S.F_COLL), ("load_step", S.F_LOAD)]
TRUE = {"freeze": S.C_ENC, "offset": S.C_ENC, "disconnect": S.C_ENC,
        "phase_open": S.C_PHASE, "collision": S.C_COLL, "load_step": S.C_LOAD}
NMC = 50
mc = {}
pos = {"rv": [], "rm": [], "fe_enc": [], "fe_coll": []}
for fname, fcode in FAULTS:
    rng = np.random.default_rng(10 + fcode)
    rows = []
    for i in range(NMC):
        e = draw_err(rng)
        v = rng.uniform(0.5, 4.0) * rng.choice([-1, 1])
        tf = rng.uniform(0.25, 0.40)
        ref = S.ref_cruise(tf + 0.25, Ts, v)
        fmag = 0.0; qobs = 0.0
        if fcode == S.F_OFFSET:
            fmag = np.deg2rad(rng.uniform(20, 90))
        if fcode == S.F_LOAD:
            fmag = rng.uniform(0.2, 0.8) * rng.choice([-1, 1])
        if fcode == S.F_COLL:
            v = abs(v); ref = S.ref_cruise(tf + 0.25, Ts, v)
            k = int(tf / Ts); qobs = ref[0][k]
        # proposed (with reaction) -> class, latency
        log, res = S.simulate(pv, *ref, fcode, tf, fmag, qobs, e, THR, S.R_PROPOSED, 3000 + i, 1)
        tref = res[5] if fcode == S.F_COLL else tf
        det = [x for x in res[:2] if x > 0]
        t_det = min(det) - tref if det else np.nan
        row = dict(v=float(v), fmag=float(fmag), det_ms=1e3 * t_det,
                   rv_ms=1e3 * (res[0] - tref) if res[0] > 0 else np.nan,
                   rm_ms=1e3 * (res[1] - tref) if res[1] > 0 else np.nan,
                   cls=S.CLASS_NAMES[int(res[3])],
                   iso_ms=1e3 * (res[4] - tref) if res[4] > 0 else np.nan,
                   peak=float(res[6]))
        # baseline (following error, with its reaction for collisions)
        rb = S.R_FOLLOW if fcode == S.F_COLL else S.R_NONE
        logb, resb = S.simulate(pv, *ref, fcode, tf, fmag, qobs, e, THR, rb, 3000 + i, 1)
        row["fe_ms"] = 1e3 * (resb[2] - (resb[5] if fcode == S.F_COLL else tf)) if resb[2] > 0 else np.nan
        row["peak_fe"] = float(resb[6])
        # detection-only run for ROC statistics
        logn, resn = S.simulate(pv, *ref, fcode, tf, fmag, qobs, e, THR, S.R_NONE, 3000 + i, 1)
        tr = resn[5] if fcode == S.F_COLL else tf
        k0 = int(tr / Ts); k1 = k0 + int(0.020 / Ts)
        sv = debounced(logn[:, 7] / logn[:, 8], nd)[k0:k1].max()
        sm = debounced(np.abs(logn[:, 9]) / thr_m, nd)[k0:k1].max()
        sf = debounced(logn[:, 10] / thr_fe, nd)[k0:k1].max()
        if fcode in (S.F_FREEZE, S.F_OFFSET, S.F_DISC):
            pos["rv"].append(sv); pos["fe_enc"].append(sf)
        if fcode == S.F_COLL:
            pos["rm"].append(sm); pos["fe_coll"].append(sf)
        rows.append(row)
    mc[fname] = rows
    cl = [r["cls"] for r in rows]
    print("  %-11s det med %.2f ms  classes %s" % (
        fname, np.nanmedian([r["det_ms"] for r in rows]),
        {c: cl.count(c) for c in set(cl)}))
RES["monte_carlo"] = mc


def summ(x):
    x = np.array(x, float); ok = x[~np.isnan(x)]
    if len(ok) == 0:
        return dict(median=None, p95=None, max=None, n_detected=0)
    return dict(median=float(np.median(ok)), p95=float(np.percentile(ok, 95)),
                max=float(ok.max()), n_detected=int(len(ok)))


summary = {}
for fname, rows in mc.items():
    cl = [r["cls"] for r in rows]
    summary[fname] = dict(
        n=len(rows), det=summ([r["det_ms"] for r in rows]),
        iso=summ([r["iso_ms"] for r in rows]),
        fe=summ([r["fe_ms"] for r in rows]),
        correct=int(sum(c == S.CLASS_NAMES[TRUE[fname]] for c in cl)),
        classes={c: cl.count(c) for c in set(cl)})
    if fname == "collision":
        summary[fname]["peak"] = summ([r["peak"] for r in rows])
        summary[fname]["peak_fe"] = summ([r["peak_fe"] for r in rows])
RES["summary"] = summary

# ===================================================== 5. ROC
def roc(posv, negv):
    posv = np.array(posv); negv = np.array(negv)
    ms = np.concatenate([[0], np.sort(np.concatenate([posv, negv])), [np.inf]])
    tpr = [(posv > m).mean() for m in ms]; fpr = [(negv > m).mean() for m in ms]
    o = np.argsort(fpr)
    f = np.array(fpr)[o]; tp = np.array(tpr)[o]
    auc = float(np.trapezoid(tp, f)) if hasattr(np, "trapezoid") else float(np.trapz(tp, f))
    return np.array(fpr), np.array(tpr), auc, float((posv > 1).mean()), float((negv > 1).mean())


R = {}
R["rv_enc"] = roc(pos["rv"], neg_v)
R["fe_enc"] = roc(pos["fe_enc"], neg_fe)
R["rm_coll"] = roc(pos["rm"], neg_m)
R["fe_coll"] = roc(pos["fe_coll"], neg_fe)
RES["roc"] = {k: dict(auc=v[2], tpr_at_op=v[3], fpr_at_op=v[4]) for k, v in R.items()}
print("ROC", RES["roc"])

# ===================================================== 6. speed sweeps
print("speed sweeps ...")
speeds = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
sweep = dict(speeds=speeds, freeze_rv=[], freeze_fe=[], coll_rm=[], coll_fe=[],
             peak_prop=[], peak_fe=[], peak_none=[], clamp_none=[], clamp_prop=[])
for v in speeds:
    tf = 0.3; ref = S.ref_cruise(0.7, Ts, v)
    # detection only (no reaction), so both detectors see the same motion
    _, r = S.simulate(pv, *ref, S.F_FREEZE, tf, 0., 0., np.ones(6), THR, S.R_NONE, 7, 1)
    sweep["freeze_rv"].append(1e3 * (r[0] - tf) if r[0] > 0 else None)
    sweep["freeze_fe"].append(1e3 * (r[2] - tf) if r[2] > 0 else None)
    qobs = ref[0][int(tf / Ts)]
    lp, r = S.simulate(pv, *ref, S.F_COLL, tf, 0., qobs, np.ones(6), THR, S.R_PROPOSED, 7, 1)
    sweep["coll_rm"].append(1e3 * (r[1] - r[5])); sweep["peak_prop"].append(r[6])
    sweep["clamp_prop"].append(float(-lp[int((r[5] + 0.2) / Ts), 6]))
    _, r = S.simulate(pv, *ref, S.F_COLL, tf, 0., qobs, np.ones(6), THR, S.R_FOLLOW, 7, 1)
    sweep["coll_fe"].append(1e3 * (r[2] - r[5])); sweep["peak_fe"].append(r[6])
    ln, r = S.simulate(pv, *ref, S.F_COLL, tf, 0., qobs, np.ones(6), THR, S.R_NONE, 7, 1)
    sweep["peak_none"].append(r[6])
    sweep["clamp_none"].append(float(-ln[int((r[5] + 0.2) / Ts), 6]))
RES["sweep"] = sweep
print("  ", sweep)

# ===================================================== 7. example traces + figures
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "legend.fontsize": 7, "legend.frameon": True, "legend.framealpha": 0.92,
                     "legend.edgecolor": "none", "legend.borderpad": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.5,
                     "pdf.fonttype": 42})
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
W1 = 3.5


def mark_alarm(ax, t, c, lab=None):
    if t is not None and t > 0:
        ax.axvline(t, color=c, lw=0.8, ls=":", label=lab)


# --- encoder freeze example
v = 1.0; tf = 0.3; ref = S.ref_cruise(0.4, Ts, v)
log, r = S.simulate(pv, *ref, S.F_FREEZE, tf, 0., 0., np.ones(6), THR, S.R_NONE, 7, 1)
ex = dict(freeze_v=v, freeze_rv_ms=1e3 * (r[0] - tf), freeze_rm_ms=1e3 * (r[1] - tf),
          freeze_fe_ms=1e3 * (r[2] - tf) if r[2] > 0 else None)
t = (log[:, 0] - tf) * 1e3; m = (t > -3) & (t < 12)
fig, ax = plt.subplots(3, 1, figsize=(W1, 3.0), sharex=True)
ax[0].plot(t[m], np.rad2deg(log[m, 11]), color=INK)
ax[0].set_ylabel(r"$\delta$ [deg el.]")
ax[1].plot(t[m], log[m, 7], color=BLUE, label=r"$\|r_v\|$")
ax[1].plot(t[m], log[m, 8], color=INK, ls="--", lw=0.8, label=r"$\tau_v$")
ax[1].set_ylabel("[V]"); ax[1].legend(loc="center right", ncol=2)
ax[2].plot(t[m], log[m, 9], color=AQUA, label=r"$r_m$")
ax[2].axhline(thr_m, color=INK, ls="--", lw=0.8); ax[2].axhline(-thr_m, color=INK, ls="--", lw=0.8)
ax[2].set_ylabel("[Nm]"); ax[2].legend(loc="center right")
for a in ax:
    a.axvline(0, color=GRAY, lw=0.8)
mark_alarm(ax[1], ex["freeze_rv_ms"], BLUE)
mark_alarm(ax[2], ex["freeze_rm_ms"], AQUA)
ax[2].set_xlabel("time from encoder freeze [ms]")
fig.align_ylabels(ax); fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "sim_encoder_freeze.pdf")); plt.close(fig)

# --- collision example
v = 2.0; tf = 0.3; ref = S.ref_cruise(0.5, Ts, v); qobs = ref[0][int(tf / Ts)]
lp, rp = S.simulate(pv, *ref, S.F_COLL, tf, 0., qobs, np.ones(6), THR, S.R_PROPOSED, 7, 1)
lf, rf = S.simulate(pv, *ref, S.F_COLL, tf, 0., qobs, np.ones(6), THR, S.R_FOLLOW, 7, 1)
ln, rn = S.simulate(pv, *ref, S.F_COLL, tf, 0., qobs, np.ones(6), THR, S.R_NONE, 7, 1)
tc = rp[5]
ex.update(coll_v=v, coll_rm_ms=1e3 * (rp[1] - tc), coll_fe_ms=1e3 * (rf[2] - rf[5]),
          coll_peak_prop=rp[6], coll_peak_fe=rf[6], coll_peak_none=rn[6],
          coll_rv_alarm=bool(rp[0] > 0))
fig, ax = plt.subplots(3, 1, figsize=(W1, 3.0), sharex=True)
for L_, c, lab, ls in [(ln, GRAY, "no reaction", "-"), (lf, ORANGE, "following error", "--"),
                       (lp, BLUE, r"proposed ($r_m$)", "-")]:
    tt = (L_[:, 0] - tc) * 1e3; mm = (tt > -5) & (tt < 60)
    ax[0].plot(tt[mm], -L_[mm, 6], color=c, ls=ls, label=lab)
tt = (lp[:, 0] - tc) * 1e3; mm = (tt > -5) & (tt < 60)
ax[0].set_ylabel("contact [Nm]"); ax[0].legend(loc="upper right")
ax[1].plot(tt[mm], lp[mm, 9], color=AQUA, label=r"$r_m$")
ax[1].axhline(-thr_m, color=INK, ls="--", lw=0.8); ax[1].axhline(thr_m, color=INK, ls="--", lw=0.8)
ax[1].set_ylabel("[Nm]"); ax[1].legend(loc="lower right")
ax[2].plot(tt[mm], lp[mm, 7], color=BLUE, label=r"$\|r_v\|$")
ax[2].plot(tt[mm], lp[mm, 8], color=INK, ls="--", lw=0.8, label=r"$\tau_v$")
ax[2].set_ylabel("[V]"); ax[2].legend(loc="center right", ncol=2)
mark_alarm(ax[0], ex["coll_rm_ms"], BLUE); mark_alarm(ax[0], ex["coll_fe_ms"], ORANGE)
for a in ax:
    a.axvline(0, color=GRAY, lw=0.8)
ax[2].set_xlabel("time from contact [ms]")
fig.align_ylabels(ax); fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "sim_collision.pdf")); plt.close(fig)

# --- phase open example (r_v and phase currents)
v = 2.0; tf = 0.3; ref = S.ref_cruise(0.4, Ts, v)
lg, rg = S.simulate(pv, *ref, S.F_PHASE, tf, 0., 0., np.ones(6), THR, S.R_PROPOSED, 7, 1)
ex.update(phase_rv_ms=1e3 * (rg[0] - tf), phase_iso_ms=1e3 * (rg[4] - tf),
          phase_cls=S.CLASS_NAMES[int(rg[3])])
RES["examples"] = ex

# --- sweeps
fig, ax = plt.subplots(1, 2, figsize=(W1, 1.7))
sp = np.array(speeds)
fr = np.array([np.nan if x is None else x for x in sweep["freeze_rv"]], float)
ff = np.array([np.nan if x is None else x for x in sweep["freeze_fe"]], float)
ax[0].plot(sp, fr, "o-", color=BLUE, ms=3, label=r"$r_v$")
ax[0].plot(sp, ff, "s--", color=ORANGE, ms=3, label="foll. err.")
ax[0].set_yscale("log"); ax[0].set_xlabel("joint speed [rad/s]")
ax[0].set_ylabel("latency [ms]"); ax[0].set_title("encoder freeze", fontsize=8)
ax[0].legend(loc="upper right")
ax[1].plot(sp, sweep["peak_none"], "^-", color=GRAY, ms=3, label="no reaction")
ax[1].plot(sp, sweep["peak_fe"], "s--", color=ORANGE, ms=3, label="foll. err.")
ax[1].plot(sp, sweep["peak_prop"], "o-", color=BLUE, ms=3, label=r"$r_m$")
ax[1].set_xlabel("joint speed [rad/s]"); ax[1].set_ylabel("peak contact [Nm]")
ax[1].set_title("collision", fontsize=8); ax[1].legend(loc="center right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "sim_speed_sweep.pdf")); plt.close(fig)

# --- ROC
fig, ax = plt.subplots(1, 2, figsize=(W1, 1.75), sharey=True)
for a, (kp, kb, title) in zip(ax, [("rv_enc", "fe_enc", "encoder faults"),
                                   ("rm_coll", "fe_coll", "collisions")]):
    fp, tp, auc, tpo, fpo = R[kp]
    o = np.argsort(fp); a.step(fp[o], tp[o], where="post", color=BLUE,
                               label=("$r_v$" if kp == "rv_enc" else "$r_m$") + f" (AUC {auc:.2f})")
    a.plot([fpo], [tpo], "o", color=BLUE, ms=4)
    fp, tp, auc, tpo, fpo = R[kb]
    o = np.argsort(fp); a.step(fp[o], tp[o], where="post", color=ORANGE, ls="--",
                               label=f"foll. err. (AUC {auc:.2f})")
    a.plot([fpo], [tpo], "s", color=ORANGE, ms=4)
    a.set_xscale("symlog", linthresh=0.01); a.set_xlim(0, 1)
    a.set_title(title, fontsize=8)
    a.legend(loc="lower right", fontsize=6)
ax[0].set_ylabel("TPR (20 ms)")
fig.supxlabel("false-positive rate (1-s windows of normal motion)", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "sim_roc.pdf")); plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(clean(RES), open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))
