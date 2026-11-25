"""
Joint-actuator FDI simulation for the paper
"Isolating Sensor Faults from Mechanical Events in Robot Joint Actuators
Using Back-EMF and Momentum Residuals".

Plant   : SPM PMSM in the stationary (alpha-beta) frame, 3-phase inverter with
          dead-time voltage error, gearbox, rigid link with Coulomb+viscous
          friction, spring-damper obstacle, load-torque step.
Control : 20 kHz FOC (PI current loops with decoupling) + PD position loop
          with model feedforward, all using the *measured* (possibly faulty)
          encoder angle.
FDI     : back-EMF voltage residual r_v, generalized-momentum observer r_m,
          following-error baseline, isolation state machine.

Parameters are representative of a small quasi-direct-drive actuator and are
NOT identified from the testbed. Replace PLANT/MODEL values with identified
ones when available.
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- parameters
P = dict(
    p=14,            # pole pairs
    R=0.30,          # phase resistance [ohm]
    L=100e-6,        # phase inductance (Ld = Lq) [H]
    lam=6.0e-3,      # PM flux linkage [Wb]
    N=6.0,           # gear ratio
    J=2.0e-3,        # joint-side inertia [kg m^2]
    b=0.010,         # viscous friction [Nm s/rad]
    tc=0.050,        # Coulomb friction [Nm]
    Vdc=24.0,        # bus voltage [V]
    Vdt=0.10,        # dead-time voltage error amplitude [V]
    Imax=8.0,        # current limit [A]
    Ts=50e-6,        # control period (20 kHz)
    nsub=5,          # plant substeps per control period (10 us)
    enc_bits=14,     # motor-side absolute encoder resolution
    sig_i=0.03,      # current-sensor noise std [A]
    wc_speed=2*np.pi*300.0,  # speed-estimate low-pass cutoff [rad/s]
    wc_res=2*np.pi*500.0,    # residual filter cutoff a [rad/s]
    K_mo=200.0,      # momentum-observer gain [1/s]
    wbw_i=2*np.pi*1500.0,    # current-loop bandwidth [rad/s]
    wn_pos=2*np.pi*10.0,     # position-loop natural frequency [rad/s]
    k_c=100.0,       # obstacle stiffness [Nm/rad]
    d_c=0.05,        # obstacle damping [Nm s/rad]
    nd=5,            # debounce samples (0.25 ms)
    W_iso=20,        # isolation window after first r_m alarm (1 ms)
    T_coll=0.040,    # persistence time separating COLL from LOAD [s]
)

# fault codes
F_NONE, F_FREEZE, F_OFFSET, F_DISC, F_PHASE, F_COLL, F_LOAD = range(7)
# class codes
C_OK, C_ENC, C_COLL, C_LOAD, C_PHASE, C_PEND, C_ENCPH = range(7)
CLASS_NAMES = {0: "none", 1: "ENC", 2: "COLL", 3: "LOAD", 4: "PHASE", 5: "PEND", 6: "ENC/PHASE?"}
# reaction modes
R_NONE, R_PROPOSED, R_FOLLOW = range(3)

LOG_COLS = ["t", "q", "qhat", "qref", "qd", "iq", "tau_ext", "rv", "tau_v",
            "rm", "fe", "delta", "ia", "ib", "ic", "cls", "flags", "vq",
            "wehat"]


def param_vec(P):
    keys = ["p", "R", "L", "lam", "N", "J", "b", "tc", "Vdc", "Vdt", "Imax",
            "Ts", "nsub", "enc_bits", "sig_i", "wc_speed", "wc_res", "K_mo",
            "wbw_i", "wn_pos", "k_c", "d_c", "nd", "W_iso", "T_coll"]
    return np.array([float(P[k]) for k in keys])


@njit(cache=True)
def simulate(pv, qref, qdref, qddref, fault, t_fault, fmag, q_obs,
             err, thr, react, seed, dec):
    """
    pv     : parameter vector (param_vec)
    qref.. : reference arrays at control rate
    fault  : fault code; t_fault: injection time [s]; fmag: magnitude
             (OFFSET: electrical offset [rad]; LOAD: torque [Nm])
    q_obs  : obstacle position for F_COLL [rad]
    err    : model/plant ratios used by the controller+FDI
             [R, lam, J, b, tc, L]  (model = plant * err)
    thr    : [c0, c1, c2, thr_m, thr_fe, m_v]  thresholds
    react  : reaction mode
    """
    np.random.seed(seed)
    p = pv[0]; R = pv[1]; L = pv[2]; lam = pv[3]; N = pv[4]; J = pv[5]
    b = pv[6]; tc = pv[7]; Vdc = pv[8]; Vdt = pv[9]; Imax = pv[10]
    Ts = pv[11]; nsub = int(pv[12]); bits = pv[13]; sig_i = pv[14]
    wcs = pv[15]; a = pv[16]; K = pv[17]; wbw = pv[18]; wn = pv[19]
    k_c = pv[20]; d_c = pv[21]; nd = int(pv[22]); W_iso = int(pv[23])
    T_coll = pv[24]

    # model (what the controller / FDI believe)
    Rm = R * err[0]; lamm = lam * err[1]; Jm = J * err[2]
    bm = b * err[3]; tcm = tc * err[4]; Lm = L * err[5]
    ktm = 1.5 * p * lamm
    c0 = thr[0]; c1 = thr[1]; c2 = thr[2]; thr_m = thr[3]; thr_fe = thr[4]
    m_v = thr[5]

    dt = Ts / nsub
    n = qref.shape[0]
    nlog = n // dec
    log = np.zeros((nlog, 19), dtype=np.float32)
    s3 = np.sqrt(3.0)
    qres = 2.0 * np.pi / (2.0 ** bits)

    # gains
    Kp_i = Lm * wbw; Ki_i = Rm * wbw
    Kp_q = Jm * wn * wn; Kd_q = 2.0 * 0.8 * wn * Jm
    Vmax = Vdc / s3

    # plant state
    ia_ = 0.0; ib_ = 0.0          # alpha, beta currents
    q = qref[0]; qd = 0.0
    tau_ext_now = 0.0

    # controller state
    th_prev_meas = 0.0; th_unw = 0.0; first = True
    q0_offset = 0.0
    wm_hat = 0.0
    xi_d = 0.0; xi_q = 0.0
    vd_c = 0.0; vq_c = 0.0       # command applied during the coming period
    vd_prev = 0.0; vq_prev = 0.0
    # residual filters
    xf_d = 0.0; xf_q = 0.0; if_d = 0.0; if_q = 0.0
    # momentum observer
    I_mo = 0.0; r_m = 0.0; rho0 = 0.0
    # phase current RMS trackers
    msa = 0.0; msb = 0.0; msc = 0.0
    # alarms
    cnt_v = 0; cnt_m = 0; cnt_fe = 0
    first_v = -1.0; first_m = -1.0; first_fe = -1.0
    cls = C_OK; t_cls = -1.0; k_alarm0 = -1
    t_contact = -1.0; peak_ct = 0.0
    react_off = False     # torque off (compliant), current loop alive
    sto = False           # voltage off
    th_freeze = 0.0
    load_ff = False
    kw = 0; t_det = -1.0
    di0 = np.zeros(3); di1 = np.zeros(3); Vp = 2.0
    ae = np.exp(-a * Ts)
    aws = np.exp(-wcs * Ts)
    ams = np.exp(-2.0 * np.pi * 200.0 * Ts)

    for k in range(n):
        t = k * Ts
        fault_on = (fault != F_NONE) and (t >= t_fault)

        # ------------------------------------------------ measurements
        the = p * N * q
        ia_t = ia_
        ib_t = -0.5 * ia_ + 0.5 * s3 * ib_
        ic_t = -0.5 * ia_ - 0.5 * s3 * ib_
        ia_m = ia_t + sig_i * np.random.randn()
        ib_m = ib_t + sig_i * np.random.randn()
        ic_m = ic_t + sig_i * np.random.randn()
        ial_m = (2.0 / 3.0) * (ia_m - 0.5 * ib_m - 0.5 * ic_m)
        ibe_m = (ib_m - ic_m) / s3

        thm_true = (N * q) % (2.0 * np.pi)
        thm_meas = np.floor(thm_true / qres) * qres
        if fault_on and fault == F_FREEZE:
            thm_meas = th_freeze
        else:
            th_freeze = thm_meas
        if fault_on and fault == F_OFFSET:
            thm_meas = (thm_meas + fmag / p) % (2.0 * np.pi)
        if fault_on and fault == F_DISC:
            thm_meas = 0.0

        if first:
            th_prev_meas = thm_meas; th_unw = 0.0
            q0_offset = q - thm_meas / N
            first = False
        dth = thm_meas - th_prev_meas
        if dth > np.pi:
            dth -= 2.0 * np.pi
        elif dth < -np.pi:
            dth += 2.0 * np.pi
        th_prev_meas = thm_meas
        th_unw += dth
        wm_hat = aws * wm_hat + (1.0 - aws) * dth / Ts
        qhat = th_unw / N + q0_offset + (thm_meas - thm_meas)
        qdhat = wm_hat / N
        the_hat = p * thm_meas
        we_hat = p * wm_hat

        c_ = np.cos(the_hat); s_ = np.sin(the_hat)
        id_m = c_ * ial_m + s_ * ibe_m
        iq_m = -s_ * ial_m + c_ * ibe_m

        # ------------------------------------------------ position loop
        e_q = qref[k] - qhat
        fric_ff = bm * qdref[k] + tcm * np.tanh(qdref[k] / 0.01)
        tau_ref = (Jm * qddref[k] + fric_ff + Kp_q * e_q
                   + Kd_q * (qdref[k] - qdhat))
        if load_ff:
            tau_ref -= r_m
        iq_ref = tau_ref / (N * ktm)
        if react_off:
            iq_ref = 0.0
        if iq_ref > Imax:
            iq_ref = Imax
        elif iq_ref < -Imax:
            iq_ref = -Imax
        id_ref = 0.0

        # ------------------------------------------------ current loop
        ed = id_ref - id_m; eq = iq_ref - iq_m
        vd_ff = -we_hat * Lm * iq_m
        vq_ff = we_hat * (Lm * id_m + lamm)
        vd_u = Kp_i * ed + xi_d + vd_ff
        vq_u = Kp_i * eq + xi_q + vq_ff
        mag = np.sqrt(vd_u * vd_u + vq_u * vq_u)
        sat = 1.0
        if mag > Vmax:
            sat = Vmax / mag
        vd_new = vd_u * sat; vq_new = vq_u * sat
        if sat == 1.0:
            xi_d += Ki_i * ed * Ts; xi_q += Ki_i * eq * Ts
        if sto and react == R_PROPOSED:
            vd_new = 0.0; vq_new = 0.0; xi_d = 0.0; xi_q = 0.0

        # ------------------------------------------------ back-EMF residual
        # uses the command applied over the previous period (vd_c, vq_c)
        ud = vd_c - Rm * id_m + we_hat * Lm * iq_m
        uq = vq_c - Rm * iq_m - we_hat * (Lm * id_m + lamm)
        xf_d = ae * xf_d + (1.0 - ae) * ud
        xf_q = ae * xf_q + (1.0 - ae) * uq
        if_d = ae * if_d + (1.0 - ae) * id_m
        if_q = ae * if_q + (1.0 - ae) * iq_m
        rv_d = xf_d - Lm * a * (id_m - if_d)
        rv_q = xf_q - Lm * a * (iq_m - if_q)
        rv = np.sqrt(rv_d * rv_d + rv_q * rv_q)
        inorm = np.sqrt(id_m * id_m + iq_m * iq_m)
        tau_v = c0 + c1 * abs(we_hat) + c2 * inorm

        # ------------------------------------------------ momentum observer
        rho = Jm * qdhat
        if k == 0:
            rho0 = rho
        tau_m_hat = N * ktm * iq_m
        I_mo += Ts * (tau_m_hat - bm * qdhat - tcm * np.tanh(qdhat / 0.01)
                      + r_m)
        r_m = K * (rho - I_mo - rho0)

        # phase-current mean squares (for PHASE isolation)
        msa = ams * msa + (1.0 - ams) * ia_m * ia_m
        msb = ams * msb + (1.0 - ams) * ib_m * ib_m
        msc = ams * msc + (1.0 - ams) * ic_m * ic_m

        # ------------------------------------------------ alarms
        warm = t > 0.05  # filters settle
        cnt_v = cnt_v + 1 if (warm and rv > m_v * tau_v) else 0
        cnt_m = cnt_m + 1 if (warm and abs(r_m) > thr_m) else 0
        fe = abs(e_q)
        cnt_fe = cnt_fe + 1 if (warm and fe > thr_fe) else 0
        al_v = cnt_v >= nd; al_m = cnt_m >= nd; al_fe = cnt_fe >= nd
        if al_v and first_v < 0: first_v = t
        if al_m and first_m < 0: first_m = t
        if al_fe and first_fe < 0: first_fe = t

        # ------------------------------------------------ isolation
        pulse = -1
        if cls == C_ENCPH and react == R_PROPOSED:
            # ENC vs PHASE: angle-free diagnostic. Under active short circuit,
            # apply a 2-sample stationary-frame voltage pulse along each phase
            # axis in turn and measure that phase's current response. An open
            # phase gives no response along its own axis.
            kk = kw - 40          # 2 ms settling under ASC first
            m = kk // 6; ph = kk % 6
            if kk >= 0 and m < 3:
                ix = ia_m if m == 0 else (ib_m if m == 1 else ic_m)
                if ph == 0:
                    di0[m] = ix
                if ph == 2:
                    di1[m] = ix - di0[m]
                if ph < 2:
                    pulse = m
            kw += 1
            if kw > 40 + 18:
                med = np.median(di1)
                if med > 0.3 and di1.min() < 0.3 * med:
                    cls = C_PHASE
                else:
                    cls = C_ENC
                t_cls = t
        if cls == C_OK or cls == C_PEND:
            if al_v:
                cls = C_ENCPH; t_det = t; kw = 0
                sto = True   # ENC and PHASE both lead to active short circuit
            elif cls == C_OK and al_m:
                cls = C_PEND; k_alarm0 = k
                if react == R_PROPOSED:
                    react_off = True
            elif cls == C_PEND:
                if (k - k_alarm0) * Ts >= T_coll:
                    if abs(r_m) > thr_m:
                        cls = C_LOAD
                        if react == R_PROPOSED:
                            react_off = False; load_ff = True
                    else:
                        cls = C_COLL
                    t_cls = t
        if react == R_FOLLOW and al_fe:
            react_off = True

        # ------------------------------------------------ plant (substeps)
        vd_c = vd_new; vq_c = vq_new
        val_c = c_ * vd_c - s_ * vq_c
        vbe_c = s_ * vd_c + c_ * vq_c
        for j in range(nsub):
            the = p * N * q
            we = p * N * qd
            ia_t = ia_
            ib_t = -0.5 * ia_ + 0.5 * s3 * ib_
            ic_t = -0.5 * ia_ - 0.5 * s3 * ib_
            va = val_c - Vdt * np.tanh(ia_t / 0.05)
            vb = (-0.5 * val_c + 0.5 * s3 * vbe_c) - Vdt * np.tanh(ib_t / 0.05)
            vc = (-0.5 * val_c - 0.5 * s3 * vbe_c) - Vdt * np.tanh(ic_t / 0.05)
            if sto and react == R_PROPOSED:
                va = 0.0; vb = 0.0; vc = 0.0   # active short circuit
                if pulse == 0:
                    va = Vp * 2.0 / 3.0; vb = -Vp / 3.0; vc = -Vp / 3.0
                elif pulse == 1:
                    vb = Vp * 2.0 / 3.0; va = -Vp / 3.0; vc = -Vp / 3.0
                elif pulse == 2:
                    vc = Vp * 2.0 / 3.0; va = -Vp / 3.0; vb = -Vp / 3.0
            val = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
            vbe = (vb - vc) / s3
            ea = -we * lam * np.sin(the)
            eb = we * lam * np.cos(the)
            dia = (val - R * ia_ - ea) / L
            dib = (vbe - R * ib_ - eb) / L
            ia_ += dt * dia
            ib_ += dt * dib
            if fault_on and fault == F_PHASE:
                ia_ = 0.0
            tau_m = N * 1.5 * p * lam * (-ia_ * np.sin(the) + ib_ * np.cos(the))
            tau_f = b * qd + tc * np.tanh(qd / 0.01)
            tau_ext_now = 0.0
            if fault == F_COLL and q > q_obs:
                fc = k_c * (q - q_obs) + d_c * qd
                if fc > 0.0:
                    tau_ext_now = -fc
                    if t_contact < 0.0:
                        t_contact = t + j * dt
                    if fc > peak_ct:
                        peak_ct = fc
            if fault == F_LOAD and fault_on:
                tau_ext_now = -fmag
            qdd = (tau_m - tau_f + tau_ext_now) / J
            qd += dt * qdd
            q += dt * qd

        # ------------------------------------------------ log
        if k % dec == 0 and k // dec < nlog:
            r = k // dec
            d = (p * N * q - the_hat + np.pi) % (2.0 * np.pi) - np.pi
            log[r, 0] = t; log[r, 1] = q; log[r, 2] = qhat
            log[r, 3] = qref[k]; log[r, 4] = qd; log[r, 5] = iq_m
            log[r, 6] = tau_ext_now; log[r, 7] = rv; log[r, 8] = tau_v
            log[r, 9] = r_m; log[r, 10] = fe; log[r, 11] = d
            log[r, 12] = ia_m; log[r, 13] = ib_m; log[r, 14] = ic_m
            log[r, 15] = cls
            log[r, 16] = (1 if al_v else 0) + (2 if al_m else 0) + (4 if al_fe else 0)
            log[r, 17] = vq_c; log[r, 18] = we_hat

    res = np.array([first_v, first_m, first_fe, float(cls), t_cls,
                    t_contact, peak_ct])
    return log, res


# ---------------------------------------------------------------- references
def ref_from_q(q, Ts):
    qd = np.gradient(q, Ts)
    qdd = np.gradient(qd, Ts)
    return q, qd, qdd


def ref_sine(T, Ts, A=1.0, f=0.5):
    t = np.arange(int(T / Ts)) * Ts
    w = 2 * np.pi * f
    return A * np.sin(w * t), A * w * np.cos(w * t), -A * w * w * np.sin(w * t)


def ref_chirp(T, Ts, A=0.3, f0=0.1, f1=4.0):
    t = np.arange(int(T / Ts)) * Ts
    k = (f1 - f0) / T
    ph = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    w = 2 * np.pi * (f0 + k * t)
    q = A * np.sin(ph)
    qd = A * w * np.cos(ph)
    qdd = -A * w * w * np.sin(ph) + A * 2 * np.pi * k * np.cos(ph)
    return q, qd, qdd


def _minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def ref_random(T, Ts, rng, qmax=1.5, vmax=6.0):
    n = int(T / Ts)
    q = np.zeros(n)
    k = 0; cur = 0.0
    while k < n:
        tgt = rng.uniform(-qmax, qmax)
        dist = abs(tgt - cur)
        dur = max(0.25, 1.875 * dist / vmax * rng.uniform(1.0, 3.0))
        m = int(dur / Ts)
        s = np.linspace(0, 1, m)
        seg = cur + (tgt - cur) * _minjerk(s)
        q[k:k + m] = seg[: max(0, min(m, n - k))]
        k += m; cur = tgt
        dw = int(rng.uniform(0.05, 0.5) / Ts)
        q[k:k + dw] = cur
        k += dw
    q = q[:n]
    t = np.arange(n) * Ts
    qd = np.empty(n); qdd = np.empty(n)
    qd[:] = np.gradient(q, Ts); qdd[:] = np.gradient(qd, Ts)
    return q, qd, qdd


def ref_cruise(T, Ts, v, t_acc=0.1, q0=0.0):
    """Accelerate to constant speed v in t_acc (smooth), then cruise."""
    t = np.arange(int(T / Ts)) * Ts
    s = np.clip(t / t_acc, 0, 1)
    # smooth velocity ramp (min-jerk on velocity)
    vel = v * _minjerk(s)
    q = q0 + np.cumsum(vel) * Ts
    acc = np.gradient(vel, Ts)
    return q, vel, acc
