import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from scipy.integrate import solve_ivp



"""
<pk models>

pk_1c_iv
One-compartment IV bolus with first-order elimination from the central compartment.
Captures mono-exponential decay with parameters CL, V.

pk_1c_oral
One-compartment with first-order oral absorption (gut → central) and linear elimination.
Classic oral PK shape controlled by ka, CL, V.

pk_1c_oral_lag
Oral absorption model with a lag time Tlag before drug enters the gut compartment.
Produces delayed onset/Tmax compared to standard oral absorption.

pk_1c_iv_mm
One-compartment IV with linear clearance plus Michaelis–Menten nonlinear elimination.
Generates concentration-dependent clearance via Vmax, Km (with CL, V).

pk_2c_iv
Two-compartment IV bolus with distribution between central/peripheral and linear elimination.
Multi-phase decay controlled by CL, Vc, Vp, Q.

pk_2c_iv_infusion
Two-compartment IV where each dosing event is delivered as a short infusion over tau.
Models IV infusion regimens using CL, Vc, Vp, Q, tau.

pk_2c_oral
Two-compartment model with first-order oral absorption into central plus distribution.
Adds absorption-driven rise and multi-phase decline (ka, CL, Vc, Vp, Q).

pk_1c_transit_abs
One-compartment with a fixed chain of transit compartments before reaching central.
Smooths/delays absorption using ktr (plus CL, V).

pk_3c_iv
Three-compartment IV bolus (two peripheral compartments) with two distribution clearances.
More complex multi-exponential profiles using CL, Vc, Vp, Q, Vp2, Q2.

pk_1c_oral_flipflop
Same equations as 1c oral, but parameter priors favor very small ka (absorption-limited).
Produces flip-flop kinetics where terminal phase reflects absorption rather than elimination.



<pd models>
pd_direct_linear
Direct effect model: biomarker changes linearly with concentration (R = R0 + SLOPE·C).
Useful as a baseline exposure–response model with R0, SLOPE.

pd_direct_emax
Direct saturable stimulatory effect (R = R0 + Emax·C/(EC50+C)).
Captures hyperbolic saturation with R0, Emax, EC50.

pd_direct_sigmoid
Sigmoid Emax with Hill coefficient for steeper/shallower transitions around EC50.
Models cooperative-like response using R0, Emax, EC50, Hill.

pd_direct_inhib_emax
Direct inhibitory Emax: reduces baseline signal by fraction Imax·C/(IC50+C).
Represents inhibition with R0, Imax, IC50.

pd_effect_comp_emax
Effect-compartment hysteresis where effect-site concentration Ce lags plasma via ke0.
Response follows Emax on Ce using R0, Emax, EC50, ke0.

pd_indirect_inhib_kin
Turnover model where drug inhibits production rate kin (IC50 relationship).
Gives delayed suppression governed by kin, kout, IC50 (baseline R0=kin/kout).

pd_indirect_stim_kin
Turnover model where drug stimulates production kin via Emax/EC50.
Delayed biomarker rise controlled by kin, kout, Emax, EC50.

pd_indirect_inhib_kout
Turnover model where drug inhibits loss/removal kout, increasing response.
Produces delayed elevation with kin, kout, Imax, IC50.

pd_indirect_stim_kout
Turnover model where drug stimulates kout, decreasing response faster.
Produces delayed drop with kin, kout, Emax, EC50.

pd_transit_delay
Transit-compartment delay from exposure-driven input to observed response.
Smooths and delays biomarker kinetics via ktr plus Emax, EC50, R0.




"""
# TrajectoryGenerator.py
# Scale-tuned priors for:
#   dose ~ 0–10
#   concentration ~ 0–15
#   biomarker ~ 5–15
#
# Main changes vs your pasted version:
#   (1) generate_dose_vector(): dose_range -> (0.0, 10.0)
#   (2) build_pk_models(): shrink V/Vc and CL/Q ranges to lift concentration scale
#   (3) build_pd_models(): set R0 ~ (5,15) and tighten effect parameter ranges to keep biomarker ~ (5,15)
#   (4) clip conc_at_pd_times in generate_pd_sample (extra safety)

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from scipy.integrate import solve_ivp

# =========================
# 0) Your timesteps
# =========================
PK_TIMES = np.array([
    0.000e+00, 1.000e+00, 2.000e+00, 4.000e+00, 8.000e+00, 1.200e+01,
    2.400e+01, 3.600e+01, 4.800e+01, 7.200e+01, 9.600e+01, 1.200e+02,
    1.440e+02, 1.680e+02, 1.920e+02, 2.160e+02, 2.400e+02, 2.640e+02,
    2.880e+02, 3.120e+02, 3.360e+02, 3.600e+02, 3.840e+02, 4.080e+02,
    4.320e+02, 4.560e+02, 4.800e+02, 4.810e+02, 4.820e+02, 4.840e+02,
    4.880e+02, 4.920e+02, 5.040e+02, 5.520e+02, 6.000e+02, 6.720e+02,
    8.400e+02, 1.008e+03, 1.176e+03
], dtype=float)

PD_TIMES = np.array([
    0.000e+00, 1.000e+00, 2.000e+00, 4.000e+00, 8.000e+00, 1.200e+01,
    2.400e+01, 3.600e+01, 4.800e+01, 7.200e+01, 1.680e+02, 3.360e+02,
    4.800e+02, 4.810e+02, 4.820e+02, 4.840e+02, 4.880e+02, 4.920e+02,
    5.040e+02, 5.520e+02, 6.000e+02, 6.720e+02, 8.400e+02, 1.008e+03,
    1.176e+03
], dtype=float)

# =========================
# 1) Utilities: noise + IIV
# =========================
def apply_iiv_lognormal(x: float, omega: float, rng: np.random.Generator) -> float:
    # lognormal around x with std omega in log-space
    return float(x * np.exp(rng.normal(0.0, omega)))

def add_residual_error(y: np.ndarray,
                       prop_sigma: float,
                       add_sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    eps_prop = rng.normal(0.0, prop_sigma, size=y.shape)
    eps_add  = rng.normal(0.0, add_sigma,  size=y.shape)
    y_noisy = y * (1.0 + eps_prop) + eps_add
    return np.clip(y_noisy, 0.0, None)

# =========================
# 2) Dosing regimen generator (dose vector aligned to PK_TIMES)
# =========================
def generate_dose_vector(times: np.ndarray,
                         rng: np.random.Generator,
                         dose_range=(0.0, 10.0),                 # <<< CHANGED (was (1.0,10.0))
                         interval_choices=(24.0, 48.0, 72.0),
                         max_doses=25,
                         miss_prob=0.10) -> np.ndarray:
    """
    Returns dose_amt_at_obs_times (same length as times).
    Dose is modeled as a bolus "administered at that timestamp".
    Observation concentration at that timestamp is interpreted as *pre-dose*,
    i.e., we apply dose AFTER recording at that time.
    """
    dose_amt = rng.uniform(*dose_range)
    interval = float(rng.choice(interval_choices))
    # schedule doses starting at t=0
    sched = [0.0]
    for _ in range(max_doses - 1):
        nxt = sched[-1] + interval
        if nxt > times[-1]:
            break
        sched.append(nxt)

    # map to observation grid (exact matches only)
    dose_vec = np.zeros_like(times, dtype=float)
    for t in sched:
        if rng.random() < miss_prob:
            continue
        idx = np.where(np.isclose(times, t))[0]
        if len(idx) > 0:
            dose_vec[idx[0]] = dose_amt
    return dose_vec

# =========================
# 3) Generic piecewise ODE simulator
# =========================
def simulate_piecewise(
    odefun: Callable[[float, np.ndarray, Dict], np.ndarray],
    y0: np.ndarray,
    obs_times: np.ndarray,
    bolus_events: Dict[float, float],
    apply_bolus: Callable[[np.ndarray, float], np.ndarray],
    infusion_events: Optional[List[Tuple[float, float, float]]] = None,
    params: Optional[Dict] = None
) -> np.ndarray:
    """
    Integrates across boundaries from bolus times and infusion start/end times.
    Returns states at obs_times (including obs_times[0]).
    """
    params = {} if params is None else params
    infusion_events = [] if infusion_events is None else infusion_events

    obs_times = np.array(obs_times, dtype=float)
    assert np.all(np.diff(obs_times) >= 0)

    boundaries = {obs_times[0], obs_times[-1]}
    for t in bolus_events.keys():
        if obs_times[0] <= t <= obs_times[-1]:
            boundaries.add(float(t))
    for (ts, te, _rate) in infusion_events:
        if obs_times[0] <= ts <= obs_times[-1]:
            boundaries.add(float(ts))
        if obs_times[0] <= te <= obs_times[-1]:
            boundaries.add(float(te))
    boundaries = np.array(sorted(boundaries), dtype=float)

    def infusion_rate_at(t: float) -> float:
        r = 0.0
        for (ts, te, rate) in infusion_events:
            if ts <= t < te:
                r += rate
        return r

    y_out = np.zeros((len(y0), len(obs_times)), dtype=float)
    y = y0.copy()
    y_out[:, 0] = y

    def eval_segment(t0, t1, y_init):
        mask = (obs_times > t0) & (obs_times <= t1)
        t_eval = obs_times[mask]
        t_eval_full = np.unique(np.concatenate([t_eval, np.array([t1])]))
        rate = infusion_rate_at(t0 + 1e-9)

        def f(t, yy):
            return odefun(t, yy, {**params, "inf_rate": rate})

        sol = solve_ivp(f, (t0, t1), y_init, t_eval=t_eval_full, rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError("ODE solve failed")

        for k, tt in enumerate(sol.t):
            idx = np.where(np.isclose(obs_times, tt))[0]
            if len(idx) > 0:
                y_out[:, idx[0]] = sol.y[:, k]

        return sol.y[:, -1]

    t_cur = obs_times[0]
    for b in boundaries[1:]:
        if b > t_cur:
            y = eval_segment(t_cur, b, y)
            t_cur = b

        # Apply bolus AFTER recording concentration at time b
        if b in bolus_events:
            y = apply_bolus(y, bolus_events[b])

    idx_end = np.where(np.isclose(obs_times, obs_times[-1]))[0][0]
    if np.allclose(y_out[:, idx_end], 0.0) and obs_times[-1] != obs_times[0]:
        y_out[:, idx_end] = y

    return y_out

# =========================
# 4) PK models (~10)
# =========================
PK_PARAM_KEYS = ["ka","CL","V","Vc","Vp","Q","Vp2","Q2","Vmax","Km","Tlag","ktr","tau"]
PK_PARAM_INDEX = {k:i for i,k in enumerate(PK_PARAM_KEYS)}
PK_PARAM_DIM = len(PK_PARAM_KEYS)

@dataclass
class PKModel:
    name: str
    sample_params: Callable[[np.random.Generator], Dict[str, float]]
    simulate_conc: Callable[[np.ndarray, np.ndarray, Dict[str, float], np.random.Generator], np.ndarray]
    param_mask: np.ndarray  # (PK_PARAM_DIM,)

def pk_param_vector(params: Dict[str,float]) -> np.ndarray:
    v = np.zeros((PK_PARAM_DIM,), dtype=float)
    for k,val in params.items():
        if k in PK_PARAM_INDEX:
            v[PK_PARAM_INDEX[k]] = float(val)
    return v

# ---- PK model implementations ----
def _pk_one_comp_iv_conc(times, dose_vec, p, rng):
    CL, V = p["CL"], p["V"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    def odefun(t, y, pp):
        A = y[0]
        inf = pp.get("inf_rate", 0.0)
        return np.array([- (CL / V)*A + inf], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[0] / V

def _pk_one_comp_oral_conc(times, dose_vec, p, rng):
    ka, CL, V = p["ka"], p["CL"], p["V"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    def odefun(t, y, pp):
        Ag, Ac = y
        dAg = -ka*Ag
        dAc = ka*Ag - (CL/V)*Ac
        return np.array([dAg, dAc], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0, 0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[1] / V

def _pk_one_comp_oral_lag_conc(times, dose_vec, p, rng):
    ka, CL, V, Tlag = p["ka"], p["CL"], p["V"], p["Tlag"]

    bolus = {}
    for t,d in zip(times, dose_vec):
        if d > 0:
            t_eff = float(t + Tlag)
            if t_eff <= times[-1]:
                bolus[t_eff] = bolus.get(t_eff, 0.0) + float(d)

    def odefun(t, y, pp):
        Ag, Ac = y
        return np.array([-ka*Ag, ka*Ag - (CL/V)*Ac], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0, 0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[1] / V

def _pk_one_comp_iv_mm_conc(times, dose_vec, p, rng):
    CL, V, Vmax, Km = p["CL"], p["V"], p["Vmax"], p["Km"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    def odefun(t, y, pp):
        A = y[0]
        C = A / V
        mm = Vmax * C / (Km + C + 1e-12)  # amount/time
        return np.array([- (CL / V)*A - mm], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[0] / V

def _pk_two_comp_iv_conc(times, dose_vec, p, rng, infusion=False):
    CL, Vc, Vp, Q = p["CL"], p["Vc"], p["Vp"], p["Q"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    infusion_events = []
    if infusion:
        tau = p["tau"]
        bolus = {}
        for t,d in zip(times, dose_vec):
            if d > 0:
                rate = float(d / tau)
                infusion_events.append((float(t), float(t+tau), rate))

    def odefun(t, y, pp):
        Ac, Ap = y
        inf = pp.get("inf_rate", 0.0)
        dAc = -(CL/Vc)*Ac - (Q/Vc)*Ac + (Q/Vp)*Ap + inf
        dAp = (Q/Vc)*Ac - (Q/Vp)*Ap
        return np.array([dAc, dAp], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0, 0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus,
                                infusion_events=infusion_events, params={})
    return states[0] / Vc

def _pk_two_comp_oral_conc(times, dose_vec, p, rng):
    ka, CL, Vc, Vp, Q = p["ka"], p["CL"], p["Vc"], p["Vp"], p["Q"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    def odefun(t, y, pp):
        Ag, Ac, Ap = y
        dAg = -ka*Ag
        dAc = ka*Ag - (CL/Vc)*Ac - (Q/Vc)*Ac + (Q/Vp)*Ap
        dAp = (Q/Vc)*Ac - (Q/Vp)*Ap
        return np.array([dAg, dAc, dAp], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0, 0.0, 0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[1] / Vc

def _pk_one_comp_transit_abs_conc(times, dose_vec, p, rng, n_tr=3):
    ktr, CL, V = p["ktr"], p["CL"], p["V"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    def odefun(t, y, pp):
        dy = np.zeros_like(y)
        dy[0] = -ktr*y[0]
        for i in range(1, n_tr):
            dy[i] = ktr*(y[i-1] - y[i])
        dy[n_tr] = ktr*y[n_tr-1] - (CL/V)*y[n_tr]
        return dy

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.zeros((n_tr+1,), dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[n_tr] / V

def _pk_three_comp_iv_conc(times, dose_vec, p, rng):
    CL, Vc, Vp, Q = p["CL"], p["Vc"], p["Vp"], p["Q"]
    Vp2, Q2 = p["Vp2"], p["Q2"]
    bolus = {float(t): float(d) for t,d in zip(times, dose_vec) if d > 0}

    def odefun(t, y, pp):
        Ac, Ap1, Ap2 = y
        dAc  = -(CL/Vc)*Ac - (Q/Vc)*Ac + (Q/Vp)*Ap1 - (Q2/Vc)*Ac + (Q2/Vp2)*Ap2
        dAp1 = (Q/Vc)*Ac - (Q/Vp)*Ap1
        dAp2 = (Q2/Vc)*Ac - (Q2/Vp2)*Ap2
        return np.array([dAc, dAp1, dAp2], dtype=float)

    def apply_bolus(y, amt):
        y2 = y.copy()
        y2[0] += amt
        return y2

    y0 = np.array([0.0, 0.0, 0.0], dtype=float)
    states = simulate_piecewise(odefun, y0, times, bolus, apply_bolus, params={})
    return states[0] / Vc

# ---- Build PK model library (10) ----
def build_pk_models(rng: np.random.Generator, omega_iiv=0.25) -> List[PKModel]:
    def mask_for(keys_used: List[str]) -> np.ndarray:
        m = np.zeros((PK_PARAM_DIM,), dtype=float)
        for k in keys_used:
            m[PK_PARAM_INDEX[k]] = 1.0
        return m

    # <<< MORE AGGRESSIVE concentration scaling >>>
    # Goal: pk conc up to ~15 for dose up to 10 given first post-dose obs at t=1h.
    CL_rng = (0.03, 1.0)      # smaller CL so C(1h) doesn't decay too much
    V_rng  = (0.20, 2.0)      # smaller V => higher concentration
    Vc_rng = (0.20, 2.0)      # smaller Vc => higher concentration
    Vp_rng = (1.0, 12.0)
    Q_rng  = (0.05, 2.0)

    models: List[PKModel] = []

    # 0) 1c IV bolus (linear)
    def sample_0(r):
        return {
            "CL": apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "V":  apply_iiv_lognormal(r.uniform(*V_rng),  omega_iiv, r),
        }
    models.append(PKModel("pk_1c_iv", sample_0, _pk_one_comp_iv_conc, mask_for(["CL","V"])))

    # 1) 1c oral (ka)
    def sample_1(r):
        return {
            "ka": apply_iiv_lognormal(r.uniform(0.1, 3.0), omega_iiv, r),
            "CL": apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "V":  apply_iiv_lognormal(r.uniform(*V_rng),  omega_iiv, r),
        }
    models.append(PKModel("pk_1c_oral", sample_1, _pk_one_comp_oral_conc, mask_for(["ka","CL","V"])))

    # 2) 1c oral + lag
    def sample_2(r):
        return {
            "ka":   apply_iiv_lognormal(r.uniform(0.1, 3.0), omega_iiv, r),
            "CL":   apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "V":    apply_iiv_lognormal(r.uniform(*V_rng),  omega_iiv, r),
            "Tlag": r.uniform(0.25, 6.0),
        }
    models.append(PKModel("pk_1c_oral_lag", sample_2, _pk_one_comp_oral_lag_conc, mask_for(["ka","CL","V","Tlag"])))

    # 3) 1c IV + Michaelis-Menten elimination
    def sample_3(r):
        return {
            "CL":   apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "V":    apply_iiv_lognormal(r.uniform(*V_rng),  omega_iiv, r),
            # <<< CHANGED >>> lower Vmax range to avoid too-aggressive nonlinear elimination with small V
             "Vmax": apply_iiv_lognormal(r.uniform(0.05, 2.5), omega_iiv, r),
            "Km":   apply_iiv_lognormal(r.uniform(1.0, 20.0), omega_iiv, r),
        }
    models.append(PKModel("pk_1c_iv_mm", sample_3, _pk_one_comp_iv_mm_conc, mask_for(["CL","V","Vmax","Km"])))

    # 4) 2c IV bolus
    def sample_4(r):
        return {
            "CL": apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "Vc": apply_iiv_lognormal(r.uniform(*Vc_rng), omega_iiv, r),
            "Vp": apply_iiv_lognormal(r.uniform(*Vp_rng), omega_iiv, r),
            "Q":  apply_iiv_lognormal(r.uniform(*Q_rng),  omega_iiv, r),
        }
    models.append(PKModel(
        "pk_2c_iv",
        sample_4,
        lambda t,d,p,r: _pk_two_comp_iv_conc(t,d,p,r,infusion=False),
        mask_for(["CL","Vc","Vp","Q"])
    ))

    # 5) 2c IV infusion (each dose starts infusion over tau)
    def sample_5(r):
        p = sample_4(r)
        p["tau"] = r.uniform(0.5, 8.0)
        return p
    models.append(PKModel(
        "pk_2c_iv_infusion",
        sample_5,
        lambda t,d,p,r: _pk_two_comp_iv_conc(t,d,p,r,infusion=True),
        mask_for(["CL","Vc","Vp","Q","tau"])
    ))

    # 6) 2c oral
    def sample_6(r):
        return {
            "ka": apply_iiv_lognormal(r.uniform(0.1, 3.0), omega_iiv, r),
            "CL": apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "Vc": apply_iiv_lognormal(r.uniform(*Vc_rng), omega_iiv, r),
            "Vp": apply_iiv_lognormal(r.uniform(*Vp_rng), omega_iiv, r),
            "Q":  apply_iiv_lognormal(r.uniform(*Q_rng),  omega_iiv, r),
        }
    models.append(PKModel("pk_2c_oral", sample_6, _pk_two_comp_oral_conc, mask_for(["ka","CL","Vc","Vp","Q"])))

    # 7) 1c oral transit absorption (n_tr fixed)
    def sample_7(r):
        return {
            "ktr": apply_iiv_lognormal(r.uniform(0.05, 2.5), omega_iiv, r),
            "CL":  apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "V":   apply_iiv_lognormal(r.uniform(*V_rng),  omega_iiv, r),
        }
    models.append(PKModel("pk_1c_transit_abs", sample_7, _pk_one_comp_transit_abs_conc, mask_for(["ktr","CL","V"])))

    # 8) 3c IV bolus
    def sample_8(r):
        return {
            "CL":  apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "Vc":  apply_iiv_lognormal(r.uniform(*Vc_rng), omega_iiv, r),
            "Vp":  apply_iiv_lognormal(r.uniform(*Vp_rng), omega_iiv, r),
            "Q":   apply_iiv_lognormal(r.uniform(*Q_rng),  omega_iiv, r),
            "Vp2": apply_iiv_lognormal(r.uniform(1.0, 15.0), omega_iiv, r),
            "Q2":  apply_iiv_lognormal(r.uniform(0.05, 1.5), omega_iiv, r),
        }
    models.append(PKModel("pk_3c_iv", sample_8, _pk_three_comp_iv_conc, mask_for(["CL","Vc","Vp","Q","Vp2","Q2"])))

    # 9) 1c oral flip-flop (very small ka)
    def sample_9(r):
        return {
            "ka": apply_iiv_lognormal(r.uniform(0.01, 0.20), omega_iiv, r),
            "CL": apply_iiv_lognormal(r.uniform(*CL_rng), omega_iiv, r),
            "V":  apply_iiv_lognormal(r.uniform(*V_rng),  omega_iiv, r),
        }
    models.append(PKModel("pk_1c_oral_flipflop", sample_9, _pk_one_comp_oral_conc, mask_for(["ka","CL","V"])))

    return models

# =========================
# 5) PD models (~10)
# =========================
PD_PARAM_KEYS = ["R0","Emax","EC50","Hill","SLOPE","Imax","IC50","ke0","kin","kout","ktr"]
PD_PARAM_INDEX = {k:i for i,k in enumerate(PD_PARAM_KEYS)}
PD_PARAM_DIM = len(PD_PARAM_KEYS)

@dataclass
class PDModel:
    name: str
    sample_params: Callable[[np.random.Generator], Dict[str, float]]
    simulate_response: Callable[[np.ndarray, np.ndarray, Dict[str, float], np.random.Generator], np.ndarray]
    param_mask: np.ndarray

def pd_param_vector(params: Dict[str,float]) -> np.ndarray:
    v = np.zeros((PD_PARAM_DIM,), dtype=float)
    for k,val in params.items():
        if k in PD_PARAM_INDEX:
            v[PD_PARAM_INDEX[k]] = float(val)
    return v

def _pd_direct_linear(times, conc, p, rng):
    R0, S = p["R0"], p["SLOPE"]
    conc = np.clip(conc, 0.0, None)
    return np.clip(R0 + S * conc, 0.0, None)

def _pd_direct_emax(times, conc, p, rng):
    R0, Emax, EC50 = p["R0"], p["Emax"], p["EC50"]
    conc = np.clip(conc, 0.0, None)
    return np.clip(R0 + Emax * conc / (EC50 + conc + 1e-12), 0.0, None)

def _pd_direct_sigmoid(times, conc, p, rng):
    R0, Emax, EC50, H = p["R0"], p["Emax"], p["EC50"], p["Hill"]

    conc = np.clip(conc, 0.0, None)
    H = max(float(H), 1e-3)

    concH = np.exp(H * np.log(conc + 1e-12))
    EC50H = float(np.exp(H * np.log(EC50 + 1e-12)))

    R = R0 + Emax * concH / (EC50H + concH + 1e-12)
    return np.clip(R, 0.0, None)

def _pd_direct_inhib_emax(times, conc, p, rng):
    R0, Imax, IC50 = p["R0"], p["Imax"], p["IC50"]
    conc = np.clip(conc, 0.0, None)
    inhib = Imax * conc / (IC50 + conc + 1e-12)
    return np.clip(R0 * (1.0 - inhib), 0.0, None)

def _pd_effect_compartment_emax(times, conc, p, rng):
    R0, Emax, EC50, ke0 = p["R0"], p["Emax"], p["EC50"], p["ke0"]
    times = np.array(times, dtype=float)
    conc = np.clip(np.array(conc, dtype=float), 0.0, None)

    def odefun(t, y):
        C = np.interp(t, times, conc)
        Ce = y[0]
        return [ke0*(C - Ce)]

    sol = solve_ivp(lambda t,y: odefun(t,y), (times[0], times[-1]), [0.0], t_eval=times, rtol=1e-6, atol=1e-9)
    Ce = sol.y[0]
    return np.clip(R0 + Emax * Ce / (EC50 + Ce + 1e-12), 0.0, None)

# Indirect response family
def _pd_indirect_inhib_kin(times, conc, p, rng):
    kin, kout, IC50, R0 = p["kin"], p["kout"], p["IC50"], p["R0"]
    times = np.array(times, dtype=float)
    conc = np.clip(np.array(conc, dtype=float), 0.0, None)

    def odefun(t, y):
        C = np.interp(t, times, conc)
        inh = C/(IC50 + C + 1e-12)
        R = y[0]
        return [kin*(1.0 - inh) - kout*R]

    sol = solve_ivp(odefun, (times[0], times[-1]), [R0], t_eval=times, rtol=1e-6, atol=1e-9)
    return np.clip(sol.y[0], 0.0, None)

def _pd_indirect_stim_kin(times, conc, p, rng):
    kin, kout, EC50, Emax, R0 = p["kin"], p["kout"], p["EC50"], p["Emax"], p["R0"]
    times = np.array(times, dtype=float)
    conc = np.clip(np.array(conc, dtype=float), 0.0, None)

    def odefun(t, y):
        C = np.interp(t, times, conc)
        stim = Emax*C/(EC50 + C + 1e-12)
        R = y[0]
        return [kin*(1.0 + stim) - kout*R]

    sol = solve_ivp(odefun, (times[0], times[-1]), [R0], t_eval=times, rtol=1e-6, atol=1e-9)
    return np.clip(sol.y[0], 0.0, None)

def _pd_indirect_inhib_kout(times, conc, p, rng):
    kin, kout, IC50, Imax, R0 = p["kin"], p["kout"], p["IC50"], p["Imax"], p["R0"]
    times = np.array(times, dtype=float)
    conc = np.clip(np.array(conc, dtype=float), 0.0, None)

    def odefun(t, y):
        C = np.interp(t, times, conc)
        inh = Imax*C/(IC50 + C + 1e-12)
        R = y[0]
        return [kin - kout*(1.0 - inh)*R]

    sol = solve_ivp(odefun, (times[0], times[-1]), [R0], t_eval=times, rtol=1e-6, atol=1e-9)
    return np.clip(sol.y[0], 0.0, None)

def _pd_indirect_stim_kout(times, conc, p, rng):
    kin, kout, EC50, Emax, R0 = p["kin"], p["kout"], p["EC50"], p["Emax"], p["R0"]
    times = np.array(times, dtype=float)
    conc = np.clip(np.array(conc, dtype=float), 0.0, None)

    def odefun(t, y):
        C = np.interp(t, times, conc)
        stim = Emax*C/(EC50 + C + 1e-12)
        R = y[0]
        return [kin - kout*(1.0 + stim)*R]

    sol = solve_ivp(odefun, (times[0], times[-1]), [R0], t_eval=times, rtol=1e-6, atol=1e-9)
    return np.clip(sol.y[0], 0.0, None)

def _pd_transit_delay(times, conc, p, rng, n_tr=3):
    R0, ktr, Emax, EC50 = p["R0"], p["ktr"], p["Emax"], p["EC50"]
    times = np.array(times, dtype=float)
    conc = np.clip(np.array(conc, dtype=float), 0.0, None)

    def odefun(t, y):
        C = np.interp(t, times, conc)
        stim = Emax*C/(EC50 + C + 1e-12)
        dy = np.zeros_like(y)
        dy[0] = ktr*((R0*(1.0+stim)) - y[0])
        for i in range(1, n_tr):
            dy[i] = ktr*(y[i-1] - y[i])
        return dy

    y0 = np.ones((n_tr,), dtype=float)*R0
    sol = solve_ivp(odefun, (times[0], times[-1]), y0, t_eval=times, rtol=1e-6, atol=1e-9)
    return np.clip(sol.y[n_tr-1], 0.0, None)

def build_pd_models(rng: np.random.Generator, omega_iiv=0.20) -> List[PDModel]:
    def mask_for(keys_used: List[str]) -> np.ndarray:
        m = np.zeros((PD_PARAM_DIM,), dtype=float)
        for k in keys_used:
            m[PD_PARAM_INDEX[k]] = 1.0
        return m

    models: List[PDModel] = []

    # --- Scale-tuned PD priors for biomarker ~ 5–15 ---
    R0_rng = (6.0, 14.0)

    # 0) direct linear
    def s0(r):
        return {"R0": r.uniform(*R0_rng),
            "SLOPE": apply_iiv_lognormal(r.uniform(0.002, 0.08), omega_iiv, r)}
    
    models.append(PDModel("pd_direct_linear", s0, _pd_direct_linear, mask_for(["R0","SLOPE"])))

    # 1) direct Emax
    def s1(r):
        return {"R0": r.uniform(*R0_rng),
            "Emax": apply_iiv_lognormal(r.uniform(0.05, 1.8), omega_iiv, r),
            "EC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r)}
    models.append(PDModel("pd_direct_emax", s1, _pd_direct_emax, mask_for(["R0","Emax","EC50"])))

    # 2) direct sigmoid Emax
    def s2(r):
        p = s1(r)
        p["Hill"] = apply_iiv_lognormal(r.uniform(0.8, 3.0), omega_iiv, r)
        return p
    models.append(PDModel("pd_direct_sigmoid", s2, _pd_direct_sigmoid, mask_for(["R0","Emax","EC50","Hill"])))

    # 3) direct inhibitory Emax
    def s3(r):
        return {"R0": r.uniform(*R0_rng),
                "Imax": apply_iiv_lognormal(r.uniform(0.05, 0.35), omega_iiv, r),
                "IC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r)}

    models.append(PDModel("pd_direct_inhib_emax", s3, _pd_direct_inhib_emax, mask_for(["R0","Imax","IC50"])))

    # 4) effect compartment Emax
    def s4(r):
        p = s1(r)
        p["ke0"] = apply_iiv_lognormal(r.uniform(0.01, 1.0), omega_iiv, r)
        return p
    models.append(PDModel("pd_effect_comp_emax", s4, _pd_effect_compartment_emax, mask_for(["R0","Emax","EC50","ke0"])))

    # 5) indirect inhib kin
    def s5(r):
        R0 = r.uniform(*R0_rng)
        kout = apply_iiv_lognormal(r.uniform(0.03, 0.18), omega_iiv, r)
        kin = R0 * kout
        return {"kin": kin, "kout": kout,
                "IC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r),
                "R0": R0}
    models.append(PDModel("pd_indirect_inhib_kin", s5, _pd_indirect_inhib_kin, mask_for(["kin","kout","IC50","R0"])))

    # 6) indirect stim kin
    def s6(r):
        R0 = r.uniform(*R0_rng)
        kout = apply_iiv_lognormal(r.uniform(0.05, 0.25), omega_iiv, r)
        kin = R0 * kout
        return {
            "kin": kin,
            "kout": kout,
            # <<< CHANGED >>> smaller Emax + larger EC50 to keep biomarker in range
            "Emax": apply_iiv_lognormal(r.uniform(0.02, 0.45), omega_iiv, r),
            "EC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r),
            "R0": R0
        }
    models.append(PDModel("pd_indirect_stim_kin", s6, _pd_indirect_stim_kin, mask_for(["kin","kout","Emax","EC50","R0"])))

    # 7) indirect inhib kout
    def s7(r):
        R0 = r.uniform(*R0_rng)
        kout = apply_iiv_lognormal(r.uniform(0.05, 0.25), omega_iiv, r)
        kin = R0 * kout
        return {
            "kin": kin,
            "kout": kout,
            "Imax": apply_iiv_lognormal(r.uniform(0.05, 0.35), omega_iiv, r),
            "IC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r),
            "R0": R0
        }
    models.append(PDModel("pd_indirect_inhib_kout", s7, _pd_indirect_inhib_kout, mask_for(["kin","kout","Imax","IC50","R0"])))

    # 8) indirect stim kout
    def s8(r):
        R0 = r.uniform(*R0_rng)
        kout = apply_iiv_lognormal(r.uniform(0.05, 0.25), omega_iiv, r)
        kin = R0 * kout
        return {
            "kin": kin,
            "kout": kout,
            "Emax": apply_iiv_lognormal(r.uniform(0.02, 0.45), omega_iiv, r),
            "EC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r),
            "R0": R0
        }
    models.append(PDModel("pd_indirect_stim_kout", s8, _pd_indirect_stim_kout, mask_for(["kin","kout","Emax","EC50","R0"])))

    # 9) transit delay
    def s9(r):
        return {"R0": r.uniform(*R0_rng),
            "ktr": apply_iiv_lognormal(r.uniform(0.01, 0.15), omega_iiv, r),
            "Emax": apply_iiv_lognormal(r.uniform(0.02, 0.45), omega_iiv, r),
            "EC50": apply_iiv_lognormal(r.uniform(8.0, 30.0), omega_iiv, r)}
    models.append(PDModel("pd_transit_delay", s9, _pd_transit_delay, mask_for(["R0","ktr","Emax","EC50"])))

    return models

# =========================
# 6) Dataset generation (PK & PD separated; matches your [2,L] formats)
# =========================
def generate_pk_sample(pk_models: List[PKModel],
                       pk_times: np.ndarray,
                       rng: np.random.Generator,
                       conc_noise=(0.15, 0.02)) -> Dict:
    dose_vec = generate_dose_vector(pk_times, rng)
    m_id = int(rng.integers(0, len(pk_models)))
    model = pk_models[m_id]
    params = model.sample_params(rng)

    conc_true = model.simulate_conc(pk_times, dose_vec, params, rng)
    conc_true = np.clip(conc_true, 0.0, None)
    conc_meas = add_residual_error(conc_true, conc_noise[0], conc_noise[1], rng)

    traj = np.stack([dose_vec, conc_meas], axis=0)  # (2,39)

    return {
        "pk_model_id": m_id,
        "pk_model_name": model.name,
        "pk_params": params,
        "pk_param_vec": pk_param_vector(params),
        "pk_param_mask": model.param_mask.copy(),
        "pk_traj_2xT": traj,
        "pk_conc_true": conc_true,
    }

def generate_pd_sample(pd_models: List[PDModel],
                       pd_times: np.ndarray,
                       conc_at_pd_times: np.ndarray,
                       rng: np.random.Generator,
                       conc_noise=(0.10, 0.01),
                       biom_noise=(0.10, 0.05)) -> Dict:
    m_id = int(rng.integers(0, len(pd_models)))
    model = pd_models[m_id]
    params = model.sample_params(rng)

    # <<< ADDED >>> (extra safety)
    conc_at_pd_times = np.clip(np.array(conc_at_pd_times, dtype=float), 0.0, None)

    conc_meas = add_residual_error(conc_at_pd_times, conc_noise[0], conc_noise[1], rng)

    biom_true = model.simulate_response(pd_times, conc_at_pd_times, params, rng)
    biom_meas = add_residual_error(biom_true, biom_noise[0], biom_noise[1], rng)

    traj = np.stack([conc_meas, biom_meas], axis=0)  # (2,25)

    return {
        "pd_model_id": m_id,
        "pd_model_name": model.name,
        "pd_params": params,
        "pd_param_vec": pd_param_vector(params),
        "pd_param_mask": model.param_mask.copy(),
        "pd_traj_2xT": traj,
        "pd_biom_true": biom_true,
    }

def generate_pk_pd_datasets(
    N: int,
    seed: int = 0,
    pk_times: np.ndarray = PK_TIMES,
    pd_times: np.ndarray = PD_TIMES,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pk_models = build_pk_models(rng)
    pd_models = build_pd_models(rng)

    X_pk = np.zeros((N, 2, len(pk_times)), dtype=float)
    y_pk_cls = np.zeros((N,), dtype=int)
    y_pk_reg = np.zeros((N, PK_PARAM_DIM), dtype=float)
    y_pk_mask = np.zeros((N, PK_PARAM_DIM), dtype=float)

    X_pd = np.zeros((N, 2, len(pd_times)), dtype=float)
    y_pd_cls = np.zeros((N,), dtype=int)
    y_pd_reg = np.zeros((N, PD_PARAM_DIM), dtype=float)
    y_pd_mask = np.zeros((N, PD_PARAM_DIM), dtype=float)

    for i in range(N):
        pk = generate_pk_sample(pk_models, pk_times, rng)
        conc_pd_true = np.interp(pd_times, pk_times, pk["pk_conc_true"])
        pd = generate_pd_sample(pd_models, pd_times, conc_pd_true, rng)

        X_pk[i] = pk["pk_traj_2xT"]
        y_pk_cls[i] = pk["pk_model_id"]
        y_pk_reg[i] = pk["pk_param_vec"]
        y_pk_mask[i] = pk["pk_param_mask"]

        X_pd[i] = pd["pd_traj_2xT"]
        y_pd_cls[i] = pd["pd_model_id"]
        y_pd_reg[i] = pd["pd_param_vec"]
        y_pd_mask[i] = pd["pd_param_mask"]

    return {
        "X_pk_2xT": X_pk, "y_pk_cls": y_pk_cls, "y_pk_reg": y_pk_reg, "y_pk_mask": y_pk_mask,
        "X_pd_2xT": X_pd, "y_pd_cls": y_pd_cls, "y_pd_reg": y_pd_reg, "y_pd_mask": y_pd_mask,
        "pk_times": pk_times, "pd_times": pd_times,
    }




