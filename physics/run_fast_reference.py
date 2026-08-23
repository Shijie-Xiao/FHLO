#!/usr/bin/env python3
"""Run FAST reference ODE on prepared dataset pkl files (single-track, self-contained).

Pipeline position: run.py stage 3 (after prep/prepare_complete_training_data.py).

Logic (identical to ODE/run_fast_reference.py):
  - 48h init with Vtarget (V_axisym from obs inversion)
  - F forcing accumulation during init, smoothed window=12
  - Forecast with F_init_end * exp(-(lead/24)^2) decay
  - 4 sub-steps per hour
  - axi_to_max_wind for V_axisym -> V_max conversion
Removed vs the old version: basin discovery loops, legacy config plumbing,
duplicate trailing helper copies. This file only needs numpy/pandas/matplotlib.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

MS_TO_KNOTS = 1.94384
XS_NAN_FALLBACK = 1e-5
STEP_SIZE = 1.0 / 4.0
CHI_D = 4.0
CHI_MULTIPLIER = 5
VMAX_START_MS = 45 * 0.514444
INIT_HOURS = 48
T0_DECAY_HOURS = 24.0
Cd_CONST = 1.2e-3
H_BL = 1400.0


# ---------- Scalar helpers ----------

def _safe(x, default):
    return float(default) if np.isnan(x) else float(x)


def _coeff_from_cd(cd, h_bl):
    return 0.5 * float(cd) / float(h_bl) * 3600.0


def _median_filter(arr, size=3):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    padded = np.pad(arr, 1, mode='edge')
    return np.array([np.median(padded[i:i + size]) for i in range(n)], dtype=np.float64)


def _chi_calibrated(chi_val):
    chi_val = np.asarray(chi_val, dtype=np.float64)
    chi_val = np.maximum(np.nan_to_num(chi_val, nan=1e-10), 1e-10)
    return np.clip(chi_val * CHI_MULTIPLIER, 0.0, CHI_D)


# ---------- FAST ODE ----------

def fast_step(xs, V, m, alpha, beta, gamma, vp, coeff, dV_extra=0.0):
    """One Heun (2nd-order) integration step split into 4 sub-steps by caller."""
    vp = _safe(vp, 0.0); alpha = _safe(alpha, 1.0)
    beta = _safe(beta, 0.57); gamma = _safe(gamma, 0.43); xs = _safe(xs, 0.0)
    dV = coeff * (alpha * beta * vp**2 * m**3 - (1.0 - gamma * m**3) * V**2) + dV_extra
    dm = coeff * ((1.0 - m) * V - xs * m)
    V_mid = max(0.0, min(200.0, V + dV * STEP_SIZE))
    m_mid = max(0.0, min(1.0, m + dm * STEP_SIZE))
    dV2 = coeff * (alpha * beta * vp**2 * m_mid**3 - (1.0 - gamma * m_mid**3) * V_mid**2) + dV_extra
    dm2 = coeff * ((1.0 - m_mid) * V_mid - xs * m_mid)
    return (max(0.0, min(200.0, V + 0.5*(dV+dV2)*STEP_SIZE)),
            max(0.0, min(1.0, m + 0.5*(dm+dm2)*STEP_SIZE)))


def calculate_m0(v, dv_dt, alpha, beta, gamma, vp, coeff):
    vp = _safe(vp, 0.0); alpha = _safe(alpha, 1.0)
    beta = _safe(beta, 0.57); gamma = _safe(gamma, 0.43)
    num = dv_dt / (coeff + 1e-12) + v**2
    den = alpha * beta * vp**2 + gamma * v**2
    return float(np.clip(np.power(np.clip(num / (den + 1e-8), 0, None), 1.0/3.0), 0.01, 1.0))


# ---------- axi_to_max_wind ----------

def _axi_to_max_wind_single(v_axi, s, env, ut, vt, lat):
    v_axi = float(v_axi); s = float(np.nan_to_num(s, nan=0.0))
    env = np.atleast_1d(env).flatten()
    if len(env) < 4:
        env = np.zeros(4)
    G = min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((abs(float(lat)) - 35.0) / 10.0)))
    u_shr = env[0] - env[2]; v_shr = env[1] - env[3]
    shear_mag = np.sqrt(u_shr**2 + v_shr**2 + 1e-12)
    has = not (np.isnan(u_shr) or np.isnan(v_shr) or shear_mag < 1e-6)
    u_dir = (u_shr/shear_mag) if has else 0.0
    v_dir = (v_shr/shear_mag) if has else 0.0
    sc = 0.1 * s * v_axi / 15.0
    U_inc = G * float(ut) + sc * u_dir; V_inc = G * float(vt) + sc * v_dir
    mag = np.sqrt(U_inc**2 + V_inc**2 + 1e-12)
    fac = min(1.0, (v_axi * 0.5) / mag) if mag > 1e-12 else 0.0
    th = np.arctan2(-U_inc, V_inc)
    ug = v_axi * (-np.sin(th)) + U_inc * fac
    vg = v_axi * np.cos(th) + V_inc * fac
    return float(np.sqrt(ug**2 + vg**2 + 1e-12))


def _invert_vmax_to_V_axisym(v_max_obs, s, env, ut, vt, lat, tol=0.01, max_iter=50):
    if np.isnan(v_max_obs) or v_max_obs <= 0:
        return np.nan
    v_lo, v_hi = 0.0, 200.0
    for _ in range(max_iter):
        v_mid = (v_lo + v_hi) * 0.5
        vm = _axi_to_max_wind_single(v_mid, s, env, ut, vt, lat)
        if abs(vm - float(v_max_obs)) < tol:
            return v_mid
        if vm < float(v_max_obs):
            v_lo = v_mid
        else:
            v_hi = v_mid
    return (v_lo + v_hi) * 0.5


def axi_to_max_wind(tc_v, s_ref, env_wnds, utran, vtran, lats):
    """Vectorized V_axisym -> V_max conversion (matches ODE exactly)."""
    G = np.minimum(1.0, 0.8 + 0.35 * (1.0 + np.tanh((np.abs(lats) - 35.0) / 10.0)))
    u_shr = env_wnds[:, 0] - env_wnds[:, 2]
    v_shr = env_wnds[:, 1] - env_wnds[:, 3]
    shear_mag = np.sqrt(u_shr**2 + v_shr**2 + 1e-12)
    has = ~(np.isnan(u_shr) | np.isnan(v_shr) | (shear_mag < 1e-6))
    u_dir = np.where(has, u_shr / shear_mag, 0.0)
    v_dir = np.where(has, v_shr / shear_mag, 0.0)
    s_safe = np.nan_to_num(s_ref, nan=0.0)
    sc = 0.1 * s_safe * tc_v / 15.0
    U_inc = G * utran + sc * u_dir; V_inc = G * vtran + sc * v_dir
    mag = np.sqrt(U_inc**2 + V_inc**2 + 1e-12)
    fac = np.minimum(1.0, (tc_v * 0.5) / mag)
    th = np.arctan2(-U_inc, V_inc)
    ug = tc_v * (-np.sin(th)) + U_inc * fac
    vg = tc_v * np.cos(th) + V_inc * fac
    return np.sqrt(ug**2 + vg**2 + 1e-12)


# ---------- Main FAST integration ----------

def run_fast_with_init(scalars, xs_ref, v_gt, env_wnds, utran, vtran, lats, s_ref, lons=None, data=None):
    T = scalars.shape[1]
    sc = np.array(scalars[0, :, :], dtype=np.float64)
    vp = sc[:, 3]; alpha = sc[:, 0]; beta = sc[:, 1]; gamma = sc[:, 2]
    xs = np.maximum(np.nan_to_num(np.array(xs_ref[0, :, 0], dtype=np.float64), nan=XS_NAN_FALLBACK), XS_NAN_FALLBACK)
    v_obz = np.array(v_gt[0, :, 0], dtype=np.float64)

    # Per-timestep Cd and BLH from pkl if available, otherwise constants
    cd_arr = np.full(T, Cd_CONST)
    blh_arr = np.full(T, H_BL)
    if data is not None and 'cd_ref' in data and data['cd_ref'] is not None:
        cd_raw = np.asarray(data['cd_ref']).flatten()[:T]
        cd_arr[:len(cd_raw)] = cd_raw
    if data is not None and 'blh_ref' in data and data['blh_ref'] is not None:
        blh_raw = np.asarray(data['blh_ref']).flatten()[:T]
        blh_arr[:len(blh_raw)] = blh_raw
    blh_arr = np.where(np.isfinite(blh_arr) & (blh_arr > 50), blh_arr, H_BL)
    cd_arr = np.where(np.isfinite(cd_arr) & (cd_arr > 1e-5), cd_arr, Cd_CONST)
    coeff_arr = np.array([_coeff_from_cd(cd_arr[t], blh_arr[t]) for t in range(T)])

    la = np.zeros(T)
    if lats is not None:
        a = np.asarray(lats).reshape(-1); la[:min(len(a), T)] = a[:T]

    ew = np.full((T, 4), np.nan)
    if env_wnds is not None:
        arr = np.asarray(env_wnds)
        if arr.ndim == 3:
            ew = np.array(arr[0, :T, :], dtype=np.float64)
        elif arr.ndim == 2:
            ew[:min(arr.shape[0], T)] = arr[:T]
    ut = np.zeros(T); vt = np.zeros(T)
    if utran is not None:
        a = np.asarray(utran).reshape(-1); ut[:min(len(a), T)] = a[:T]
    if vtran is not None:
        a = np.asarray(vtran).reshape(-1); vt[:min(len(a), T)] = a[:T]
    s_r = np.nan_to_num(np.asarray(s_ref).reshape(T, -1)[:, 0], nan=0.0) if s_ref is not None else np.zeros(T)

    # Vtarget = V_axisym(obs) via bisection inversion
    Vtarget = np.full(T, np.nan)
    for i in range(T):
        if np.isnan(v_obz[i]) or v_obz[i] <= 0:
            continue
        Vtarget[i] = _invert_vmax_to_V_axisym(v_obz[i], s_r[i], ew[i], ut[i], vt[i], la[i])

    # Find 45kts start, init period
    t_40 = next((i for i in range(T)
                 if not np.isnan(v_obz[i]) and v_obz[i] >= VMAX_START_MS), None)
    t_start = t_40 if t_40 is not None else 0
    t_init_start = t_start - INIT_HOURS
    if t_init_start < 0:
        t_start = INIT_HOURS; t_init_start = 0
    if t_start > T:
        t_start = T; t_init_start = max(0, T - INIT_HOURS)

    v_fast = np.full(T, np.nan); m_series = np.full(T, np.nan)
    if t_init_start >= T:
        return v_fast, np.full(T, np.nan), m_series

    v0 = float(Vtarget[t_init_start]) if not np.isnan(Vtarget[t_init_start]) else 5.0
    if v0 <= 0:
        v0 = 5.0
    V = np.float64(v0)

    dv_dt = 0.0
    if t_init_start + 1 < T and np.isfinite(Vtarget[t_init_start]) and np.isfinite(Vtarget[t_init_start+1]):
        dv_dt = Vtarget[t_init_start+1] - Vtarget[t_init_start]
    m = np.float64(np.clip(calculate_m0(v0, dv_dt, alpha[t_init_start], beta[t_init_start],
                                        gamma[t_init_start], vp[t_init_start], coeff_arr[t_init_start]), 0.01, 1.0))

    F_init_end = 0.0
    F_history = []

    for t in range(t_init_start, T):
        coeff_t = coeff_arr[t]
        if t < t_start:
            # Init phase: V tracks Vtarget, accumulate forcing F
            Vtar_t = float(Vtarget[t]) if not np.isnan(Vtarget[t]) else V
            Vtar_next = float(Vtarget[t+1]) if t+1 < T and not np.isnan(Vtarget[t+1]) else Vtar_t
            observed_accel = Vtar_next - Vtar_t
            with np.errstate(invalid='ignore', divide='ignore'):
                physics_rhs = coeff_t * (alpha[t] * beta[t] * vp[t]**2 * m**3
                                         - (1.0 - gamma[t] * m**3) * Vtar_t**2)
            F_t = observed_accel - (physics_rhs if np.isfinite(physics_rhs) else 0.0)
            F_history.append(F_t)
            if t == min(t_start, T) - 1:
                window = min(12, len(F_history))
                F_init_end = float(np.mean(F_history[-window:]))
            V = np.float64(Vtar_next)
            for _ in range(4):
                with np.errstate(invalid='ignore', divide='ignore'):
                    dm = coeff_t * ((1.0 - m) * Vtar_next - xs[t] * m)
                m = np.float64(np.clip(m + dm * STEP_SIZE, 0.01, 1.0))
        else:
            # Forecast phase: 4 sub-steps with decaying F
            lead_h = t - t_start
            decay = np.exp(-1.0 * (lead_h / T0_DECAY_HOURS) ** 2)
            dV_extra = F_init_end * decay
            for _ in range(4):
                V, m = fast_step(xs[t], V, m, alpha[t], beta[t], gamma[t], vp[t], coeff_t, dV_extra)
        v_fast[t] = float(V)
        m_series[t] = float(m)

    v_max = axi_to_max_wind(v_fast, s_r, ew, ut, vt, la)
    return v_fast, v_max, m_series


# ---------- Process one pkl ----------

def process_one_pkl(pkl_path, save_csv=True, save_plot=True, out_dir=None):
    """Run the FAST ODE on one *_dataset.pkl. Returns summary dict."""
    pkl_path = Path(pkl_path)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    scalars = np.array(data['scalars'], dtype=np.float64, copy=True)
    T = scalars.shape[1]
    scalars[0, :, 3] = _median_filter(scalars[0, :, 3], size=3)

    chi_cal = _chi_calibrated(data['chi_ref'])
    xs_ref = np.maximum(np.nan_to_num(chi_cal * data['s_ref'], nan=XS_NAN_FALLBACK), XS_NAN_FALLBACK)

    times = data.get('times')
    times = pd.to_datetime(np.asarray(times).ravel()[:T]) if times is not None \
        else pd.date_range(start='2000-01-01', periods=T, freq='h')

    v_fast_ms, v_max_ms, m_series = run_fast_with_init(
        scalars, xs_ref, data['v_gt'], data.get('env_wnds'), data.get('utran'),
        data.get('vtran'), data.get('lats'), data['s_ref'],
        lons=data.get('lons'), data=data,
    )
    v_obz_kts = np.array(data['v_gt'][0, :, 0], dtype=np.float32) * MS_TO_KNOTS
    vp_kts = np.array(scalars[0, :, 3], dtype=np.float32) * MS_TO_KNOTS
    v_max_kts = v_max_ms * MS_TO_KNOTS

    storm_name = data.get('hurricane', pkl_path.stem.replace('_dataset', ''))
    out_dir = Path(out_dir) if out_dir else pkl_path.parent

    if save_csv:
        df = pd.DataFrame({
            'step': np.arange(T), 'time': times,
            'vp_kts': vp_kts, 'v_obz_kts': v_obz_kts,
            'v_fast_kts': v_fast_ms * MS_TO_KNOTS, 'v_max_kts': v_max_kts, 'm': m_series,
        })
        csv_path = out_dir / 'fast_reference.csv'
        df.to_csv(csv_path, index=False)

    if save_plot:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(times, v_max_kts, label='v_max (FAST)', color='blue', linewidth=1.5, alpha=0.9)
        ax.plot(times, v_obz_kts, label='v_obz (obs)', color='black', linewidth=1.5, alpha=0.9)
        ax.plot(times, vp_kts, label='vp (potential intensity)', color='red', linewidth=1.5, alpha=0.9)
        ax.set_xlabel('Date'); ax.set_ylabel('Intensity (knots)'); ax.set_ylim(0, 200)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        fig.autofmt_xdate(); ax.set_title(storm_name)
        ax.legend(loc='best'); ax.grid(True, alpha=0.3, color='grey')
        fig.tight_layout()
        fig.savefig(out_dir / 'fast_reference.png', dpi=150, facecolor='white')
        plt.close(fig)

    valid = np.isfinite(v_max_kts) & np.isfinite(v_obz_kts)
    mae = float(np.mean(np.abs(v_max_kts[valid] - v_obz_kts[valid]))) if valid.any() else np.nan
    return {'v_max_kts': v_max_kts, 'v_obz_kts': v_obz_kts, 'times': times,
            'mae_kts': mae, 'storm': storm_name}


def main():
    p = argparse.ArgumentParser(description='FHLO FAST reference ODE (single-track)')
    p.add_argument('pkl', nargs='*', help='dataset pkl file(s); default: auto-discover')
    p.add_argument('--data_root', default='data/ibtracs')
    p.add_argument('--basin', default='NA')
    p.add_argument('--year', type=int, default=2024)
    p.add_argument('--no_csv', action='store_true')
    p.add_argument('--no_plot', action='store_true')
    args = p.parse_args()

    pkls = [Path(x) for x in args.pkl]
    if not pkls:
        root = Path(args.data_root) / args.basin / str(args.year)
        pkls = sorted(root.glob('*/*_dataset.pkl'))
    print(f'Found {len(pkls)} dataset pkl file(s)')

    for pkl_path in pkls:
        try:
            r = process_one_pkl(pkl_path, save_csv=not args.no_csv,
                                save_plot=not args.no_plot)
            print(f"[OK] {r['storm']} MAE={r['mae_kts']:.1f} kts")
        except Exception as e:
            print(f'[FAIL] {pkl_path}: {e}')


if __name__ == '__main__':
    main()
