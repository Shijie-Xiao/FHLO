"""Initialize (r0, k) for the FHLO wind model from IBTrACS wind radii.

Paper section 3e: "we take the initial analysis of the maximum extent of the
34-, 50-, and 64-kt winds in each quadrant [CARQ lines of the ATCF a-decks],
and find the corresponding value of r0 that allows the modeled asymmetric
wind field to best match the analysis... we find r0 and k such that the full
asymmetric wind field best matches the analysis radii in each quadrant."

Here the analysis comes from IBTrACS LAST observarion (USA_R34/50/64_*_NE/
SE/SW/NW, nautical miles) at the forecast start time. If no radii exist at
all -> r0 = 700 km, k = 1 (paper default).
"""

import time

import numpy as np

from .wind_field import (K_DEFAULT, R0_DEFAULT_M, apply_shape_k,
                         get_profile, quad_wind_speeds)

NM_TO_M = 1852.0
QUADS = ['NE', 'SE', 'SW', 'NW']


def load_ibtracs_radii(ibtracs_csv, sid, time_iso):
    """Quadrant radii [m] dict at (or nearest before) time_iso.

    Returns {'R34': [ne,se,sw,nw], 'R50': ..., 'R64': ..., 'vmax_ms': ...}
    with np.nan where the analysis has no value (0 nm means "not occurring"
    in the a-deck and is treated as no constraint).
    """
    import pandas as pd
    df = pd.read_csv(ibtracs_csv, low_memory=False)
    st = df[df['SID'] == sid].copy()
    if not len(st):
        raise ValueError(f'SID {sid} not in {ibtracs_csv}')
    st['t'] = pd.to_datetime(st['ISO_TIME'])
    t0 = pd.Timestamp(time_iso)
    st = st[st['t'] <= t0]
    if not len(st):
        st = df[df['SID'] == sid].copy()
        st['t'] = pd.to_datetime(st['ISO_TIME'])
    row = st.iloc[-1]

    out = {}
    for thr in ('R34', 'R50', 'R64'):
        vals = []
        for q in QUADS:
            v = row.get(f'USA_{thr}_{q}', np.nan)
            v = pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0]
            v = np.nan if (not np.isfinite(v) or v <= 0) else v * NM_TO_M
            vals.append(v)
        out[thr] = np.array(vals)
    out['vmax_ms'] = float(pd.to_numeric(pd.Series([row.get('USA_WIND')]),
                                         errors='coerce').iloc[0]) * 0.514444
    out['lat'] = float(row['LAT'])
    out['lon'] = float(row['LON'])
    out['time'] = str(row['ISO_TIME'])
    return out


def model_quadrant_radii(v_axisym_ms, r0_m, k, lat, ut_ms, vt_ms,
                          u_shr, v_shr, thresholds_ms):
    """Model's max extent of each threshold wind in each quadrant [m]."""
    rr, vv, rmax = get_profile(v_axisym_ms, r0_m, lat)
    if rr is None:
        return {t: np.full(4, np.nan) for t in thresholds_ms}
    vvk = apply_shape_k(rr, vv, k, rmax)
    th, spd = quad_wind_speeds(rr, vvk, ut_ms, vt_ms, u_shr, v_shr, lat)
    # position azimuth CCW from east: NE=(0,90] SE=(90,180] SW=(180,270] NW=(270,360]
    quad_idx = {'NE': (th > 0) & (th <= 90), 'SE': (th > 90) & (th <= 180),
                'SW': (th > 180) & (th <= 270), 'NW': (th > 270)}
    out = {}
    for t_ms in thresholds_ms:
        ext = np.full(4, np.nan)
        reach = spd.max(axis=0) >= t_ms      # any azimuth reaching threshold
        for iq, q in enumerate(QUADS):
            s = spd[quad_idx[q], :]
            inside = (s >= t_ms).any(axis=0)
            idx = np.where(inside & reach)[0]
            if len(idx):
                ext[iq] = rr[idx[0]]
        out[t_ms] = ext
    return out


def _cost(r0_m, k, v_axisym_ms, obs_radii, lat, ut_ms, vt_ms, u_shr, v_shr,
          thresholds_ms):
    try:
        mod = model_quadrant_radii(v_axisym_ms, r0_m, k, lat, ut_ms, vt_ms,
                                   u_shr, v_shr, thresholds_ms)
    except Exception:
        return 1e9
    c = 0.0
    n = 0
    for t_ms in thresholds_ms:
        obs = obs_radii.get(t_ms)
        if obs is None:
            continue
        o = np.asarray(obs, float)
        m = mod[t_ms]
        ok = np.isfinite(o) & (o > 0)
        m_fin = np.isfinite(m)
        if ok.sum() == 0:
            continue            # threshold not observed anywhere
        # penalize model extent where observation is absent
        miss = ok & ~m_fin
        if miss.any():
            c += 0.5 * miss.sum()      # model misses an observed quadrant
        hit = ok & m_fin
        if hit.sum():
            c += float(np.nansum(((m[hit] - o[hit]) / o[hit])**2))
            n += int(hit.sum())
        over = (~ok) & m_fin
        if over.any():
            c += 0.25 * float(over.sum())
    if n == 0:
        return 1e9
    return c / n


def fit_r0_k(radii_dict, ut_ms, vt_ms, u_shr, v_shr,
             r0_grid_km=None, k_grid=None, verbose=False):
    """Jointly fit (r0, k, V_axisym) to the wind-radii + Vmax analysis.

    Paper section 3e: find r0 and k such that the full asymmetric wind field
    best matches the analysis radii in each quadrant (the analysis intensity
    constrains V_axisym simultaneously -- 34/50/64-kt quadrant radii plus
    Vmax are all part of the "analysis").
    """
    thresholds_ms = []
    obs = {}
    for thr_kt, key in ((34, 'R34'), (50, 'R50'), (64, 'R64')):
        v = radii_dict.get(key)
        if v is not None and np.isfinite(np.asarray(v, float)).any():
            thresholds_ms.append(thr_kt * 0.514444)
            obs[thresholds_ms[-1]] = v
    lat = radii_dict['lat']
    v_surface = radii_dict['vmax_ms']

    if not thresholds_ms:
        if verbose:
            print('[init_radii] no radii analysis -> r0=700 km, k=1')
        return R0_DEFAULT_M, K_DEFAULT, v_surface

    def _eval_profile(rr, vv, rmax, k):
        """One profile + asymmetry -> (vmax_mod, {thr: quad radii m})."""
        vvk = apply_shape_k(rr, vv, k, rmax)
        th, spd = quad_wind_speeds(rr, vvk, ut_ms, vt_ms, u_shr, v_shr, lat)
        vmax_mod = float(np.nanmax(spd))
        quad_idx = {'NE': (th > 0) & (th <= 90), 'SE': (th > 90) & (th <= 180),
                    'SW': (th > 180) & (th <= 270), 'NW': (th > 270)}
        out = {}
        for t_ms in thresholds_ms:
            ext = np.full(4, np.nan)
            for iq, q in enumerate(QUADS):
                inside = (spd[quad_idx[q], :] >= t_ms).any(axis=0)
                idx = np.where(inside)[0]
                if len(idx):
                    ext[iq] = rr[idx[0]]
            out[t_ms] = ext
        return vmax_mod, out

    def _cost(vmax_mod, mod_radii):
        c = ((vmax_mod - v_surface) / max(v_surface, 1.0))**2
        n = 1
        for t_ms in thresholds_ms:
            o = np.asarray(obs[t_ms], float)
            m = mod_radii[t_ms]
            ok = np.isfinite(o) & (o > 0)
            hit = ok & np.isfinite(m)
            if hit.any():
                c += float(np.nansum(((m[hit] - o[hit]) / o[hit])**2))
                n += int(hit.sum())
            miss = ok & ~np.isfinite(m)
            if miss.any():
                c += 0.5 * miss.sum()
            over = (~ok) & np.isfinite(m)
            if over.any():
                c += 0.25 * over.sum()
        return c / n

    def _search(r0_list, k_list, v_list, tag):
        """Evaluate grid with shared per-(r0,V) profile cache. Returns best."""
        best = [1e18, R0_DEFAULT_M, K_DEFAULT, v_surface, np.nan]
        n_tot = len(r0_list) * len(k_list) * len(v_list)
        n_done = 0
        t_start = time.time()
        for r0_km in r0_list:
            r0_m = r0_km * 1000
            for v_ax in v_list:
                rr, vv, rmax = get_profile(float(v_ax), r0_m, lat)
                if rr is None:
                    n_done += len(k_list)
                    continue
                for k in k_list:
                    vmax_mod, mod_radii = _eval_profile(rr, vv, rmax, float(k))
                    c = _cost(vmax_mod, mod_radii)
                    n_done += 1
                    if c < best[0]:
                        best = [c, r0_m, float(k), float(v_ax), vmax_mod]
            if verbose:
                el = time.time() - t_start
                done_frac = n_done / max(n_tot, 1)
                eta = el / max(n_done, 1) * (n_tot - n_done)
                print(f'[init_radii:{tag}] {n_done}/{n_tot} '
                      f'({done_frac*100:.0f}%, {el:.0f}s, eta {eta:.0f}s) '
                      f'r0={r0_km:.0f}km best cost={best[0]:.5f} '
                      f'k={best[2]:.2f} Vax={best[3]:.1f} '
                      f'Vmax_mod={best[4]:.1f}', flush=True)
        return best

    # stage 1: coarse grid
    r0_coarse = (np.arange(400, 3001, 200) if r0_grid_km is None
                 else np.asarray(r0_grid_km, float))
    k_coarse = (np.arange(0.8, 1.31, 0.1) if k_grid is None
                else np.asarray(k_grid, float))
    v_lo, v_hi = max(3.0, 0.55 * v_surface), 1.6 * v_surface
    v_coarse = np.linspace(v_lo, v_hi, 8)
    cost, r0_m, k, v_ax, vm = _search(r0_coarse, k_coarse, v_coarse, 'coarse')

    # stage 2: refine around the coarse best (skip if user pinned the grids)
    if r0_grid_km is None and k_grid is None:
        r0_fine = np.arange(max(300, r0_m / 1000 - 150),
                            min(3200, r0_m / 1000 + 150) + 1, 50)
        k_fine = np.arange(max(0.75, k - 0.08), min(1.35, k + 0.08) + 1e-9, 0.02)
        v_fine = np.linspace(max(3.0, v_ax - 2.5), v_ax + 2.5, 7)
        c2, r0_m2, k2, v_ax2, vm2 = _search(r0_fine, k_fine, v_fine, 'fine')
        if c2 <= cost:
            cost, r0_m, k, v_ax, vm = c2, r0_m2, k2, v_ax2, vm2

    if not np.isfinite(cost) or cost >= 1e9:
        if verbose:
            print('[init_radii] fit failed -> defaults r0=700 km k=1')
        return R0_DEFAULT_M, K_DEFAULT, v_surface
    if verbose:
        print(f'[init_radii] FIT r0={r0_m/1000:.0f} km k={k:.2f} '
              f'V_axisym={v_ax:.1f} (mod Vmax {vm:.1f} vs obs {v_surface:.1f})',
              flush=True)
    return r0_m, k, v_ax
