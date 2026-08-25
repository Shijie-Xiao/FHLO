"""34/50/64-kt wind-speed exceedance probabilities (FHLO section 4d).

For each ensemble member and each time step, the full 2-D surface wind field
is evaluated on a fixed lat/lon grid; probabilities are the member fraction
exceeding each threshold at each grid point (DeMaria et al. 2009 method, as
in the paper). Outputs a NetCDF plus PNG maps.

Usage:
  python -m wind.wind_prob --ens data/ensemble/flossie_gefs_fhlo48 \
      --storm 2025180N13261_FLOSSIE --thresholds 34,50,64
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .init_radii import fit_r0_k, load_ibtracs_radii
from .wind_field import (K_DEFAULT, R0_DEFAULT_M, apply_shape_k,
                         get_profile, wind_uv_at_points)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KT_MS = 0.514444


def load_member_inputs(ens_dir, storm):
    """Per-member hourly inputs from the ensemble run outputs."""
    ens_dir = Path(ens_dir)
    f = xr.open_dataset(ens_dir / 'ensemble_fast.nc')
    n_members, T = f.sizes['member'], f.sizes['hour']
    times = pd.to_datetime(np.asarray(f['time'].values))
    out = {
        'v_axi_ms': np.asarray(f['fast_v_kts'].values) * KT_MS,
        'lats': np.full((n_members, T), np.nan),
        'lons': np.full((n_members, T), np.nan),
        'utran': np.full((n_members, T), np.nan),
        'vtran': np.full((n_members, T), np.nan),
        'u250': np.full((n_members, T), np.nan),
        'v250': np.full((n_members, T), np.nan),
        'u850': np.full((n_members, T), np.nan),
        'v850': np.full((n_members, T), np.nan),
        'n_members': n_members, 'T': T, 'times': times,
    }
    for mi in range(n_members):
        mdir = ens_dir / f'{storm}_M{mi:03d}'
        pkl = mdir / f'{mdir.name}_dataset.pkl'
        if not pkl.exists():
            continue
        try:
            d = pickle.load(open(pkl, 'rb'))
        except Exception:
            continue
        n = len(d.get('times', []))
        out['lats'][mi, :n] = np.asarray(d['lats'])[:n]
        out['lons'][mi, :n] = np.asarray(d['lons'])[:n]
        out['utran'][mi, :n] = np.asarray(d['utran']).reshape(-1)[:n]
        out['vtran'][mi, :n] = np.asarray(d['vtran']).reshape(-1)[:n]
        ew = np.asarray(d['env_wnds'], float)
        if ew.ndim == 3:
            ew = ew[0]
        if ew.ndim == 2 and ew.shape[0] >= n:
            out['u250'][mi, :n] = ew[:n, 0]
            out['v250'][mi, :n] = ew[:n, 1]
            out['u850'][mi, :n] = ew[:n, 2]
            out['v850'][mi, :n] = ew[:n, 3]
    return out


def analysis_grid(inputs, fc_start, window_h, ddeg=0.25, pad_deg=8.0):
    """Fixed analysis grid covering all member tracks (+pad) in window."""
    t0 = pd.Timestamp(fc_start)
    times = inputs['times']
    keep = (times >= t0) & (times < t0 + pd.Timedelta(hours=window_h))
    ki = np.where(keep)[0]
    lat_all = inputs['lats'][:, ki]
    lon_all = inputs['lons'][:, ki]
    lat_all = lat_all[np.isfinite(lat_all)]
    lon_all = lon_all[np.isfinite(lon_all)]
    lat_lo, lat_hi = np.nanpercentile(lat_all, 0.2) - pad_deg, \
        np.nanpercentile(lat_all, 99.8) + pad_deg
    lon_lo, lon_hi = np.nanpercentile(lon_all, 0.2) - pad_deg, \
        np.nanpercentile(lon_all, 99.8) + pad_deg
    glat = np.arange(np.floor(lat_lo / ddeg) * ddeg,
                     np.ceil(lat_hi / ddeg) * ddeg + ddeg / 2, ddeg)
    glon = np.arange(np.floor(lon_lo / ddeg) * ddeg,
                     np.ceil(lon_hi / ddeg) * ddeg + ddeg / 2, ddeg)
    return glat, glon, ki


def compute_probabilities(ens_dir, storm, thresholds_kt, fc_start,
                          window_h=120, ddeg=0.25, r0_m=None, k=None,
                          ibtracs_csv=None, sid=None, verbose=True):
    """Main entry: returns (prob_ds, aux dict)."""
    inputs = load_member_inputs(ens_dir, storm)
    # drive loops off actual array length, not summary n_members
    n_members = inputs['v_axi_ms'].shape[0]
    T = inputs['T']

    # ---- (r0, k) initialization from IBTrACS radii at fc_start ----
    ut0 = np.nanmean(inputs['utran'][:, 0])
    vt0 = np.nanmean(inputs['vtran'][:, 0])
    u_shr0 = np.nanmean(inputs['u250'][:, 0] - inputs['u850'][:, 0])
    v_shr0 = np.nanmean(inputs['v250'][:, 0] - inputs['v850'][:, 0])
    if r0_m is None or k is None:
        if ibtracs_csv and sid:
            radii = load_ibtracs_radii(ibtracs_csv, sid, fc_start)
            r0_fit, k_fit, _ = fit_r0_k(radii, ut0, vt0, u_shr0, v_shr0)
            r0_m = r0_fit if r0_m is None else r0_m
            k = k_fit if k is None else k
            if verbose:
                print(f'[wind] init radii @{radii["time"]}: r0='
                      f'{r0_m/1000:.0f} km k={k:.2f}')
        else:
            r0_m = R0_DEFAULT_M if r0_m is None else r0_m
            k = K_DEFAULT if k is None else k
            if verbose:
                print(f'[wind] no radii source -> r0=700 km k=1')

    glat, glon, ki = analysis_grid(inputs, fc_start, window_h, ddeg)
    if verbose:
        print(f'[wind] grid {len(glat)}x{len(glon)} @ {ddeg}deg, '
              f'{len(ki)} hourly steps in {window_h}h window')

    thr_ms = np.asarray(thresholds_kt, float) * KT_MS
    counts = np.zeros((len(thr_ms), len(glat), len(glon)), np.int32)

    # Precompute (V, lat) profile table at fixed r0 -- bilinear lookups
    # replace 0.4 s/profile CLE15 solves for the full ensemble loop.
    lat_all = inputs['lats'][:, ki]
    lat_all = lat_all[np.isfinite(lat_all)]
    v_all = inputs['v_axi_ms'][:, ki]
    v_all = v_all[np.isfinite(v_all)]
    lat_lo = float(np.clip(np.floor(lat_all.min() / 2.0) * 2.0, -60, 60))
    lat_hi = float(np.clip(np.ceil(lat_all.max() / 2.0) * 2.0 + 2.0, -60, 90))
    v_lo = max(2.0, float(np.floor(v_all.min() / 2.0) * 2.0))
    v_hi = float(np.ceil(v_all.max() / 2.0) * 2.0 + 2.0)
    from .wind_field import ProfileLookup
    lookup = ProfileLookup(r0_m,
                           lat_grid=np.arange(lat_lo, lat_hi + 1, 2.0),
                           v_grid=np.arange(v_lo, v_hi + 0.01, 2.0),
                           verbose=verbose)
    lon2d, lat2d = np.meshgrid(glon, glat)
    # a storm at (clon,clat) can only affect points within r0_m*1.15
    reach_deg_lat = (r0_m * 1.15) / 110.57e3
    reach_deg_lon = (r0_m * 1.15) / (111.32e3 * np.cos(np.radians(
        np.clip(np.nanmean(inputs['lats'][:, ki]), -60, 60))))

    for mi in range(n_members):
        # FHLO/DeMaria(2009): probability that a member EVER exceeds the
        # threshold somewhere in the window -> per-member boolean OR over
        # timesteps, then count members (not timestep-hits)
        ever = np.zeros((len(thr_ms), len(glat), len(glon)), bool)
        for ti in ki:
            v_axi = inputs['v_axi_ms'][mi, ti]
            clat = inputs['lats'][mi, ti]
            clon = inputs['lons'][mi, ti]
            if not (np.isfinite(v_axi) and np.isfinite(clat)
                    and np.isfinite(clon)) or v_axi < thr_ms.max():
                continue
            rr, vv, rmax = lookup(v_axi, clat)
            if rr is None:
                continue
            vvk = apply_shape_k(rr, vv, k, rmax)
            ut = inputs['utran'][mi, ti]
            vt = inputs['vtran'][mi, ti]
            u_shr = inputs['u250'][mi, ti] - inputs['u850'][mi, ti]
            v_shr = inputs['v250'][mi, ti] - inputs['v850'][mi, ti]
            # sub-window bounding the vortex (grid points outside r0 have
            # zero wind; skip them entirely)
            ila = np.searchsorted(glat, clat - reach_deg_lat)
            ilb = np.searchsorted(glat, clat + reach_deg_lat) + 1
            ilo = np.searchsorted(glon, clon - reach_deg_lon)
            ilb2 = np.searchsorted(glon, clon + reach_deg_lon) + 1
            ia0, ib0 = max(ila, 0), max(ilb, 0)
            io0, jo0 = max(ilo, 0), max(ilb2, 0)
            sub_lat2d = lat2d[ia0:ib0, io0:jo0]
            sub_lon2d = lon2d[ia0:ib0, io0:jo0]
            if sub_lat2d.size == 0:
                continue
            u, v = wind_uv_at_points(sub_lon2d, sub_lat2d, clon, clat, rr,
                                     vvk, ut, vt, u_shr, v_shr)
            spd = np.hypot(u, v)
            for it, t_ms in enumerate(thr_ms):
                ever[it][ia0:ib0, io0:jo0] |= (spd >= t_ms)
        counts += ever
        if verbose and (mi + 1) % max(1, n_members // 10) == 0:
            print(f'[wind] member {mi+1}/{n_members} done')

    prob = counts / float(n_members)
    t0 = pd.Timestamp(fc_start)
    ds = xr.Dataset(
        {'wind_exceedance_prob': (('threshold_kt', 'lat', 'lon'), prob),
         'member_count_exceed': (('threshold_kt', 'lat', 'lon'), counts)},
        coords={'threshold_kt': thresholds_kt, 'lat': glat, 'lon': glon},
        attrs={'storm': storm, 'fc_start': str(fc_start),
               'window_h': window_h, 'n_members': n_members,
               'r0_km': r0_m / 1000, 'shape_k': k,
               'method': 'FHLO sec 4d (DeMaria 2009), CLE15 wind field'})
    aux = {'inputs': inputs, 'ki': ki, 'times': inputs['times'],
           'fc_start': t0, 'r0_m': r0_m, 'k': k,
           'thresholds_kt': thresholds_kt}
    # IBTrACS best track over the window for plotting (black, best-track
    # convention from tracks/plot_tracks.py). Storm-dir pkl holds the full
    # replayed best track; slice to the forecast window.
    try:
        bt_dir = Path(ens_dir).parents[1] / 'ibtracs' / 'EP' / storm[0:4] / storm
        bt_pkl = bt_dir / f'{storm}_dataset.pkl'
        import pickle
        with open(bt_pkl, 'rb') as fh:
            btd = pickle.load(fh)
        btt = pd.DatetimeIndex(btd['times'])
        msk = (btt >= t0) & (btt < t0 + pd.Timedelta(hours=window_h))
        if msk.sum() >= 2:
            aux['best_track'] = (np.asarray(btd['lons'])[msk],
                                 np.asarray(btd['lats'])[msk])
    except Exception:
        pass
    return ds, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ens', required=True)
    ap.add_argument('--storm', default='2025180N13261_FLOSSIE')
    ap.add_argument('--thresholds', default='34,50,64')
    ap.add_argument('--fc-start', default='')
    ap.add_argument('--window-h', type=float, default=120)
    ap.add_argument('--grid', type=float, default=0.25)
    ap.add_argument('--ibtracs', default='')
    ap.add_argument('--sid', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    fc_start = args.fc_start
    if not fc_start:
        cfg = (Path(args.ens) / 'run_config.txt').read_text()
        for ln in cfg.splitlines():
            if ln.startswith('fc_start='):
                fc_start = ln.split('=', 1)[1].strip()
    if not fc_start:
        raise SystemExit('need --fc-start (or fc_start= in run_config.txt)')

    window_h = args.window_h
    if window_h < 0:
        # full record after fc_start
        t = pd.DatetimeIndex(xr.open_dataset(
            Path(args.ens) / 'ensemble_fast.nc')['time'].values)
        t0 = pd.Timestamp(fc_start)
        window_h = float((t[-1] - t0).total_seconds() / 3600 + 1)
        print(f'[wind] full-record window: {window_h:.0f} h '
              f'after {fc_start}')
    thresholds = [float(t) for t in args.thresholds.split(',')]

    ds, aux = compute_probabilities(
        args.ens, args.storm, thresholds, fc_start,
        window_h=window_h, ddeg=args.grid,
        ibtracs_csv=args.ibtracs or None, sid=args.sid or None)

    out = Path(args.out) if args.out else Path(args.ens) / 'wind_prob.nc'
    ds.to_netcdf(out)
    print(f'[wind] saved {out}')

    from .plot_wind_prob import plot_prob
    png = out.with_suffix('.png')
    plot_prob(ds, aux, png)
    print(f'[wind] plot {png}')


if __name__ == '__main__':
    main()
