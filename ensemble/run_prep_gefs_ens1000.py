#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 1000 ensemble datasets for Erin 2025 using GEFS environment fields.

Strategy:
  - 1000 synthetic Markov-sampled tracks come from
    /pscratch/sd/s/sixao74/Deepmind/PINN/ensemble_tracks/erin/synthetic_tracks_1000members.nc
  - 31 GEFS ensemble members (c00, gep01..gep30) provide the environment fields.
    We assign each of the 1000 tracks to a member by Monte Carlo sampling:
    tracks 0..29 -> members c00..p29; tracks 30..59 -> c00..p29; ... round-robin
    with optional random shuffle so member usage is balanced.
  - Each (track, member) pair is processed by prepare_complete_training_data
    with the gefs_ens_adapter active for that member's GRIB files.

Output:
  {OUT_ROOT}/{YEAR}/{HID}_M{NNN}/{HID}_M{NNN}_dataset.pkl
"""
import os, sys, csv, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))  # PINN/ root

from prepare_ensemble_storm import _load_best_track, _load_synthetic, _build_member_track

# --- Config (env-overridable) ---
CASE_DIR    = Path(os.environ.get('GEFS_CASE_DIR',
    '/global/cfs/cdirs/m5011/Jay/ERA5/GFS/2025_ERIN_NA'))
INIT_TIME   = os.environ.get('GEFS_INIT_TIME', '2025-08-11 12:00')  # match ECMWF TIGGE
SYNTH_NC    = os.environ.get('PREP_SYNTH_NC',
    '/pscratch/sd/s/sixao74/Deepmind/PINN/ensemble_tracks/erin/synthetic_tracks_1000members.nc')
BEST_TRACK  = os.environ.get('PREP_BEST_TRACK',
    '/pscratch/sd/s/sixao74/Deepmind/PINN/data/NA/2025/2025223N17337_ERIN/2025223N17337_ERIN_dataset.pkl')
HID         = os.environ.get('PREP_HID', '2025223N17337_ERIN')
OUT_ROOT    = Path(os.environ.get('PREP_OUT_ROOT',
    '/pscratch/sd/s/sixao74/Deepmind/PINN/ensemble_data/erin_gefs_div1000'))
YEAR        = int(os.environ.get('PREP_YEAR', '2025'))
REF_TIME    = pd.Timestamp(os.environ.get('PREP_REF_TIME', INIT_TIME))  # fully divergent from h=0
DURATION_H  = int(os.environ.get('PREP_DURATION_H', '264'))  # 11 days
N_MEMBERS   = int(os.environ.get('PREP_N_MEMBERS', '1000'))
SEED        = int(os.environ.get('PREP_SEED', '42'))
VORTEX      = os.environ.get('PREP_VORTEX', 'annulus')

GEFS_ENS_MEMBERS = ['c00'] + [f'p{int(i):02d}' for i in range(1, 31)]  # 31 members

N_WORKERS = int(os.environ.get('PREP_WORKERS', '4'))  # lower default: each is heavy
START     = int(os.environ.get('PREP_START', '0'))
END       = int(os.environ.get('PREP_END', str(N_MEMBERS)))
ASSIGN_MODE = os.environ.get('PREP_ASSIGN_MODE', 'parent_paired')  # 'parent_paired' | 'round_robin' | 'random'

# ---- shared (per-process) inputs, lazily loaded in each worker ----
_G = {}


def _ensure_inputs():
    if _G:
        return _G
    bt_path = Path(BEST_TRACK)
    if not bt_path.exists():
        # try alternative paths
        alt = Path('/pscratch/sd/s/sixao74/Deepmind/PINN/training_data') / str(YEAR)
        candidates = list(alt.glob(f'*ERIN*/*ERIN*_dataset.pkl'))
        if candidates:
            bt_path = candidates[0]
            print(f'[prep] using best-track: {bt_path}', flush=True)
    _G['bt'] = _load_best_track(bt_path)
    init_time, dt_hours, lons, lats, t_sec, picked = _load_synthetic(
        Path(SYNTH_NC), N_MEMBERS, SEED)
    _G['init_time'] = init_time
    _G['lons'] = lons; _G['lats'] = lats; _G['t_sec'] = t_sec; _G['picked'] = picked
    _G['sfc_cache'] = {}; _G['pl_cache'] = {}
    # member assignment
    if ASSIGN_MODE == 'parent_paired':
        # FHLO-style: synthetic track i was seeded FROM parent ensemble member
        # parent_track[i] (see sample_tracks member-paired bootstrap). Assign
        # the environment of that SAME member -> track & env self-consistent.
        ds = xr.open_dataset(SYNTH_NC)
        if 'parent_track' not in ds:
            ds.close()
            raise RuntimeError(
                'PREP_ASSIGN_MODE=parent_paired but synthetic NC lacks '
                'parent_track (re-sample with updated sample_tracks.py)')
        pt = ds['parent_track'].values[:N_MEMBERS]
        ds.close()
        _G['track_member'] = np.asarray(pt, int)
        uniq = set(_G['track_member'].tolist())
        print(f'[prep] parent_paired: {len(uniq)} distinct env members used', flush=True)
    elif ASSIGN_MODE == 'random':
        # Random sample of members (with replacement) for each track
        rng = np.random.default_rng(SEED)
        _G['track_member'] = rng.choice(len(GEFS_ENS_MEMBERS), size=N_MEMBERS)
    else:
        # round-robin: track i -> GEFS_ENS_MEMBERS[i % 31]
        _G['track_member'] = np.array([i % len(GEFS_ENS_MEMBERS) for i in range(N_MEMBERS)])
    return _G


def _prep_member(mi):
    """Process one synthetic track using its assigned GEFS ensemble member env."""
    # Install the GEFS adapter fresh in this process
    import gefs_ens_adapter
    gefs_ens_adapter.install()

    g = _ensure_inputs()
    name = f'{HID}_M{mi:03d}'
    sdir = OUT_ROOT / str(YEAR) / name
    sdir.mkdir(parents=True, exist_ok=True)
    ds_pkl = sdir / f'{name}_dataset.pkl'
    if ds_pkl.exists():
        return (mi, 'skip', '')

    try:
        # Configure adapter for this member
        member_idx = int(g['track_member'][mi])
        member_code = GEFS_ENS_MEMBERS[member_idx]
        gefs_ens_adapter.set_active_member(
            case_dir=CASE_DIR,
            member_code=member_code,
            init_time=INIT_TIME,
        )

        # Build track. When REF_TIME is later than the synth init, start the
        # member track AT ref_time (skip the pre-ref synthetic/bt segment) so
        # the forecast window opens at the chosen time (e.g. just before RI).
        eff_ref = max(g['init_time'], REF_TIME)
        track = _build_member_track(
            g['bt'], g['lons'][mi], g['lats'][mi], g['t_sec'], g['init_time'],
            eff_ref, DURATION_H)
        if eff_ref > g['init_time']:
            track = track[track['time'] >= eff_ref].reset_index(drop=True)
        csv_path = sdir / f'{name}_track.csv'
        track.to_csv(csv_path, index=False)

        # write assignment record
        with open(sdir / 'member_assignment.txt', 'w') as f:
            f.write(f'track_idx={mi}\nensemble_member={member_code}\n'
                    f'init_time={INIT_TIME}\ncase_dir={CASE_DIR}\n')

        # Now call process_one_storm (uses monkey-patched adapter)
        from prepare_complete_training_data import process_one_storm
        _era5 = os.environ.get('PREP_ERA5_ROOT') or None
        result = process_one_storm(str(csv_path), era5_root_override=_era5)
        if result is None:
            return (mi, 'fail', 'process_one_storm None')
        result['hurricane'] = name
        result['gefs_member'] = member_code
        with open(ds_pkl, 'wb') as f:
            pickle.dump(result, f, protocol=4)
        T = result['spatial_3d'].shape[1]
        return (mi, 'ok', f'T={T} member={member_code}')
    except Exception as e:
        return (mi, 'fail', repr(e)[:200])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / str(YEAR)).mkdir(parents=True, exist_ok=True)
    g = _ensure_inputs()
    idx_csv = OUT_ROOT / str(YEAR) / f'{HID}_ensemble_index.csv'
    with open(idx_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['member', 'storm_dir', 'synthetic_track_id', 'gefs_ens_member'])
        for mi in range(min(END, N_MEMBERS)):
            tid = int(g['picked'][mi])
            mem_idx = int(g['track_member'][mi])
            mem_code = GEFS_ENS_MEMBERS[mem_idx]
            w.writerow([mi, f'{HID}_M{mi:03d}', tid, mem_code])
    members = list(range(START, min(END, N_MEMBERS)))
    print(f'[prep] {len(members)} members [{START},{END}) workers={N_WORKERS} '
          f'synth={Path(SYNTH_NC).name} init={g["init_time"]} '
          f'assign={ASSIGN_MODE} case={CASE_DIR.name}', flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ok = fail = skip = 0
    # max_tasks_per_child: recycle workers periodically. Long-lived workers
    # accumulate heap fragmentation + xarray caches (beryl OOM'd at ~360 pkls
    # after 5.5h, 58.8GB); a fresh worker starts at ~7GB.
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             max_tasks_per_child=40) as ex:
        futs = {ex.submit(_prep_member, mi): mi for mi in members}
        done = 0
        for fut in as_completed(futs):
            mi, status, msg = fut.result()
            done += 1
            if status == 'ok': ok += 1
            elif status == 'skip': skip += 1
            else:
                fail += 1
                print(f'  [M{mi:03d}] FAIL {msg}', flush=True)
            if done % 5 == 0 or done == len(members):
                print(f'  progress {done}/{len(members)}  ok={ok} skip={skip} fail={fail}', flush=True)
    print(f'[prep] done ok={ok} skip={skip} fail={fail} out={OUT_ROOT/str(YEAR)}', flush=True)


if __name__ == '__main__':
    main()
