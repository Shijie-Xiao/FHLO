#!/usr/bin/env python3
"""FHLO end-to-end pipeline: best track -> 1h interpolation -> env fields -> FAST ODE.Stages
  1. best-track   : (optional) download IBTrACS via prep/IBtracs_datasets.py
  2. interpolate  : 6h -> 1h cubic-spline track (inside prep)
                    + env-field extraction with strict vortex_lib surgery
                    -> data/ibtracs/{basin}/{year}/{STORM}/{STORM}_dataset.pkl
  3. ode          : physics/run_fast_reference.py on each pkl
                    -> fast_reference.csv / fast_reference.png per storm

Configuration: config.txt (SST source, env source, track source, parallelism).
Parallelism: storms are processed concurrently with a process pool; each storm
is sequential in time (vortex surgery caches make this the efficient order).

Usage
  python run.py                          # all storms in config (default Beryl)
  python run.py --storms 2024181N09320_BERYL
  python run.py --sst OISST              # SST source override
  python run.py --stage prep,ode         # skip ibtracs download
  python run.py --list                   # show discovered storms and exit

Ensemble mode (full 1000-member forecast):
  python run.py --ensemble \
      --synth-nc tracks/processed/beryl_2024/2024062900/synthetic_tracks_1000members.nc \
      --gefs-init '2024-06-28 12:00' --gefs-dir data/gefs_beryl \
      --members 5 --assign ecmwf
  Stages: eprep (track x member env prep via ensemble/gefs_nc_adapter.py,
  vortex surgery included) then ode (FAST ODE per member).
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'prep'))


def load_cfg(path=None):
    path = Path(path or PROJECT_ROOT / 'config.txt')
    cfg = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k, v = [x.strip() for x in s.split('=', 1)]
            cfg[k.lower()] = v
    return cfg


def discover_storms(cfg, only=None):
    """Yield (basin, year, storm_dir) for storms listed in config or --storms."""
    root = PROJECT_ROOT / cfg.get('output_dir', 'data/ibtracs')
    basins = [b.strip().upper() for b in cfg.get('basins', 'NA').split(',') if b.strip()]
    if 'ALL' in basins:
        basins = ['NA', 'EP']
    y0, y1 = int(cfg.get('year_start', 2024)), int(cfg.get('year_end', 2024))
    want = set(only) if only else None
    out = []
    for basin in basins:
        for y in range(y0, y1 + 1):
            ydir = root / basin / str(y)
            if not ydir.is_dir():
                continue
            for sd in sorted(ydir.iterdir()):
                if not sd.is_dir() or not (sd / 'track_intensity_6h.csv').exists():
                    continue
                if want is not None and sd.name not in want:
                    continue
                out.append((basin, y, sd))
    return out


# ---------- Stage runners (module level: picklable for ProcessPool) ----------

def stage_ibtracs(cfg):
    """Stage 1: refresh IBTrACS best-track CSVs (network)."""
    print('[stage 1] IBTrACS download/refresh')
    cmd = [sys.executable, str(PROJECT_ROOT / 'prep' / 'IBtracs_datasets.py')]
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return r.returncode == 0


def _prep_one(job):
    """Stage 2 worker: one storm -> dataset pkl. Returns (storm, ok, msg)."""
    import pickle
    import prepare_complete_training_data as prep
    basin, year, storm_dir, sst_source, oisst_dir, overwrite = job
    csv = storm_dir / 'track_intensity_6h.csv'
    out_pkl = storm_dir / f'{storm_dir.name}_dataset.pkl'
    try:
        if out_pkl.exists() and not overwrite:
            return storm_dir.name, True, 'skip (exists)'
        ds = prep.process_one_storm(csv, era5_root_override=None,
                                    sst_source=sst_source,
                                    oisst_dir=oisst_dir or None)
        if ds is None:
            return storm_dir.name, False, 'no valid steps'
        with open(out_pkl, 'wb') as f:
            pickle.dump(ds, f, protocol=4)
        T = ds['spatial_3d'].shape[1]
        return storm_dir.name, True, f'T={T}'
    except Exception as e:
        return storm_dir.name, False, f'{type(e).__name__}: {e}'


def _ode_one(job):
    """Stage 3 worker: one pkl -> fast_reference csv/png. Returns (storm, ok, msg)."""
    sys.path.insert(0, str(PROJECT_ROOT / 'physics'))
    import run_fast_reference as fast
    basin, year, storm_dir = job
    pkl = storm_dir / f'{storm_dir.name}_dataset.pkl'
    try:
        r = fast.process_one_pkl(pkl)
        return storm_dir.name, True, f"MAE={r['mae_kts']:.1f} kts"
    except Exception as e:
        return storm_dir.name, False, f'{type(e).__name__}: {e}'


# ---------- Ensemble mode (track x GEFS-member forecast) ----------

GEFS_ENS_MEMBERS = ['c00'] + [f'p{i:02d}' for i in range(1, 31)]  # 31 members


def resolve_member_assignment(synth_nc, n_members, assign):
    """Map synthetic track index -> GEFS member code.

    'ecmwf': NC carries parent_track (ECMWF e00-e50 parents); member index
             = parent_track[i] % 31 (51 parents -> 31 GEFS members).
    'gefs' : NC's parent_members attrs are GEFS codes (c00/p01..p30) -> use
             the parent directly (self-consistent track & environment).
    'round_robin': track i -> GEFS_ENS_MEMBERS[i % 31] (balanced fallback).
    Returns (codes[n], parents_used | None).
    """
    import numpy as np
    import xarray as xr
    ds = xr.open_dataset(synth_nc)
    n_total = ds.sizes['track']
    n_members = min(n_members, n_total)
    codes = None
    if assign in ('ecmwf', 'gefs') and 'parent_track' in ds:
        pt = ds['parent_track'].values[:n_members]
        attr = str(ds.attrs.get('parent_members', ''))
        parents = attr.split(',') if attr else []
        if assign == 'gefs' and parents and parents[0].startswith(('c', 'p')):
            codes = [parents[int(k)] if int(k) < len(parents) else 'c00'
                     for k in pt]
        else:
            codes = [GEFS_ENS_MEMBERS[int(k) % len(GEFS_ENS_MEMBERS)]
                     for k in pt]
    if codes is None:
        codes = [GEFS_ENS_MEMBERS[i % len(GEFS_ENS_MEMBERS)]
                 for i in range(n_members)]
    ds.close()
    return codes[:n_members]


def _ens_prep_one(job):
    """Ensemble prep worker: one (track, member) -> dataset pkl."""
    import pickle
    mi, track_csv, out_dir, member_code, gefs_init, gefs_dir = job
    out_dir = Path(out_dir)
    name = out_dir.name
    out_pkl = out_dir / f'{name}_dataset.pkl'
    try:
        if out_pkl.exists():
            return mi, True, 'skip (exists)'
        sys.path.insert(0, str(PROJECT_ROOT / 'ensemble'))
        sys.path.insert(0, str(PROJECT_ROOT / 'prep'))
        import gefs_nc_adapter
        gefs_nc_adapter.set_active_member(member_code, gefs_init, gefs_dir)
        gefs_nc_adapter.install()
        import prepare_complete_training_data as prep
        ds = prep.process_one_storm(track_csv, era5_root_override=None,
                                    sst_source='ERA5')
        if ds is None:
            return mi, False, 'no valid steps'
        ds['gefs_member'] = member_code
        with open(out_pkl, 'wb') as f:
            pickle.dump(ds, f, protocol=4)
        T = ds['spatial_3d'].shape[1]
        return mi, True, f'T={T} member={member_code}'
    except Exception as e:
        return mi, False, f'{type(e).__name__}: {e}'


def _ens_ode_one(job):
    """Ensemble ODE worker: one member pkl -> fast csv/png."""
    mi, out_dir = job
    sys.path.insert(0, str(PROJECT_ROOT / 'physics'))
    import run_fast_reference as fast
    out_dir = Path(out_dir)
    pkl = out_dir / f'{out_dir.name}_dataset.pkl'
    try:
        import numpy as np
        r = fast.process_one_pkl(pkl)
        return mi, True, f"peak={np.nanmax(r['v_max_kts']):.0f} kts"
    except Exception as e:
        return mi, False, f'{type(e).__name__}: {e}'


def run_ensemble(args, cfg):
    """Ensemble pipeline: synth tracks x GEFS members -> prep -> FAST ODE."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    sys.path.insert(0, str(PROJECT_ROOT / 'prep'))
    from prepare_ensemble_storm import _load_best_track, _load_synthetic, _build_member_track

    storms = discover_storms(cfg, only=[s for s in args.storms.split(',') if s.strip()] or None)
    if not storms:
        print('No storm found for ensemble (check --storms)')
        return
    basin, year, sd = storms[0]
    print(f'[ens] storm={sd.name} basin={basin} year={year}')
    bt_pkl = sd / f'{sd.name}_dataset.pkl'
    if not bt_pkl.exists():
        print(f'Missing best-track pkl {bt_pkl} (run prep stage first)')
        return

    synth_nc = Path(args.synth_nc)
    if not synth_nc.exists():
        print(f'Missing synthetic NC {synth_nc}')
        return
    n_members = args.members
    codes = resolve_member_assignment(synth_nc, n_members, args.assign)
    n_members = len(codes)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[ens] {n_members} members | assign={args.assign} | '
          f'synth={synth_nc.parent.name} | init={args.gefs_init} | '
          f'env={args.gefs_dir}')

    # ---- stage eprep: build per-member track CSV then env dataset ----
    bt = _load_best_track(bt_pkl)
    init_time, dt_hours, lons, lats, t_sec, _ = _load_synthetic(
        synth_nc, n_members, seed=0)
    import pandas as _pd
    try:
        init_delta = abs((_pd.Timestamp(str(init_time))
                          - _pd.Timestamp(args.gefs_init)).total_seconds())
    except Exception:
        init_delta = 0
    if init_delta > 3601:
        print(f'[ens] NOTE synth init {init_time} != GEFS init {args.gefs_init} '
              f'(env valid-time interpolation handles the offset)')
    jobs, member_dirs = [], []
    for mi in range(n_members):
        mdir = out_root / f'{sd.name}_M{mi:03d}'
        mdir.mkdir(parents=True, exist_ok=True)
        member_dirs.append(mdir)
        if (mdir / f'{mdir.name}_dataset.pkl').exists() and not args.overwrite:
            continue
        track = _build_member_track(bt, lons[mi], lats[mi], t_sec,
                                    init_time, init_time, args.duration_h)
        csv_path = mdir / f'{mdir.name}_track.csv'
        track.to_csv(csv_path, index=False)
        with open(mdir / 'member_assignment.txt', 'w') as f:
            f.write(f'track_idx={mi}\ngefs_member={codes[mi]}\n'
                    f'init_time={args.gefs_init}\n')
        jobs.append((mi, csv_path, mdir, codes[mi], args.gefs_init,
                     args.gefs_dir))

    stages = [s.strip().lower() for s in args.stage.split(',') if s.strip()]
    t0 = time.time()
    if 'eprep' in stages and jobs:
        print(f'[ens eprep] {len(jobs)} member preps, workers={args.workers}')
        n_ok = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_ens_prep_one, j): j[0] for j in jobs}
            done = 0
            for fut in as_completed(futs):
                mi, ok, msg = fut.result()
                n_ok += ok
                done += 1
                if not ok or done % 10 == 0 or done == len(jobs):
                    print(f'  [{"OK" if ok else "FAIL"}] M{mi:03d}: {msg}')
        print(f'[ens eprep] done: {n_ok}/{len(jobs)} '
              f'({time.time() - t0:.0f}s)')

    if 'ode' in stages:
        ode_jobs = [(i, d) for i, d in enumerate(member_dirs)
                    if (d / f'{d.name}_dataset.pkl').exists()]
        print(f'[ens ode] {len(ode_jobs)} FAST ODE runs, workers={args.workers}')
        n_ok = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_ens_ode_one, j): j[0] for j in ode_jobs}
            done = 0
            for fut in as_completed(futs):
                mi, ok, msg = fut.result()
                n_ok += ok
                done += 1
                if not ok or done % 10 == 0 or done == len(ode_jobs):
                    print(f'  [{"OK" if ok else "FAIL"}] M{mi:03d}: {msg}')
        print(f'[ens ode] done: {n_ok}/{len(ode_jobs)} '
              f'({time.time() - t0:.0f}s)')
        _summarize_ensemble(out_root, sd.name)


def _summarize_ensemble(out_root, storm_name):
    """Print ensemble peak-intensity stats across members."""
    import glob
    import numpy as np
    import pandas as pd
    peaks, pkl_n = [], 0
    for csvf in sorted(Path(out_root).glob('*/fast_reference.csv')):
        try:
            df = pd.read_csv(csvf)
            peaks.append(df['v_max_kts'].max())
            pkl_n += 1
        except Exception:
            pass
    if peaks:
        p = np.asarray(peaks)
        print(f'[ens summary] {len(p)} members | peak kts: '
              f'mean={p.mean():.0f} median={np.median(p):.0f} '
              f'min={p.min():.0f} max={p.max():.0f} sd={p.std():.1f}')


# ---------- Pipeline ----------

def main():
    ap = argparse.ArgumentParser(description='FHLO pipeline')
    ap.add_argument('--config', default=str(PROJECT_ROOT / 'config.txt'))
    ap.add_argument('--storms', default='', help='comma-separated storm dir names')
    ap.add_argument('--sst', default='', help='SST source override: ERA5 | OISST')
    ap.add_argument('--stage', default='prep,ode',
                    help='stages to run: ibtracs,prep,ode (default prep,ode); '
                         'ensemble mode: eprep,ode')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--ensemble', action='store_true',
                    help='full ensemble forecast mode (track x GEFS member)')
    ap.add_argument('--synth-nc', default='',
                    help='synthetic_tracks_*.nc (ensemble mode)')
    ap.add_argument('--gefs-init', default='2024-06-28 12:00',
                    help='GEFS forecast init time (ensemble mode)')
    ap.add_argument('--gefs-dir', default='data/gefs_beryl',
                    help='local GEFS nc dir (ensemble mode)')
    ap.add_argument('--members', type=int, default=1000,
                    help='number of ensemble members to run')
    ap.add_argument('--assign', default='ecmwf',
                    choices=['ecmwf', 'gefs', 'round_robin'],
                    help='env-member assignment mode (ensemble mode)')
    ap.add_argument('--out-root', default='',
                    help='ensemble output root (default data/ensemble/{storm})')
    ap.add_argument('--duration-h', type=float, default=240.0,
                    help='forecast duration in hours per member')
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    cfg_storms = [s.strip() for s in cfg.get('storms', '').split(',') if s.strip() and s.strip().upper() != 'ALL']
    if not args.storms:
        args.storms = ','.join(cfg_storms)
    # ensemble-mode defaults from config.txt
    if not args.gefs_init or args.gefs_init == '2024-06-28 12:00':
        args.gefs_init = cfg.get('gefs_init', '2024-06-28 12:00')
    if not args.gefs_dir or args.gefs_dir == 'data/gefs_beryl':
        args.gefs_dir = cfg.get('gefs_beryl_dir', 'data/gefs_beryl')

    if args.ensemble:
        if not args.synth_nc:
            print('--ensemble requires --synth-nc')
            return
        if not args.out_root:
            args.out_root = str(PROJECT_ROOT / 'data' / 'ensemble'
                                / 'beryl_gefs_1000')
        if not args.workers:
            args.workers = int(cfg.get('n_workers', 4))
        if args.stage in ('prep,ode', ''):
            args.stage = 'eprep,ode'
        run_ensemble(args, cfg)
        return

    cfg_storms = [s.strip() for s in cfg.get('storms', '').split(',') if s.strip() and s.strip().upper() != 'ALL']
    only = [s.strip() for s in args.storms.split(',') if s.strip()] or cfg_storms or None
    sst = args.sst or cfg.get('sst_source', 'ERA5')
    workers = args.workers or int(cfg.get('n_workers', 4))
    stages = [s.strip().lower() for s in args.stage.split(',') if s.strip()]

    storms = discover_storms(cfg, only=only)
    if args.list:
        for b, y, sd in storms:
            print(f'{b} {y} {sd.name}')
        print(f'total: {len(storms)}')
        return
    if not storms:
        print('No storms found (check config storms=/basins=/year range)')
        return
    print(f'Storms ({len(storms)}): {[sd.name for _, _, sd in storms]}')
    print(f'Config: sst={sst} workers={workers} stages={stages}')

    t0 = time.time()

    if 'ibtracs' in stages:
        stage_ibtracs(cfg)
        storms = discover_storms(cfg, only=only)

    if 'prep' in stages:
        print(f'[stage 2] prep (vortex surgery, sst={sst})')
        jobs = [(b, y, sd, sst, cfg.get('oisst_dir', ''), args.overwrite)
                for b, y, sd in storms]
        n_ok = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_prep_one, j) for j in jobs]):
                name, ok, msg = fut.result()
                n_ok += ok
                print(f'  [{"OK" if ok else "FAIL"}] {name}: {msg}')
        print(f'[stage 2] done: {n_ok}/{len(jobs)}')

    if 'ode' in stages:
        print('[stage 3] FAST ODE')
        jobs = [(b, y, sd) for b, y, sd in storms
                if (sd / f'{sd.name}_dataset.pkl').exists()]
        n_ok = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_ode_one, j) for j in jobs]):
                name, ok, msg = fut.result()
                n_ok += ok
                print(f'  [{"OK" if ok else "FAIL"}] {name}: {msg}')
        print(f'[stage 3] done: {n_ok}/{len(jobs)}')

    print(f'Pipeline finished in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
