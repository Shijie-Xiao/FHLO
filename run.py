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
    """Yield (basin, year, storm_dir) for storms listed in config or --storms.

    When explicit storm names are given (only), every basin directory is
    scanned so cross-basin storms (e.g. EP Flossie with basins=NA) resolve.
    """
    root = PROJECT_ROOT / cfg.get('output_dir', 'data/ibtracs')
    basins = [b.strip().upper() for b in cfg.get('basins', 'NA').split(',') if b.strip()]
    explicit = only is not None
    if 'ALL' in basins:
        basins = ['NA', 'EP']
    if explicit:
        basins = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith('_')) or basins
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
ECMWF_ENS_MEMBERS = [f'e{i:02d}' for i in range(0, 51)]           # 51 members


def resolve_member_assignment(synth_nc, n_members, assign, env):
    """Map synthetic-track index -> environment-member code.

    The mapping depends on BOTH the parents carried by the synth NC
    (parent_track + parent_members attr) and the env field source (--env):

    assign='auto' (default, resolved by env before the call):
      env='era5'  -> 'ecmwf'   (member code is bookkeeping only)
      env='gefs'  -> 'gefs'

    'gefs'       : NC's parent_members attr holds GEFS codes (c00/p01..p30)
                   -> use the parent directly (self-consistent track & env:
                   the perturbation that generated the track also drives it).
    'ecmwf'      : NC's parents are ECMWF codes (e00..e50) but the env source
                   is GEFS/ERA5 -> bijective one-to-one hash
                   member = GEFS_ENS_MEMBERS[(parent * 7) % 31]. The old
                   parent % 31 was NON-uniform (51 parents: e00-e19 hit twice,
                   e20-e50 once); the multiplicative hash with gcd(7,31)=1
                   gives every GEFS member exactly ceil/floor(51/31) parents
                   and every parent exactly one member (injective per parent).
    'ecmwf_field': future --env ecmwf with ECMWF forecast fields; reads the
                   ECMWF parent code directly (e00..e50, one-to-one with the
                   driving perturbation).
    'round_robin': track i -> GEFS_ENS_MEMBERS[i % 31] (balanced fallback
                   when the NC carries no parent info).
    Returns codes[:n_members].
    """
    import numpy as np
    import xarray as xr
    ds = xr.open_dataset(synth_nc)
    n_total = ds.sizes['track']
    n_members = min(n_members, n_total)
    codes = None
    if assign in ('ecmwf', 'gefs', 'ecmwf_field') and 'parent_track' in ds:
        pt = ds['parent_track'].values[:n_members]
        attr = str(ds.attrs.get('parent_members', ''))
        parents = [p for p in attr.split(',') if p]
        if assign == 'gefs' and parents and parents[0].startswith(('c', 'p')):
            codes = [parents[int(k)] if int(k) < len(parents) else 'c00'
                     for k in pt]
        elif assign == 'ecmwf_field' and parents and parents[0].startswith('e'):
            codes = [parents[int(k)] if int(k) < len(parents) else parents[0]
                     for k in pt]
        elif assign == 'ecmwf':
            # uniform bijective hash 51 ECMWF parents -> 31 GEFS members
            codes = [GEFS_ENS_MEMBERS[(int(k) * 7) % len(GEFS_ENS_MEMBERS)]
                     for k in pt]
    if codes is None:
        codes = [GEFS_ENS_MEMBERS[i % len(GEFS_ENS_MEMBERS)]
                 for i in range(n_members)]
    ds.close()
    return codes[:n_members]


def _worker_init():
    """Per-worker: pin BLAS threads to 1 (64 procs x N threads = oversubscribe)."""
    import os
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _ens_prep_batch(batch):
    """Ensemble prep worker: a batch of same-member (track, member) jobs.

    Batching keeps each worker on ONE GEFS member for the whole batch, so the
    per-worker fhour cache stays hot (3-hourly slabs reused across tracks).
    env=era5 skips the adapter entirely: environment fields come from the
    local ERA5 analysis (data/era5), member_code is record-keeping only.
    Returns [(mi, ok, msg), ...].
    """
    import pickle
    out = []
    if not batch:
        return out
    env = batch[0][6] if len(batch[0]) > 6 else 'gefs'
    vortex_mode = batch[0][7] if len(batch[0]) > 7 else 'annulus'
    _, track_csv, out_dir, member_code, gefs_init, gefs_dir = batch[0][:6]
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'prep'))
        if env == 'gefs':
            sys.path.insert(0, str(PROJECT_ROOT / 'ensemble'))
            import gefs_nc_adapter
            gefs_nc_adapter.set_active_member(member_code, gefs_init, gefs_dir)
            gefs_nc_adapter.install()
        import prepare_complete_training_data as prep
        prep.set_vortex_mode(vortex_mode)
        import gc
        for job in batch:
            mi, track_csv, out_dir, member_code, gefs_init, gefs_dir = job[:6]
            out_dir = Path(out_dir)
            out_pkl = out_dir / f'{out_dir.name}_dataset.pkl'
            if out_pkl.exists():
                out.append((mi, True, 'skip (exists)'))
                continue
            try:
                ds = prep.process_one_storm(track_csv, era5_root_override=None,
                                            sst_source='ERA5')
                if ds is None:
                    out.append((mi, False, 'no valid steps'))
                    continue
                # Slim for storage: drop the bulky spatial env grids
                # (spatial_3d/spatial_2d); keep the ODE coefficients
                # (scalars=alpha/beta/gamma/vp, chi/s/xs refs, env winds,
                # translation, Cd/BLH, v_gt, times/track).
                for k in ('spatial_3d', 'spatial_2d'):
                    ds.pop(k, None)
                ds['gefs_member'] = member_code
                with open(out_pkl, 'wb') as f:
                    pickle.dump(ds, f, protocol=4)
                out.append((mi, True, f'T={len(ds["times"])}'))
            except Exception as e:
                out.append((mi, False, f'{type(e).__name__}: {e}'))
            finally:
                # ERA5 mode: caches are per-storm dicts, but xarray chunk
                # buffers linger; release between members to cap peak RSS
                gc.collect()
    except Exception as e:
        out.extend((j[0], False, f'init {type(e).__name__}: {e}') for j in batch)
    return out


def _ens_ode_one(job):
    """Ensemble ODE worker: one member pkl -> per-member fast csv + arrays.

    Returns (mi, ok, msg, result) where result carries the member's V(t)
    series (v_fast/v_max/vp/v_obz/m, kts) for the ensemble NC.
    """
    mi, out_dir = job
    sys.path.insert(0, str(PROJECT_ROOT / 'physics'))
    import run_fast_reference as fast
    out_dir = Path(out_dir)
    pkl = out_dir / f'{out_dir.name}_dataset.pkl'
    try:
        import numpy as np
        import pandas as pd
        r = fast.process_one_pkl(pkl, save_csv=True, save_plot=False)
        csv = out_dir / 'fast_reference.csv'
        if csv.exists():
            df = pd.read_csv(csv)
            res = {k: df[k].to_numpy(dtype=float)
                   for k in ('v_fast_kts', 'v_max_kts', 'vp_kts',
                             'v_obz_kts', 'm')}
            res['time'] = pd.to_datetime(df['time']).to_numpy()
        else:
            res = None
        return mi, True, f"peak={np.nanmax(r['v_max_kts']):.0f} kts", res
    except Exception as e:
        return mi, False, f'{type(e).__name__}: {e}', None


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
    env = getattr(args, 'env', 'gefs')
    vortex_mode = getattr(args, 'vortex_mode', 'annulus')
    codes = resolve_member_assignment(synth_nc, n_members, args.assign, env=env)
    n_members = len(codes)
    if vortex_mode == 'surgery' and env == 'gefs':
        print('[ens] ERROR: vortex_mode=surgery requires full-global env '
              'fields; GEFS crops are regional (45x70 deg). Use --env era5 '
              'for strict surgery.')
        return

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[ens] {n_members} members | env={env} | vortex={vortex_mode} | '
          f'assign={args.assign} | '
          f'synth={synth_nc.parent.name} | init={args.gefs_init}'
          + (f' | gefs_dir={args.gefs_dir}' if env == 'gefs' else ''))
    # record the experiment config next to the outputs for reproducibility
    with open(out_root / 'run_config.txt', 'w') as f:
        f.write(f'storm={sd.name}\nenv={env}\nvortex_mode={vortex_mode}\n'
                f'members={n_members}\nassign={args.assign}\n'
                f'synth_nc={synth_nc}\ngefs_init={args.gefs_init}\n'
                f'duration_h={args.duration_h}\n'
                f'gefs_dir={args.gefs_dir if env == "gefs" else ""}\n')

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
    if init_delta > 3601 and env == 'gefs':
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
                    f'env={env}\ninit_time={args.gefs_init}\n')
        jobs.append((mi, csv_path, mdir, codes[mi], args.gefs_init,
                     args.gefs_dir, env, vortex_mode))

    stages = [s.strip().lower() for s in args.stage.split(',') if s.strip()]
    t0 = time.time()
    if 'eprep' in stages and jobs:
        # group jobs by GEFS member so each worker batch keeps its fhour
        # cache hot on one member's files (era5 env: all jobs share one
        # group; batches sized to keep all workers busy)
        from collections import defaultdict
        by_member = defaultdict(list)
        for j in jobs:
            by_member[j[3]].append(j)
        target_batches = max(args.workers, 1)
        batches = []
        for code in sorted(by_member):
            mj = by_member[code]
            n_split = max(1, round(len(mj) / max(1, len(jobs)) * target_batches))
            size = -(-len(mj) // n_split)
            for k in range(0, len(mj), size):
                batches.append(mj[k:k + size])
        print(f'[ens eprep] {len(jobs)} member preps in {len(batches)} batches '
              f'(env={env}, {len(by_member)} member groups, '
              f'vortex={vortex_mode}), workers={args.workers}')
        n_ok = 0
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as ex:
            futs = {ex.submit(_ens_prep_batch, b): b[0][0] for b in batches}
            done_batches = 0
            for fut in as_completed(futs):
                results = fut.result()
                for mi, ok, msg in results:
                    n_ok += ok
                    if not ok:
                        print(f'  [FAIL] M{mi:03d}: {msg}')
                done_batches += 1
                n_done = done_batches
                if done_batches % 5 == 0 or done_batches == len(batches):
                    print(f'  [eprep] {done_batches}/{len(batches)} batches, '
                          f'{n_ok} ok ({time.time() - t0:.0f}s)')
        print(f'[ens eprep] done: {n_ok}/{len(jobs)} '
              f'({time.time() - t0:.0f}s)')

    if 'ode' in stages:
        ode_jobs = [(i, d) for i, d in enumerate(member_dirs)
                    if (d / f'{d.name}_dataset.pkl').exists()]
        print(f'[ens ode] {len(ode_jobs)} FAST ODE runs, workers={args.workers}')
        n_ok = 0
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as ex:
            futs = {ex.submit(_ens_ode_one, j): j[0] for j in ode_jobs}
            done = 0
            for fut in as_completed(futs):
                mi, ok, msg, res = fut.result()
                n_ok += ok
                done += 1
                if res is not None:
                    results[mi] = res
                if not ok or done % 10 == 0 or done == len(ode_jobs):
                    print(f'  [{"OK" if ok else "FAIL"}] M{mi:03d}: {msg}')
        print(f'[ens ode] done: {n_ok}/{len(ode_jobs)} '
              f'({time.time() - t0:.0f}s)')
        _save_ensemble_nc(out_root, sd.name, results, env, args.assign,
                          args.gefs_init)
        _summarize_ensemble(out_root, sd.name)
        _plot_ensemble(out_root, sd.name)

    if 'plot' in stages and 'ode' not in stages:
        # plot-only invocation (e.g. --stage plot) reads the saved NC
        _plot_ensemble(out_root, sd.name)


def _plot_ensemble(out_root, storm_name):
    """Render ensemble_fast.png/svg from ensemble_fast.nc (FAST-only)."""
    out_root = Path(out_root)
    nc = out_root / 'ensemble_fast.nc'
    if not nc.exists():
        print(f'[ens plot] missing {nc}, skip')
        return
    try:
        import subprocess
        cmd = [sys.executable,
               str(PROJECT_ROOT / 'ensemble' / 'plot_ensemble_fast.py'),
               '--ens-nc', str(nc),
               '--out_png', str(out_root / 'ensemble_fast.png'),
               '--storm', storm_name.split('_')[-1] if '_' in storm_name
               else storm_name]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for ln in (r.stdout or '').strip().splitlines():
            print(f'  {ln}')
        if r.returncode != 0:
            print(f'[ens plot] FAILED: {(r.stderr or "").strip()[-300:]}')
        else:
            print(f'[ens plot] saved {out_root / "ensemble_fast.png"} (+svg)')
    except Exception as e:
        print(f'[ens plot] {type(e).__name__}: {e}')


def _save_ensemble_nc(out_root, storm_name, results, env, assign, init_time):
    """Assemble per-member FAST series into ONE ensemble NetCDF.

    Layout (plot_vs_google_vmax.py ready):
      fast_vmax_kts (member, hour)  -- ODE Vmax
      fast_v_kts / vp_kts / v_obz_kts / m likewise; hour = hours since init.
    Also fast_chi/fast_s (from prep pkls) for the vent panel, seq_len marks
    each member's valid length (ragged tracks are NaN-padded).
    """
    import pickle
    import numpy as np
    import xarray as xr

    out_root = Path(out_root)
    if not results:
        print('[ens nc] no ODE results, skip')
        return
    mis = sorted(results)
    T = max(len(results[mi]['v_max_kts']) for mi in mis)
    n = len(mis)
    times0 = results[mis[0]].get('time')
    hours = np.arange(T, dtype=float)
    base = np.datetime64('2000-01-01') if times0 is None else times0[0]

    def pad(key):
        arr = np.full((n, T), np.nan)
        for i, mi in enumerate(mis):
            v = np.asarray(results[mi].get(key), float)
            arr[i, :len(v)] = v
        return arr

    fields = {
        'fast_vmax_kts': (('member', 'hour'), pad('v_max_kts')),
        'fast_v_kts': (('member', 'hour'), pad('v_fast_kts')),
        'vp_kts': (('member', 'hour'), pad('vp_kts')),
        'v_obz_kts': (('member', 'hour'), pad('v_obz_kts')),
        'm': (('member', 'hour'), pad('m')),
        'seq_len': (('member',), np.array([len(results[mi]['v_max_kts'])
                                           for mi in mis], int)),
    }
    # chi/s per member from the slim pkls (vent panel input)
    chi_arr = np.full((n, T), np.nan)
    s_arr = np.full((n, T), np.nan)
    for i, mi in enumerate(mis):
        pkl = out_root / f'{storm_name}_M{mi:03d}' / \
            f'{storm_name}_M{mi:03d}_dataset.pkl'
        if not pkl.exists():
            continue
        try:
            ds = pickle.load(open(pkl, 'rb'))
            c = np.asarray(ds.get('chi_ref', []), float).ravel()
            s = np.asarray(ds.get('s_ref', []), float).ravel()
            chi_arr[i, :len(c)] = c
            s_arr[i, :len(s)] = s
        except Exception:
            pass
    fields['fast_chi'] = (('member', 'hour'), chi_arr)
    fields['fast_s'] = (('member', 'hour'), s_arr)

    # member codes from assignment files
    codes = []
    for mi in mis:
        asg = out_root / f'{storm_name}_M{mi:03d}' / 'member_assignment.txt'
        c = ''
        if asg.exists():
            for ln in asg.read_text().splitlines():
                if ln.startswith('gefs_member='):
                    c = ln.split('=', 1)[1].strip()
        codes.append(c)
    try:
        time_coord = np.array([base + np.timedelta64(int(h), 'h')
                               for h in hours])
    except Exception:
        time_coord = hours.astype('datetime64[h]')

    ds_out = xr.Dataset(
        fields,
        coords={'member': np.array(mis), 'hour': hours, 'time': time_coord},
        attrs={'storm': storm_name, 'env': env, 'assign': assign,
               'init_time': str(init_time),
               'gefs_members': ','.join(codes)})
    out = out_root / 'ensemble_fast.nc'
    ds_out.to_netcdf(out)
    peaks = np.nanmax(fields['fast_vmax_kts'][1], axis=1)
    print(f'[ens nc] saved {out} ({n} members x {T} h, '
          f'peak mean={np.nanmean(peaks):.0f} max={np.nanmax(peaks):.0f} kts)')


def _summarize_ensemble(out_root, storm_name, save=True):
    """Collect per-member coefficients + winds into ensemble-level files.

    Outputs (in out_root):
      ensemble_winds.csv   long-format V(t): member, gefs_member, hour,
                           v_max_kts, v_obz_kts (per ODE step)
      ensemble_summary.nc  (member, hour): chi/u250/v250/u850/v850/shear from
                           the prep pkls + per-member peak_kts
    Prints the peak-intensity summary line.
    """
    import pickle
    import numpy as np
    import pandas as pd

    out_root = Path(out_root)
    winds_rows, coef = [], []
    peaks = []
    for mdir in sorted(out_root.glob('*/')):
        csvf = mdir / 'fast_reference.csv'
        pklf = mdir / f'{mdir.name}_dataset.pkl'
        if not csvf.exists():
            continue
        mi = int(mdir.name.rsplit('_M', 1)[-1])
        mem_code = None
        asg = mdir / 'member_assignment.txt'
        if asg.exists():
            for ln in asg.read_text().splitlines():
                if ln.startswith('gefs_member='):
                    mem_code = ln.split('=', 1)[1].strip()
        try:
            df = pd.read_csv(csvf)
        except Exception:
            continue
        hours = pd.to_datetime(df['time'])
        hr = (hours - hours.iloc[0]).dt.total_seconds() / 3600.0
        obz = df.get('v_obz_kts')
        for t, v, vo in zip(hr, df['v_max_kts'],
                            obz if obz is not None else [np.nan] * len(df)):
            winds_rows.append((mi, mem_code, float(t), float(v),
                               float(vo) if vo == vo else np.nan))
        if len(df):
            peaks.append(float(df['v_max_kts'].max()))
        # coefficients from the prep pkl (chi + env winds per hour)
        if pklf.exists():
            try:
                ds = pickle.load(open(pklf, 'rb'))
                n = len(ds.get('times', []))
                chi = np.asarray(ds.get('chi_ref', []), float).ravel()
                chi = chi[:n] if chi.size >= n else np.full(n, np.nan)
                env = ds.get('env_wnds')
                env = env if env is not None else [None] * n
                cols = np.full((n, 4), np.nan)
                for k, e in enumerate(env[:n]):
                    ev = np.asarray(e, float).ravel()
                    if ev.size >= 4:
                        cols[k] = ev[:4]
                u250, v250, u850, v850 = cols.T
                shear = np.hypot(u250 - u850, v250 - v850)
                coef.append((mi, mem_code, n, chi, u250, v250, u850, v850, shear))
            except Exception:
                pass

    if not peaks:
        print('[ens summary] no completed members found')
        return
    p = np.asarray(peaks)
    print(f'[ens summary] {len(p)} members | peak kts: '
          f'mean={p.mean():.0f} median={np.median(p):.0f} '
          f'min={p.min():.0f} max={p.max():.0f} sd={p.std():.1f}')
    if not save:
        return

    wdf = pd.DataFrame(winds_rows,
                       columns=['member', 'gefs_member', 'hour',
                                'v_max_kts', 'v_obz_kts'])
    wdf.to_csv(out_root / 'ensemble_winds.csv', index=False)

    try:
        import xarray as xr
        T = max(c[2] for c in coef) if coef else 0
        M = len(coef)
        fields = {}
        for fi, fname in enumerate(['chi', 'u250', 'v250', 'u850', 'v850',
                                    'shear']):
            arr = np.full((M, T), np.nan)
            for i, c in enumerate(coef):
                v = c[3 + fi]
                arr[i, :len(v)] = v
            fields[fname] = (('member', 'hour'), arr)
        if M:
            fields['peak_kts'] = (('member',), [c[0] for c in coef])
        ds_out = xr.Dataset(
            fields,
            coords={'member': np.array([c[0] for c in coef]),
                    'hour': np.arange(T, dtype=float)},
            attrs={'storm': storm_name,
                   'gefs_members': ','.join(str(c[1]) for c in coef if c[1])})
        ds_out.to_netcdf(out_root / 'ensemble_summary.nc')
        print(f'[ens summary] saved ensemble_winds.csv ({len(wdf)} rows) '
              f'+ ensemble_summary.nc ({M} members x {T} h)')
    except ImportError:
        pass


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
                    help='full ensemble forecast mode (track x env member)')
    ap.add_argument('--env', default='gefs', choices=['gefs', 'era5'],
                    help='ensemble environment source: gefs = GEFS forecast '
                         'fields (member-paired); era5 = ERA5 analysis fields '
                         '(pairs with ECMWF-sampled tracks by default)')
    ap.add_argument('--synth-nc', default='',
                    help='synthetic_tracks_*.nc (ensemble mode); default from '
                         'config.txt: {storm}_synth_gefs_nc / {storm}_synth_ecmwf_nc')
    ap.add_argument('--gefs-init', default='',
                    help='GEFS forecast init time (ensemble mode); default '
                         'from config.txt gefs_init_{storm} or gefs_init')
    ap.add_argument('--gefs-dir', default='',
                    help='local GEFS nc dir (ensemble mode); default from '
                         'config.txt gefs_dir_{storm} or gefs_{storm}_dir')
    ap.add_argument('--members', type=int, default=1000,
                    help='number of ensemble members to run')
    ap.add_argument('--assign', default='auto',
                    choices=['auto', 'ecmwf', 'gefs', 'ecmwf_field',
                             'round_robin'],
                    help='env-member assignment (ensemble mode). auto: gefs '
                         'for --env gefs, ecmwf otherwise. ecmwf = uniform '
                         'one-to-one hash ECMWF parents -> GEFS members; '
                         'gefs = parent GEFS member directly; ecmwf_field = '
                         'ECMWF parent code directly (future --env ecmwf); '
                         'round_robin = balanced fallback')
    ap.add_argument('--out-root', default='',
                    help='ensemble output root (default data/ensemble/{storm})')
    ap.add_argument('--duration-h', type=float, default=240.0,
                    help='forecast duration in hours per member')
    ap.add_argument('--vortex-mode', default='annulus',
                    choices=['annulus', 'surgery'],
                    help='vortex removal for env winds: annulus (200-800 km '
                         'mean, ODE training convention, default) or surgery '
                         '(strict Lin et al. vortex surgery on full-global '
                         'fields; any failure rejects the member)')
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    cfg_storms = [s.strip() for s in cfg.get('storms', '').split(',') if s.strip() and s.strip().upper() != 'ALL']
    if not args.storms:
        args.storms = ','.join(cfg_storms)

    if args.ensemble:
        # Per-storm config resolution: keys are lowercase storm-dir names
        # without the leading id, e.g. flossie / beryl (from
        # 2025180N13261_FLOSSIE -> 'flossie').
        storms_arg = [s for s in args.storms.split(',') if s.strip()]
        tag = storms_arg[0].split('_', 1)[-1].lower() if storms_arg else 'storm'
        if not args.gefs_init:
            args.gefs_init = (cfg.get(f'gefs_init_{tag}')
                              or cfg.get('gefs_init', ''))
        if not args.gefs_dir:
            args.gefs_dir = (cfg.get(f'gefs_dir_{tag}')
                             or cfg.get(f'gefs_{tag}_dir')
                             or cfg.get('gefs_beryl_dir', ''))
        if not args.synth_nc:
            key = (f'synth_gefs_nc_{tag}' if args.env == 'gefs'
                   else f'synth_ecmwf_nc_{tag}')
            args.synth_nc = (cfg.get(key)
                             or cfg.get('synth_gefs_nc' if args.env == 'gefs'
                                        else 'synth_ecmwf_nc', ''))
        if not args.synth_nc:
            print('--ensemble requires --synth-nc (or config '
                  f'synth_gefs_nc_{tag} / synth_ecmwf_nc_{tag})')
            return
        if args.assign == 'auto':
            args.assign = 'gefs' if args.env == 'gefs' else 'ecmwf'
        if args.assign == 'ecmwf_field' and args.env != 'ecmwf':
            print('[ens] NOTE: assign=ecmwf_field pairs tracks with their ECMWF '
                  'parent member; use it once --env ecmwf (ECMWF forecast '
                  'fields) is available. Continuing (code is recorded in '
                  'member_assignment.txt only).')
        if not args.out_root:
            # carry the full experiment configuration in the directory name:
            # {storm}_{env-source}_{vortex-removal}[_{n}m]
            mem_tag = '' if args.members >= 1000 else f'_{args.members}m'
            args.out_root = str(PROJECT_ROOT / 'data' / 'ensemble'
                                / f'{tag}_{args.env}_{args.vortex_mode}'
                                  f'{mem_tag}')
        if not args.workers:
            args.workers = int(cfg.get('n_workers', 4))
        if args.stage in ('prep,ode', ''):
            args.stage = 'eprep,ode,plot'
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
