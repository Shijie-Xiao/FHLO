#!/usr/bin/env python3
"""FHLO end-to-end pipeline: best track -> 1h interpolation -> env fields -> FAST ODE.

Stages
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


# ---------- Pipeline ----------

def main():
    ap = argparse.ArgumentParser(description='FHLO pipeline')
    ap.add_argument('--config', default=str(PROJECT_ROOT / 'config.txt'))
    ap.add_argument('--storms', default='', help='comma-separated storm dir names')
    ap.add_argument('--sst', default='', help='SST source override: ERA5 | OISST')
    ap.add_argument('--stage', default='prep,ode',
                    help='stages to run: ibtracs,prep,ode (default prep,ode)')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
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
