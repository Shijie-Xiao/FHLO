#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crop ERA5 demo data BY DATE ONLY (full spatial domain kept) to FHLO/data/era5.

Rationale: vortex_lib surgery needs a full 25-deg box around the storm center,
so regional crops break storms near the domain edge. Here we only subset time
and keep the full spatial grid (lat 0-80N, lon 0-360E) so the demo data works
for any NA/EP storm and can be shipped as a self-contained demo.

Sources (original full-domain archives stay untouched):
  ERA5 NA/EP /global/cfs/cdirs/m5011/Jay/ERA5/{NA,EP}/{T,Q,U,V,Z,BLH,MSL,SSTK}
  OISST/GHRSST /global/cfs/cdirs/m5011/Jay/OHC/{year}/{MM}/OHC-{NA,NP}QG3_*.nc

Output layout (local flat, consumed by prep/prepare_complete_training_data.py):
  data/era5/{T,Q,U,V,Z}_{YYYYMMDD}.nc   daily PL, all levels, full domain
  data/era5/{SSTK,MSL,BLH}_{YYYYMM}.nc  monthly SFC, full domain
  data/oisst/{YYYYMMDD}.nc              daily OISST sst (degC) on its native grid

Usage:
  python crop_beryl_sample.py                                # Beryl default
  python crop_beryl_sample.py NA 2024 2024181N09320_BERYL
  python crop_beryl_sample.py EP 2015 2015322N13253_RICK
  --vars T,Q,U     subset of variables (default all)
"""
import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Community archive roots (override: env FHLO_ERA5_ROOT/FHLO_OISST_ROOT or config.txt)
ERA5_ROOT = os.environ.get('FHLO_ERA5_ROOT', '/global/cfs/cdirs/m5011/Jay/ERA5')
OISST_ROOT = os.environ.get('FHLO_OISST_ROOT', '/global/cfs/cdirs/m5011/Jay/OHC')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ERA5 = PROJECT_ROOT / 'data' / 'era5'
OUT_OISST = PROJECT_ROOT / 'data' / 'oisst'

# Storm dates padded by +-2 days (covers the 24h SST lag used in prep).
PAD_DAYS = 2
PL_VARS = ['T', 'Q', 'U', 'V', 'Z']
SFC_VARS = ['BLH', 'MSL', 'SSTK']
ERA5_SFC_BASIN = {'NA': 'NA', 'EP': 'NA'}   # EP uses NA SFC archive (domain overlap)
OISST_BASIN = {'NA': 'NA', 'EP': 'NP'}


def storm_dates(track_csv):
    """All distinct dates covered by the 6h track, padded by PAD_DAYS."""
    df = pd.read_csv(track_csv)
    t = pd.to_datetime(df['time'])
    d0 = (t.min() - pd.Timedelta(days=PAD_DAYS)).normalize()
    d1 = (t.max() + pd.Timedelta(days=PAD_DAYS)).normalize()
    return sorted({d.strftime('%Y%m%d') for d in pd.date_range(d0, d1)})


def _crop_one_pl(args):
    """Worker: crop one PL (var, date). Returns (ok, var, date, msg)."""
    var, d, basin = args
    out_f = OUT_ERA5 / f'{var}_{d}.nc'
    if out_f.exists():
        return True, var, d, 'exists'
    srcs = sorted(glob.glob(f'{ERA5_ROOT}/{basin}/{var}/*{d}00_{d}23.nc'))
    if not srcs:
        return False, var, d, 'missing'
    try:
        src = srcs[-1]  # latest download batch wins (full-level version)
        ds = xr.open_dataset(src)
        v = next((k for k in ds.data_vars if k != 'utc_date'), None)
        if v is None:
            ds.close()
            return False, var, d, 'no-var'
        # time-only crop: keep full spatial domain + all levels
        sub = ds[[v]].rename({v: var})
        sub.to_netcdf(out_f, encoding={var: {'zlib': True, 'complevel': 4}})
        ds.close()
        return True, var, d, 'ok'
    except Exception as e:
        return False, var, d, str(e)[:80]


def _crop_one_sfc(args):
    """Worker: crop one SFC (var, month). Time-subset only, full grid."""
    var, ym, t0, t1, basin = args
    out_f = OUT_ERA5 / f'{var}_{ym}.nc'
    if out_f.exists():
        return True, var, ym, 'exists'
    srcs = sorted(glob.glob(f'{ERA5_ROOT}/{basin}/{var}/*{ym}*.nc'))
    if not srcs:
        return False, var, ym, 'missing'
    try:
        src = srcs[-1]
        ds = xr.open_dataset(src)
        v = next((k for k in ds.data_vars if k != 'utc_date'), None)
        if v is None:
            ds.close()
            return False, var, ym, 'no-var'
        sel = ds[[v]].sel(time=slice(t0, t1)).rename({v: var})
        if sel.sizes.get('time', 0) == 0:
            ds.close()
            return False, var, ym, 'empty-time'
        sel.to_netcdf(out_f, encoding={var: {'zlib': True, 'complevel': 4}})
        ds.close()
        return True, var, ym, 'ok'
    except Exception as e:
        return False, var, ym, str(e)[:80]


def _crop_one_oisst(args):
    """Worker: crop one OISST day -> data/oisst/{YYYYMMDD}.nc (sst degC)."""
    d, basin = args
    out_f = OUT_OISST / f'{d}.nc'
    if out_f.exists():
        return True, d, 'exists'
    y, m = d[:4], d[4:6]
    tag = OISST_BASIN[basin]
    srcs = sorted(glob.glob(f'{OISST_ROOT}/{y}/{m}/OHC-{tag}QG3_*.nc'))
    if not srcs:
        return False, d, 'missing'
    tgt = pd.Timestamp(f'{y}-{m}-{d[6:8]}')
    try:
        picked = None
        for f in srcs:
            m2 = re.search(r'_s(\d{8})\d*_e(\d{8})', Path(f).name)
            if not m2:
                continue
            if m2.group(1) <= d and d <= m2.group(2):
                picked = f
                break
        if picked is None:
            return False, d, 'no-cover'
        ds = xr.open_dataset(picked)
        # Each file covers 14 days from s; pick the exact day by date offset.
        m2 = re.search(r'_s(\d{8})', Path(picked).name)
        s0 = pd.Timestamp(f'{m2.group(1)[:4]}-{m2.group(1)[4:6]}-{m2.group(1)[6:8]}')
        idx = int((tgt - s0).days)
        da = ds['sst'].isel(time=min(idx, ds.sizes['time'] - 1))
        da = da.drop_vars(['crs', 'landmask', 'quality_information'], errors='ignore')
        da = da.rename('sst').to_dataset()
        da.attrs['source'] = f'GHRSST/OHC-{tag}QG3 {Path(picked).name}'
        da.to_netcdf(out_f, encoding={'sst': {'zlib': True, 'complevel': 4}})
        ds.close()
        return True, d, 'ok'
    except Exception as e:
        return False, d, str(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('basin', nargs='?', default='NA')
    ap.add_argument('year', nargs='?', default='2024')
    ap.add_argument('storm', nargs='?', default='2024181N09320_BERYL')
    ap.add_argument('--vars', default='T,Q,U,V,Z,BLH,MSL,SSTK')
    ap.add_argument('--oisst', action='store_true', default=True)
    ap.add_argument('--no-oisst', dest='oisst', action='store_false')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    track_csv = (PROJECT_ROOT / 'data' / 'ibtracs' / args.basin / args.year /
                 args.storm / 'track_intensity_6h.csv')
    if not track_csv.exists():
        sys.exit(f'track not found: {track_csv}')

    dates = storm_dates(track_csv)
    months = sorted({d[:6] for d in dates})
    t_all = pd.to_datetime([f'{d[:4]}-{d[4:6]}-{d[6:8]}' for d in dates])
    t0, t1 = t_all.min(), t_all.max() + pd.Timedelta(hours=23)
    print(f'Storm {args.storm}: {len(dates)} days [{dates[0]} .. {dates[-1]}]')

    want = [v.strip().upper() for v in args.vars.split(',') if v.strip()]
    pl_vars = [v for v in PL_VARS if v in want]
    sfc_vars = [v for v in SFC_VARS if v in want]
    sfc_basin = ERA5_SFC_BASIN[args.basin]

    jobs_pl = [(v, d, args.basin) for v in pl_vars for d in dates]
    jobs_sfc = [(v, ym, t0, t1, sfc_basin) for v in sfc_vars for ym in months]
    jobs_ois = [(d, args.basin) for d in dates] if args.oisst else []
    jobs = jobs_pl + jobs_sfc + jobs_ois
    print(f'Jobs: PL={len(jobs_pl)} SFC={len(jobs_sfc)} OISST={len(jobs_ois)}')

    OUT_ERA5.mkdir(parents=True, exist_ok=True)
    if jobs_ois:
        OUT_OISST.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for j in jobs:
            if len(j) == 3 and isinstance(j[0], str) and j[0] in PL_VARS:
                futs[ex.submit(_crop_one_pl, j)] = j
            elif len(j) == 5:
                futs[ex.submit(_crop_one_sfc, j)] = j
            else:
                futs[ex.submit(_crop_one_oisst, j)] = j
        for fut in as_completed(futs):
            ok, a, b, msg = fut.result()
            n_ok += ok
            n_fail += (not ok)
            if not ok:
                print(f'  [FAIL] {a} {b}: {msg}')
    print(f'DONE: ok={n_ok} fail={n_fail}')


if __name__ == '__main__':
    main()
