#!/usr/bin/env python3
"""Rebuild data/era5 for Beryl: GLOBAL U/V/Z + 30-level T/Q + SFC.

Sources (verified 2026-08-23, Beryl window 20240625-20240713):
  U/V/Z : Test/PL/{YYYYMM}/e5.oper.an.pl.128_{code}_{v}.ll025*.nc
          global 721x1440, lat 90->-90 DESC, lon 0-359.75 no gap, 10 levels
          (200/250/300/400/500/600/700/850/925/1000) - surgery needs 250/850,
          chi needs 600: all present. FULL SPATIAL DOMAIN for vortex_lib.
  T/Q   : NA/{T,Q}/*.nc 30-level (30-1000hPa) regional 321x601 (lat 80->0,
          lon gap 20E-230E but Atlantic covered). PI/chi need the full
          pressure ladder -> keep 30 levels; annulus means don't need globe.
  SSTK/MSL/BLH: NA/ monthly files, regional 321x601, monthly all hours.

Outputs (matches prep/_find_pl 'local flat layout' + sfc loader):
  data/era5/{U,V,Z,T,Q}_{YYYYMMDD}.nc  daily PL
  data/era5/{SSTK,MSL,BLH}_{YYYYMM}.nc monthly SFC
float32 + zlib. Atomic .tmp replace; resumable (skip if exists & valid).
"""
import glob
import os
import sys

import numpy as np
import xarray as xr

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PL = '/global/cfs/cdirs/m5011/Jay/ERA5/Test/PL'
NA = '/global/cfs/cdirs/m5011/Jay/ERA5/NA'
OUT = os.path.join(PROJECT, 'data', 'era5')

DAYS = [f'202406{d:02d}' for d in range(25, 31)] + \
       [f'202407{d:02d}' for d in range(1, 14)]
MONTHS = ['202406', '202407']

GLOBAL_VARS = {'U': '131', 'V': '132', 'Z': '129'}   # from Test/PL (global)
REGIONAL_VARS = {'T': '130', 'Q': '133'}             # from NA (30 levels)
SFC_VARS = {'SSTK': '034', 'MSL': '151', 'BLH': '159'}


def _src_global(var, code, day):
    fs = sorted(glob.glob(f'{TEST_PL}/{day[:6]}/e5.oper.an.pl.128_{code}_'
                          f'{var.lower()}.*ll025*.{day}00_{day}23.nc'))
    return fs[0] if fs else None


def _src_regional(var, day):
    fs = sorted(glob.glob(f'{NA}/{var}/*.{var}.*.{day}00_{day}23.nc')) or \
         sorted(glob.glob(f'{NA}/{var}/*.{var.lower()}.*.{day}00_{day}23.nc'))
    return fs[0] if fs else None


def _valid(out_f, want_global):
    if not os.path.exists(out_f):
        return False
    try:
        ds = xr.open_dataset(out_f)
        nlat = ds.sizes.get('latitude', 0)
        ds.close()
        return (nlat == 721) if want_global else (nlat > 0)
    except Exception:
        return False


def crop_pl(var, code, day, want_global):
    out_f = os.path.join(OUT, f'{var}_{day}.nc')
    if _valid(out_f, want_global):
        return True, 'exists'
    src = _src_global(var, code, day) if want_global else _src_regional(var, day)
    if src is None:
        return False, 'missing-source'
    try:
        ds = xr.open_dataset(src)
        keep = [v for v in ds.data_vars if v != 'utc_date']
        out = ds[keep].load()
        ds.close()
        enc = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for v in keep}
        out.to_netcdf(out_f + '.tmp', encoding=enc)
        os.replace(out_f + '.tmp', out_f)
        return True, f'{dict(out.sizes)}'
    except Exception as e:
        for f in (out_f, out_f + '.tmp'):
            if os.path.exists(f):
                os.remove(f)
        return False, f'{type(e).__name__}:{str(e)[:60]}'


def crop_sfc(var, ym):
    out_f = os.path.join(OUT, f'{var}_{ym}.nc')
    if _valid(out_f, False):
        return True, 'exists'
    fs = sorted(glob.glob(f'{NA}/{var}/*.{var}.*.{ym}01*.nc')) or \
         sorted(glob.glob(f'{NA}/{var}/*.{var.lower()}.*.{ym}01*.nc'))
    if not fs:
        return False, 'missing-source'
    try:
        ds = xr.open_dataset(fs[0])
        keep = [v for v in ds.data_vars if v != 'utc_date']
        out = ds[keep].load()
        ds.close()
        enc = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for v in keep}
        out.to_netcdf(out_f + '.tmp', encoding=enc)
        os.replace(out_f + '.tmp', out_f)
        return True, f'{dict(out.sizes)}'
    except Exception as e:
        return False, f'{type(e).__name__}:{str(e)[:60]}'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print(f'OUT={OUT}', flush=True)
    fail = []
    for var, code in GLOBAL_VARS.items():
        for day in DAYS:
            ok, msg = crop_pl(var, code, day, want_global=True)
            print(f'  {var} {day}: {msg}', flush=True)
            if not ok:
                fail.append((var, day))
    for var, code in REGIONAL_VARS.items():
        for day in DAYS:
            ok, msg = crop_pl(var, code, day, want_global=False)
            print(f'  {var} {day}: {msg}', flush=True)
            if not ok:
                fail.append((var, day))
    for var in SFC_VARS:
        for ym in MONTHS:
            ok, msg = crop_sfc(var, ym)
            print(f'  {var} {ym}: {msg}', flush=True)
            if not ok:
                fail.append((var, ym))
    print(f'CROP {"DONE" if not fail else "FAILED: " + str(fail)}', flush=True)
    sys.exit(1 if fail else 0)
