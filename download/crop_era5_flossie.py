#!/usr/bin/env python3
"""Crop the Flossie 2025 (EP) window from Jay's 2025 global archive into data/era5.

This REPLACES the old Beryl regional files - data/era5 becomes the demo
dataset (user: later demos reuse it directly).

All sources are the FULL-GLOBAL 721x1440 (lat 90->-90 desc 0.25, lon 0-359.75
gapless) verified per-file before this run:
  2025/PL/{var}: daily U/V/Z (7 lev incl 250/850/600) + T/Q (FULL 30 lev)
  2025/SFC/{var}: monthly SSTK/MSL/BLH/SP (SSTK NaN over land is expected)

Outputs (names match prep/_find_pl local flat layout + sfc loader):
  data/era5/{U,V,Z,T,Q}_{YYYYMMDD}.nc   daily PL, global
  data/era5/{SSTK,MSL,BLH,SP}_{YYYYMM}.nc monthly SFC, global
float32 + zlib, atomic replace, resumable.
"""
import glob
import os
import sys

import xarray as xr

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL25 = '/global/cfs/cdirs/m5011/Jay/ERA5/2025/PL'
SFC25 = '/global/cfs/cdirs/m5011/Jay/ERA5/2025/SFC'
OUT = os.path.join(PROJECT, 'data', 'era5')

DAYS = [f'202506{d:02d}' for d in range(27, 31)] + \
       [f'202507{d:02d}' for d in range(1, 9)]
MONTHS = ['202506', '202507']
PL_VARS = {'U': '131', 'V': '132', 'Z': '129', 'T': '130', 'Q': '133'}
SFC_VARS = {'SSTK': '034', 'MSL': '151', 'BLH': '159', 'SP': '134'}


def _done(out_f):
    if not os.path.exists(out_f):
        return False
    try:
        ds = xr.open_dataset(out_f)
        ok = ds.sizes.get('latitude') == 721
        ds.close()
        return ok
    except Exception:
        return False


def crop(kind, var, code, key):
    if kind == 'pl':
        out_f = os.path.join(OUT, f'{var}_{key}.nc')
        srcs = sorted(glob.glob(f'{PL25}/*128_{code}_{var.lower()}*.{key}00_{key}23.nc'))
    else:
        out_f = os.path.join(OUT, f'{var}_{key}.nc')
        srcs = sorted(glob.glob(f'{SFC25}/*{var}*.{key}01*.nc'))
    if _done(out_f):
        return True, 'exists'
    if not srcs:
        return False, 'missing-source'
    try:
        ds = xr.open_dataset(srcs[0])
        # keep only the main float field(s); skip utc_date / quantization_info
        # (a |S1 char var that float32-encoding chokes on)
        keep = [v for v in ds.data_vars
                if ds[v].dtype.kind == 'f' and v != 'utc_date']
        out = ds[keep].load()
        ds.close()
        enc = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for v in keep}
        out.to_netcdf(out_f + '.tmp', encoding=enc)
        os.replace(out_f + '.tmp', out_f)
        return True, str(dict(out.sizes))
    except Exception as e:
        for f in (out_f, out_f + '.tmp'):
            if os.path.exists(f):
                os.remove(f)
        return False, f'{type(e).__name__}:{str(e)[:50]}'


if __name__ == '__main__':
    # wipe stale Beryl-era files (regional grid) so the dir is purely Flossie demo
    os.makedirs(OUT, exist_ok=True)
    stale = [f for f in os.listdir(OUT) if f.endswith('.nc')
             and any(f.startswith(v + '_') for v in list(PL_VARS) + list(SFC_VARS))
             and not any(f == f'{v}_{k}.nc' for v in list(PL_VARS) + list(SFC_VARS)
                         for k in DAYS + MONTHS)]
    for f in stale:
        os.remove(os.path.join(OUT, f))
    if stale:
        print(f'removed {len(stale)} stale Beryl files', flush=True)

    fail = []
    for var, code in PL_VARS.items():
        for day in DAYS:
            ok, msg = crop('pl', var, code, day)
            print(f'  {var} {day}: {msg}', flush=True)
            if not ok:
                fail.append((var, day))
    for var, code in SFC_VARS.items():
        for ym in MONTHS:
            ok, msg = crop('sfc', var, code, ym)
            print(f'  {var} {ym}: {msg}', flush=True)
            if not ok:
                fail.append((var, ym))
    print(f'FLOSSIE ERA5 CROP {"DONE" if not fail else "FAILED: " + str(fail)}', flush=True)
    sys.exit(1 if fail else 0)
