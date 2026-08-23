#!/usr/bin/env python3
"""EXHAUSTIVE per-file verification of U/V/Z (global) and T/Q (30-level).

U/V/Z from Test/PL: every file must be 721x1440, lat 90->-90 desc 0.25,
lon 0-359.75 gapless, 24 hourly steps, 10 levels incl 250/850/600, and
NaN-free on sampled slabs (250 & 850, first+last hour).

T/Q from NA: every file must be 30 levels (30-1000), 24 hourly steps,
NaN-free at Beryl neighborhood (lat 5-30N, lon 260-320), and dates complete.

Also verifies SSTK/MSL/BLH monthly SFC files exist with hourly steps.
Prints one line per file so nothing slips through.
"""
import glob
import re

import numpy as np
import xarray as xr

TEST_PL = '/global/cfs/cdirs/m5011/Jay/ERA5/Test/PL'
NA = '/global/cfs/cdirs/m5011/Jay/ERA5/NA'
DAYS = [f'202406{d:02d}' for d in range(25, 31)] + \
       [f'202407{d:02d}' for d in range(1, 14)]

bad = []

print('=== U/V/Z 全球文件逐个验证 (Test/PL) ===')
for var, code in [('U', '131'), ('V', '132'), ('Z', '129')]:
    for day in DAYS:
        fs = sorted(glob.glob(f'{TEST_PL}/{day[:6]}/e5.oper.an.pl.128_{code}_'
                              f'{var.lower()}.*ll025*.{day}00_{day}23.nc'))
        tag = f'{var} {day}'
        if not fs:
            bad.append((tag, 'missing'))
            print(f'{tag}: MISSING')
            continue
        ds = xr.open_dataset(fs[0])
        lat, lon = ds.latitude.values, ds.longitude.values
        gaps = int((np.abs(np.diff(lon) - 0.25) > 1e-6).sum())
        levs = sorted(int(l) for l in ds.level.values)
        vname = [v for v in ds.data_vars if v != 'utc_date'][0]
        # NaN 抽查: 250/850 层首末时次全网格
        nanf = 0.0
        for lev in (250, 850):
            li = int(np.argmin(np.abs(np.asarray(levs) - lev)))
            for ti in (0, 23):
                nanf = max(nanf, float(np.isnan(
                    ds[vname].isel(time=ti, level=li).values).mean()))
        ok = (ds.sizes['latitude'] == 721 and ds.sizes['longitude'] == 1440
              and lat[0] == 90.0 and lat[-1] == -90.0 and gaps == 0
              and ds.sizes['time'] == 24 and {250, 850, 600} <= set(levs)
              and nanf < 0.001)
        msg = (f"{'OK ' if ok else 'BAD'} "
               f"{ds.sizes['latitude']}x{ds.sizes['longitude']} gaps={gaps} "
               f"t={ds.sizes['time']} lev={len(levs)} nan={nanf:.1%}")
        if not ok:
            bad.append((tag, msg))
        print(f'{tag}: {msg}', flush=True)
        ds.close()

print('\n=== T/Q 30层文件逐个验证 (NA) ===')
for var in ['T', 'Q']:
    for day in DAYS:
        fs = sorted(glob.glob(f'{NA}/{var}/*.{var}.*.{day}00_{day}23.nc'))
        tag = f'{var} {day}'
        if not fs:
            bad.append((tag, 'missing'))
            print(f'{tag}: MISSING')
            continue
        ds = xr.open_dataset(fs[0])
        levs = sorted(int(l) for l in ds.level.values)
        vname = [v for v in ds.data_vars if v != 'utc_date'][0]
        # NaN 抽查: Beryl 邻域 (5-30N, 260-320E) 全时次全部层
        sub = ds[vname].sel(latitude=slice(30, 5), longitude=slice(260, 320))
        nanf = float(np.isnan(sub.values).mean())
        ok = (len(levs) == 30 and levs[0] == 30 and levs[-1] == 1000
              and ds.sizes['time'] == 24 and nanf < 0.001)
        msg = f"{'OK ' if ok else 'BAD'} lev={len(levs)}({levs[0]}-{levs[-1]}) t={ds.sizes['time']} nan={nanf:.1%}"
        if not ok:
            bad.append((tag, msg))
        print(f'{tag}: {msg}', flush=True)
        ds.close()

print('\n=== SFC 月文件 (NA) ===')
for var in ['SSTK', 'MSL', 'BLH']:
    for ym in ['202406', '202407']:
        fs = sorted(glob.glob(f'{NA}/{var}/*.{var}.*.{ym}01*.nc'))
        if not fs:
            bad.append((f'{var} {ym}', 'missing'))
            print(f'{var} {ym}: MISSING')
            continue
        ds = xr.open_dataset(fs[0])
        vname = [v for v in ds.data_vars if v != 'utc_date'][0]
        nt = ds.sizes['time']
        print(f'{var} {ym}: {"OK" if nt >= 24 else "BAD"} t={nt}', flush=True)
        ds.close()

print(f'\n===== 结论: {"全部通过, 可以裁剪" if not bad else f"{len(bad)} 个问题: {bad[:10]}"} =====')
