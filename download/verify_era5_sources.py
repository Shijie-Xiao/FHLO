#!/usr/bin/env python3
"""Full scan of ERA5 source files for the Beryl window before cropping.

Checks EVERY candidate file for: existence, grid (global 721x1440, lat
90->-90 desc 0.25, lon 0-359.75 no gap), time coverage (24 hourly steps),
levels (PL: which pressure levels), NaN fraction on one slab.

Sources:
  PL U/V/Z (global): /global/cfs/cdirs/m5011/Jay/ERA5/Test/PL/{YYYYMM}/
  PL T/Q   (30 lev): /global/cfs/cdirs/m5011/Jay/ERA5/Test/PL/{YYYYMM}/ (if global)
                     else /global/cfs/cdirs/m5011/Jay/ERA5/NA/{T,Q}/ (regional 30 lev)
  SFC SSTK/MSL/BLH : /global/cfs/cdirs/m5011/Jay/ERA5/2025/SFC (wrong year - check NA)
"""
import glob
import os
import sys

import numpy as np
import xarray as xr

TEST_PL = '/global/cfs/cdirs/m5011/Jay/ERA5/Test/PL'
NA = '/global/cfs/cdirs/m5011/Jay/ERA5/NA'
DAYS = [f'202406{d:02d}' for d in range(25, 31)] + \
       [f'202407{d:02d}' for d in range(1, 14)]

pl_files = {}
for var, code in [('U', '131'), ('V', '132'), ('T', '130'), ('Q', '133'), ('Z', '129')]:
    for day in DAYS:
        ym = day[:6]
        cands = sorted(glob.glob(f'{TEST_PL}/{ym}/e5.oper.an.pl.128_{code}_*.'
                                 f'll025*.{day}00_{day}23.nc'))
        pl_files[(var, day)] = cands

print('=== PL 文件存在性 (Test/PL 全球域) ===')
miss = []
for var in ['U', 'V', 'T', 'Q', 'Z']:
    n_ok = sum(1 for d in DAYS if pl_files.get((var, d)))
    n_all = len(DAYS)
    if n_ok < n_all:
        miss += [(var, d) for d in DAYS if not pl_files.get((var, d))]
    print(f'{var}: {n_ok}/{n_all} 天', flush=True)
print('缺失:', miss if miss else '无')

print('\n=== 逐文件网格/层次/时段抽查 + 全量存在性 ===')
issues = []
for (var, day), fs in sorted(pl_files.items()):
    if not fs:
        continue
    try:
        ds = xr.open_dataset(fs[0])
        lat = ds.latitude.values; lon = ds.longitude.values
        d = np.diff(lon)
        gaps = np.where(np.abs(d - 0.25) > 1e-6)[0]
        lat_desc = lat[0] == 90.0 and lat[-1] == -90.0
        global_ok = (len(lat) == 721 and len(lon) == 1440 and len(gaps) == 0 and lat_desc)
        t24 = ds.sizes.get('time') == 24
        nlev = ds.sizes.get('level', 0)
        # NaN 抽查: 第一个时次、850 层(或最接近)
        if nlev:
            levs = ds.level.values
            li = int(np.argmin(np.abs(np.asarray(levs) - 850)))
            varname = [v for v in ds.data_vars if v not in ('utc_date',)][0]
            slab = ds[varname].isel(time=0, level=li).values
            nanf = float(np.isnan(slab).mean())
        else:
            nanf = -1
        ds.close()
        status = []
        if not global_ok:
            status.append(f'GRID(lat={len(lat)},lon={len(lon)},desc={lat_desc},gaps={len(gaps)})')
        if not t24:
            status.append(f'TIME={ds.sizes.get("time")}')
        if nanf > 0.01:
            status.append(f'NAN={nanf:.2%}')
        if status:
            issues.append((var, day, ' '.join(status)))
        if day in ('20240628', '20240701', '20240710'):  # 抽样打印
            print(f'{var} {day}: {nlev}lev global={global_ok} t24={t24} nan={nanf:.1%}',
                  flush=True)
    except Exception as e:
        issues.append((var, day, f'OPEN-FAIL {type(e).__name__}: {str(e)[:60]}'))

print('\n=== 异常清单 ===')
if issues:
    for it in issues[:30]:
        print(' ', it)
else:
    print('  全部通过: 全球 721x1440 / lat 90->-90 / lon 无缺口 / 24h / NaN<1%')

print('\n=== T/Q 层次详情 (PI 需要完整 30 层) ===')
for var in ['T', 'Q']:
    fs = pl_files[(var, '20240701')]
    if fs:
        ds = xr.open_dataset(fs[0])
        levs = sorted(ds.level.values)
        print(f'{var}: {len(levs)} 层: {[int(l) for l in levs]}')
        ds.close()
    else:
        print(f'{var}: Test/PL 无, 需从 NA/ 拿 30 层区域版本')
print('\nVERIFY DONE')
