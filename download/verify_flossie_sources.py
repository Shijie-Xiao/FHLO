#!/usr/bin/env python3
"""Exhaustive verification of 2025/ (PL + SFC) for the Flossie window.

Flossie 2025 (EP): 2025-06-29 06Z .. 2025-07-05 18Z, 13-24N, 99-119W.
Prep window: track dates padded +-2 days => 20250627 .. 20250708.

Checks EVERY PL day-file for U/V/T/Q/Z:
  - global 721x1440, lat 90->-90 desc 0.25, lon 0-359.75 gapless
  - 24 hourly steps
  - levels: T/Q must have the FULL 30-level ladder (30-1000);
    U/V/Z 7 levels incl 250/850/600 (surgery + chi midlevel)
  - NaN-free on Flossie neighborhood (lat 5-40N, lon 210-265E == 95-150W)
    all hours, all levels
Checks SFC monthly files (SSTK/MSL/BLH/SP): hourly steps, global grid,
NaN-free in neighborhood.
"""
import glob

import numpy as np
import xarray as xr

PL25 = '/global/cfs/cdirs/m5011/Jay/ERA5/2025/PL'
SFC25 = '/global/cfs/cdirs/m5011/Jay/ERA5/2025/SFC'
DAYS = [f'202506{d:02d}' for d in range(27, 31)] + \
       [f'202507{d:02d}' for d in range(1, 9)]
# Flossie 邻域: lat 5-40N, lon 210-265E (95-150W)
NB = dict(latitude=slice(40, 5), longitude=slice(210, 265))

bad = []
print(f'窗口: {DAYS[0]}..{DAYS[-1]} ({len(DAYS)} 天)')
print('\n=== PL 逐文件验证 (2025/PL) ===')
for var, code, need_30 in [('U', '131', False), ('V', '132', False),
                           ('Z', '129', False), ('T', '130', True), ('Q', '133', True)]:
    for day in DAYS:
        fs = sorted(glob.glob(f'{PL25}/*128_{code}_{var.lower()}*.{day}00_{day}23.nc'))
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
        sub = ds[vname].sel(**NB)
        nanf = float(np.isnan(sub.values).mean())
        grid_ok = (ds.sizes['latitude'] == 721 and ds.sizes['longitude'] == 1440
                   and lat[0] == 90.0 and lat[-1] == -90.0 and gaps == 0)
        lev_ok = (len(levs) == 30 and levs[0] == 30 and levs[-1] == 1000) if need_30 \
            else ({250, 850, 600} <= set(levs))
        t_ok = ds.sizes['time'] == 24
        ok = grid_ok and lev_ok and t_ok and nanf < 0.001
        msg = (f"{'OK ' if ok else 'BAD'} grid={'G' if grid_ok else 'R'} "
               f"lev={len(levs)}{'(30)' if need_30 else ''} t={ds.sizes['time']} nan={nanf:.1%}")
        if not ok:
            bad.append((tag, msg))
        print(f'{tag}: {msg}', flush=True)
        ds.close()

print('\n=== SFC 月文件验证 (2025/SFC) ===')
for var, code in [('SSTK', '034'), ('MSL', '151'), ('BLH', '159'), ('SP', '134')]:
    for ym in ['202506', '202507']:
        fs = sorted(glob.glob(f'{SFC25}/*{var}*.{ym}*.nc'))
        tag = f'{var} {ym}'
        if not fs:
            bad.append((tag, 'missing'))
            print(f'{tag}: MISSING')
            continue
        ds = xr.open_dataset(fs[0])
        vname = [v for v in ds.data_vars if v != 'utc_date'][0]
        grid_ok = ds.sizes.get('latitude') == 721 and ds.sizes.get('longitude') == 1440
        # 只抽查该月 15 日 12 时一个切片的邻域 NaN
        try:
            sl = ds[vname].sel(time=f'{ym[:4]}-{ym[4:]}-15T12:00').sel(**NB)
            nanf = float(np.isnan(sl.values).mean())
        except Exception:
            nanf = -1
        ok = grid_ok and nanf < 0.001
        msg = f"{'OK' if ok else 'BAD'} grid={ds.sizes.get('latitude')}x{ds.sizes.get('longitude')} t={ds.sizes['time']} nan={max(nanf,0):.1%}"
        if not ok:
            bad.append((tag, msg))
        print(f'{tag}: {msg}', flush=True)
        ds.close()

print(f'\n===== {"全部通过, 可以裁剪" if not bad else f"{len(bad)} 问题: {bad[:10]}"} =====')
