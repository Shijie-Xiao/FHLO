#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finish the GEFS BERYL crop: remaining pgrb2b members (p24..p30).

Same regional crop as the original run (lat 3.5-48.5, lon 258.5-326 degE,
91x136 on the 0.5-deg grid), with the p24 crash fixed:
  - cfgrib attaches scalar coords like 'surface'/'heightAboveGround' to the
    2-D fields; different fhours label them differently, which breaks
    xr.concat. Fix: reset_coords(drop=True) before merging, concat with
    coords='minimal', compat='override'.
Resumable: skips members whose output file already exists.
"""
import os
import sys

import xarray as xr

LAT0, LAT1 = 3.5, 48.5
LON0, LON1 = 258.5, 326.0
# GEFS GRIB2 archive root (override: env FHLO_GEFS_ROOT or config.txt gefs_root)
GEFS_ROOT = os.environ.get('FHLO_GEFS_ROOT',
                           '/global/cfs/cdirs/m5011/Jay/ERA5/GFS/2024_BERYL_NA/grib2')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'gefs_beryl')

MEMBERS = ['c00'] + [f'p{i:02d}' for i in range(1, 31)]
FHOURS = list(range(0, 241, 3))


def crop_member(stream, mem):
    out_f = f'{OUT}/{stream}_{mem}.nc'
    if os.path.exists(out_f):
        print(f'  {stream}/{mem}: exists, skip', flush=True)
        return
    import cfgrib
    prefix = 'gec00' if mem == 'c00' else f'ge{mem}'
    parts = []
    for fh in FHOURS:
        f = f'{GEFS_ROOT}/{stream}/{prefix}.t12z.{stream}.0p50.f{fh:03d}.grib2'
        if not os.path.exists(f):
            continue
        try:
            dss = cfgrib.open_datasets(f, indexpath='')
        except Exception as e:
            print(f'  [err] {stream}/{mem} f{fh:03d}: {str(e)[:100]}', flush=True)
            continue
        merged = None
        for ds in dss:
            keep = {}
            for v in ds.data_vars:
                da = ds[v]
                if 'isobaricInhPa' in da.dims or v in ('prmsl', 'sp', 't'):
                    da = da.sel(latitude=slice(LAT1, LAT0),
                                longitude=slice(LON0, LON1))
                    keep[v] = da
            if not keep:
                continue
            dsub = xr.Dataset(keep).reset_coords(drop=True)
            merged = dsub if merged is None else xr.merge(
                [merged, dsub], compat='override', join='outer')
        if merged is not None:
            merged = merged.reset_coords(drop=True)
            merged = merged.assign_coords(fhour=fh).expand_dims('fhour')
            parts.append(merged)
    if not parts:
        print(f'  [empty] {stream}/{mem}', flush=True)
        return
    out = xr.concat(parts, dim='fhour', coords='minimal', compat='override')
    enc = {v: {'zlib': True, 'complevel': 4} for v in out.data_vars}
    tmp = out_f + '.tmp'
    out.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out_f)
    print(f'  {stream}/{mem}: {dict(out.sizes)}', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    streams = [a for a in args if a in ('pgrb2b', 'pgrb2a', 'all')] or ['pgrb2b']
    members = [a for a in args if a not in ('pgrb2b', 'pgrb2a', 'all')] or None
    for stream in ['pgrb2b', 'pgrb2a']:
        if stream not in streams and 'all' not in streams:
            continue
        todo = members if members else MEMBERS
        for mem in todo:
            crop_member(stream, mem)
    print('GEFS crop DONE', flush=True)
