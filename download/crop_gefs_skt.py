#!/usr/bin/env python3
"""Extract GEFS surface skin temperature (SST over ocean) for the ensemble.

Source: {gefs_root}/2024_BERYL_NA/grib2/pgrb2b/geXX.t12z.pgrb2b.*.grib2
  typeOfLevel=surface -> t (skin temp, K). Regional crop identical to
  data/gefs_beryl (lat 3.5-48.5, lon 258.5-326, 91x136, 0.5-deg).

Output: data/gefs_beryl/skt_{member}.nc  (fhour, latitude, longitude)
Resumable: skips existing outputs.
"""
import os
import sys

import xarray as xr

LAT0, LAT1 = 3.5, 48.5
LON0, LON1 = 258.5, 326.0
GEFS_ROOT = os.environ.get('FHLO_GEFS_ROOT',
                           '/global/cfs/cdirs/m5011/Jay/ERA5/GFS/2024_BERYL_NA/grib2')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'gefs_beryl')

MEMBERS = ['c00'] + [f'p{i:02d}' for i in range(1, 31)]
FHOURS = list(range(0, 241, 3))


def crop_member(mem):
    out_f = f'{OUT}/skt_{mem}.nc'
    if os.path.exists(out_f):
        print(f'  skt/{mem}: exists, skip', flush=True)
        return
    prefix = 'gec00' if mem == 'c00' else f'ge{mem}'
    parts = []
    for fh in FHOURS:
        f = f'{GEFS_ROOT}/pgrb2b/{prefix}.t12z.pgrb2b.0p50.f{fh:03d}.grib2'
        if not os.path.exists(f):
            continue
        try:
            ds = xr.open_dataset(
                f, engine='cfgrib', backend_kwargs={'indexpath': ''},
                filter_by_keys={'typeOfLevel': 'surface'})
            da = ds['t'].sel(latitude=slice(LAT1, LAT0),
                             longitude=slice(LON0, LON1)).reset_coords(drop=True)
            da = da.rename('skt').assign_coords(fhour=fh).expand_dims('fhour')
            parts.append(da.to_dataset())
            ds.close()
        except Exception as e:
            print(f'  [err] skt/{mem} f{fh:03d}: {str(e)[:100]}', flush=True)
            continue
    if not parts:
        print(f'  [empty] skt/{mem}', flush=True)
        return
    out = xr.concat(parts, dim='fhour', coords='minimal', compat='override')
    out.to_netcdf(out_f + '.tmp',
                  encoding={'skt': {'zlib': True, 'complevel': 4}})
    os.replace(out_f + '.tmp', out_f)
    print(f'  skt/{mem}: {dict(out.sizes)}', flush=True)


if __name__ == '__main__':
    todo = sys.argv[1:] or MEMBERS
    for mem in todo:
        crop_member(mem)
    print('GEFS skt crop DONE', flush=True)
