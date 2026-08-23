#!/usr/bin/env python3
"""GEFS local-NetCDF adapter for the ensemble forecast pipeline.

Serves environment fields for one GEFS ensemble member from the local
pre-converted NetCDF files (data/gefs_beryl/pgrb2a_{m}.nc + pgrb2b_{m}.nc):

  pgrb2b  t/u/v/q/gh (fhour, isobaricInhPa(31), lat, lon)   <- PL fields
  pgrb2a  t/u/v/gh  (fhour, isobaricInhPa(12), lat, lon)
          prmsl     (fhour, lat, lon)                       <- MSL

Drop-in monkey-patch for prepare_complete_training_data with the same patched
interface as the retired GRIB-based ensemble/gefs_ens_adapter.py, but reading
the local 0.5-deg regional nc directly with a per-worker fhour cache. No
regridding: vortex surgery and the 72x72 spatial crop operate natively on the
0.5-deg grid. Levels: pgrb2b alone covers all 7 ML levels (1000..200), so the
a-stream PL fields are not needed for env/surgery/chi; MSL comes from a.

Usage (inside a worker process):
    import gefs_nc_adapter
    gefs_nc_adapter.set_active_member('p07', init_time='2024-06-28 12:00',
                                      nc_dir='data/gefs_beryl')
    gefs_nc_adapter.install()
"""
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent / 'prep'))

TIME_TOLERANCE = pd.Timedelta(hours=3, minutes=30)
PL_LEVELS = [1000, 850, 700, 600, 500, 250, 200]  # must match prep's

_MEMBER_CODE = None
_INIT_TIME = None
_NC_DIR = None
_FHOURS = None
_FH_CACHE = {}          # (stream, var, fhour) -> np.ndarray, LRU-evicted
_FH_CACHE_ORDER = []
MAX_CACHED_FHOURS = 4   # per (stream, var); ~5MB each, tiny
_DS_B = None            # pgrb2b (PL, 31 levels), lazy per worker
_DS_A = None            # pgrb2a (MSL), lazy per worker
_DS_SKT = None          # skt_{m}.nc (skin temp = SST), lazy per worker
_LOCK = threading.Lock()


def set_active_member(member_code, init_time, nc_dir):
    """Point this worker at one member's local nc pair. Clears caches."""
    global _MEMBER_CODE, _INIT_TIME, _NC_DIR, _FHOURS, _DS_B, _DS_A
    _MEMBER_CODE = str(member_code)
    _INIT_TIME = pd.Timestamp(init_time) if isinstance(init_time, str) else init_time
    _NC_DIR = Path(nc_dir)
    _close_datasets()
    b = _open('b')
    if b is None:
        raise FileNotFoundError(
            f'{_NC_DIR}/pgrb2b_{_MEMBER_CODE}.nc not found')
    _FHOURS = np.asarray(b['fhour'].values, dtype=int)
    print(f'[gefs-nc] member={_MEMBER_CODE} init={_INIT_TIME} '
          f'fhours={len(_FHOURS)} grid={b.sizes["latitude"]}x{b.sizes["longitude"]}',
          flush=True)


def _open(stream):
    """Open the member's pgrb2a/pgrb2b/skt nc; None if missing."""
    global _DS_B, _DS_A, _DS_SKT
    if stream == 'b':
        if _DS_B is None:
            p = _NC_DIR / f'pgrb2b_{_MEMBER_CODE}.nc'
            if p.exists():
                _DS_B = xr.open_dataset(p)
        return _DS_B
    if stream == 'a':
        if _DS_A is None:
            p = _NC_DIR / f'pgrb2a_{_MEMBER_CODE}.nc'
            if p.exists():
                _DS_A = xr.open_dataset(p)
        return _DS_A
    if _DS_SKT is None:
        p = _NC_DIR / f'skt_{_MEMBER_CODE}.nc'
        if p.exists():
            _DS_SKT = xr.open_dataset(p)
    return _DS_SKT


def _close_datasets():
    global _DS_B, _DS_A, _DS_SKT
    for ds in (_DS_B, _DS_A, _DS_SKT):
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass
    _DS_B = _DS_A = _DS_SKT = None
    _FH_CACHE.clear()
    _FH_CACHE_ORDER.clear()


def _fh_load(ds, var, fh):
    """Cached decompress of one (var, fhour) slab; avoids repeat inflates."""
    key = (id(ds), var, int(fh))
    if key in _FH_CACHE:
        try:
            _FH_CACHE_ORDER.remove(key)
        except ValueError:
            pass
        _FH_CACHE_ORDER.append(key)
        return _FH_CACHE[key]
    arr = np.asarray(ds[var].sel(fhour=fh).load().values, dtype=np.float64)
    with _LOCK:
        same = [k for k in _FH_CACHE_ORDER if k[1] == var]
        while len(same) >= MAX_CACHED_FHOURS:
            old = same.pop(0)
            try:
                _FH_CACHE_ORDER.remove(old)
            except ValueError:
                pass
            _FH_CACHE.pop(old, None)
        _FH_CACHE[key] = arr
        _FH_CACHE_ORDER.append(key)
    return arr


def _bracket(valid_time):
    """fhours (fh0, fh1) bracketing valid_time (fh0 == fh1 near an endpoint)."""
    target = (pd.Timestamp(valid_time) - _INIT_TIME).total_seconds() / 3600.0
    i = int(np.searchsorted(_FHOURS, target))
    if i <= 0:
        return (_FHOURS[0], _FHOURS[0])
    if i >= len(_FHOURS):
        return (_FHOURS[-1], _FHOURS[-1])
    return (int(_FHOURS[i - 1]), int(_FHOURS[i]))


def _time_interp(ds, var, fh0, fh1, t, keep_da=False):
    """var (fhour, ...) linearly interpolated in time to t; clamped outside.

    Returns ndarray (fast path for PL/surgery, coords rebuilt by caller) or
    DataArray (keep_da=True, for _open_sfc merges).
    """
    if fh0 == fh1:
        arr = _fh_load(ds, var, fh0)
        return xr.DataArray(arr) if keep_da else arr
    v0 = _fh_load(ds, var, fh0)
    v1 = _fh_load(ds, var, fh1)
    w = ((pd.Timestamp(t) - (_INIT_TIME + pd.Timedelta(hours=fh0))).total_seconds()
         / 3600.0) / (fh1 - fh0)
    w = max(0.0, min(1.0, w))
    arr = (1.0 - w) * v0 + w * v1
    return xr.DataArray(arr) if keep_da else arr


# ---------------- patched functions ----------------

def get_era5_config(basin='NA', era5_root=None):
    return {'basin': (basin or 'NA').upper(), 'pl_root': '', 'sfc_root': ''}


def _find_pl(var, date_str, cfg):
    return None


def _fill_missing_levels(arr, lev):
    """Fill NaN levels (2-D lat/lon slabs) by linear interp in log-p.

    GEFS pgrb2b p25-p30 only shipped partial t/q levels (mandatory ones were
    lost); u/v/gh come from pgrb2a (6 levels) and q/t have holes. Interpolate
    vertically onto the full 31-level ladder so PI/chi profiles stay physical.
    """
    lev = np.asarray(lev, float)
    good = ~np.isnan(arr).all(axis=(1, 2)) if arr.ndim == 3 \
        else ~np.isnan(arr)
    if good.all() or good.sum() < 2:
        return arr, lev
    lp = np.log(lev)
    lg = lp[good]
    for j in np.where(~good)[0]:
        arr[j] = _interp_slab(arr, good, lp[j], lg)
    return arr, lev


def _interp_slab(arr, good, lp_j, lg):
    """Column-wise vertical interpolation of one NaN level slab."""
    # arr[good]: (n_good, lat, lon); linear in log-p per grid column
    lo = np.searchsorted(lg, lp_j) - 1
    lo = max(0, min(lo, len(lg) - 2))
    w = (lp_j - lg[lo]) / (lg[lo + 1] - lg[lo])
    slabs = arr[good]
    return (1 - w) * slabs[lo] + w * slabs[lo + 1]


def _get_pl_slices(t, pl_cache, cfg):
    """T/Q/U/V/Z with ALL levels, time-interpolated, from a+b merged.

    pgrb2a holds the 12 mandatory levels (250, 500, 850, ...), pgrb2b holds
    the complementary intermediate levels (incl. 600 hPa for chi) and q on
    all of its levels. NaN placeholder levels in one stream are filled from
    the other, so the union is the full 31-level set.
    """
    ds_b = _open('b')
    if ds_b is None:
        return {}
    ds_a = _open('a')
    fh0, fh1 = _bracket(t)
    latc = ds_b['latitude'].values
    lonc = ds_b['longitude'].values
    dims = ('isobaricInhPa', 'latitude', 'longitude')

    def merged_var(short):
        in_b = ds_b is not None and short in ds_b.data_vars
        in_a = ds_a is not None and short in ds_a.data_vars
        if not in_b and not in_a:
            return None
        if not in_a:  # b-only variable (e.g. t: pgrb2b partial levels)
            arr_b = _time_interp(ds_b, short, fh0, fh1, t)
            lev = ds_b['isobaricInhPa'].values
            arr_b, lev = _fill_missing_levels(arr_b, lev)
            return xr.DataArray(arr_b, dims=dims,
                                coords={'isobaricInhPa': lev,
                                        'latitude': latc, 'longitude': lonc})
        if not in_b:  # a-only variable (p25-p30: u/v/gh live in pgrb2a only)
            arr_a = _time_interp(ds_a, short, fh0, fh1, t)
            lev = ds_a['isobaricInhPa'].values
            arr_a, lev = _fill_missing_levels(arr_a, lev)
            return xr.DataArray(arr_a, dims=dims,
                                coords={'isobaricInhPa': lev,
                                        'latitude': latc, 'longitude': lonc})
        # both streams: b slab + NaN levels backfilled from a's mandatory set
        arr_b = _time_interp(ds_b, short, fh0, fh1, t)
        lev_b = ds_b['isobaricInhPa'].values
        arr_a = _time_interp(ds_a, short, fh0, fh1, t)
        lev_a = ds_a['isobaricInhPa'].values
        out = arr_b.copy()
        lev_out = lev_b.copy()
        idx_a = {int(v): i for i, v in enumerate(lev_a)}
        for j, v in enumerate(lev_b):
            if np.isnan(out[j]).all() and int(v) in idx_a:
                out[j] = arr_a[idx_a[int(v)]]
        out, lev_out = _fill_missing_levels(out, lev_out)
        return xr.DataArray(out, dims=dims,
                            coords={'isobaricInhPa': lev_out,
                                    'latitude': latc, 'longitude': lonc})

    slices = {}
    for short, key in [('t', 'T'), ('q', 'Q'), ('u', 'U'), ('v', 'V')]:
        slices[key] = merged_var(short)
    slices['Z'] = merged_var('gh')
    if slices['Z'] is not None:
        slices['Z'] = slices['Z'] * 9.80665
    return slices


def _open_sfc(t, sfc_cache, cfg):
    """MSL (pgrb2a prmsl) + SST (skt skin temp), time-interpolated.

    SST rides along as 'sst' (Kelvin) so the prep ERA5 branch reads it
    directly; no OISST involved.
    """
    ds_a = _open('a')
    ds_k = _open('skt')
    if ds_a is None:
        return None
    fh0, fh1 = _bracket(t)
    latc = ds_a['latitude'].values
    lonc = ds_a['longitude'].values
    dims = ('latitude', 'longitude')
    coords = {'latitude': latc, 'longitude': lonc}
    msl = xr.DataArray(_time_interp(ds_a, 'prmsl', fh0, fh1, t),
                       dims=dims, coords=coords, name='msl').to_dataset()
    if ds_k is not None:
        sst = xr.DataArray(_time_interp(ds_k, 'skt', fh0, fh1, t),
                           dims=dims, coords=coords, name='sst')
        msl = msl.assign(sst=sst)
    return {'ds': msl, 'sfc_era5': None}


def _gfs_sel_time(ds, t, tol=None):
    return ds


def close_caches(sfc_cache, pl_cache):
    _close_datasets()


def install():
    """Monkey-patch this adapter into prepare_complete_training_data."""
    import prepare_complete_training_data as P
    P.get_era5_config = get_era5_config
    P._open_sfc = _open_sfc
    P._open_pl = lambda *a, **k: {}
    P._find_pl = _find_pl
    P._get_pl_slices = _get_pl_slices
    P.close_caches = close_caches
    P._sel_time = _gfs_sel_time
    P.TIME_TOLERANCE = TIME_TOLERANCE
    print('[gefs-nc] installed: GEFS local nc (T/Q/U/V/Z/MSL) native 0.5deg grid',
          flush=True)


if __name__ == '__main__':
    # Smoke test: read one time slice for c00 and run the patched interface.
    set_active_member('c00', '2024-06-28 12:00',
                      nc_dir=Path(__file__).resolve().parent.parent / 'data' / 'gefs_beryl')
    install()
    t = pd.Timestamp('2024-06-29 00:00')
    sfc = _open_sfc(t, {}, get_era5_config('NA'))
    if sfc:
        for v in sfc['ds'].data_vars:
            print(f'{v}: shape={sfc["ds"][v].shape}')
    pl = _get_pl_slices(t, {}, get_era5_config('NA'))
    for k in ('T', 'Q', 'U', 'V', 'Z'):
        da = pl.get(k)
        print(f'{k}: shape={None if da is None else da.shape} '
              f'levels={None if da is None else len(da["isobaricInhPa"])}')
    close_caches({}, {})
