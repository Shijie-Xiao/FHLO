#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEFS-ensemble -> ERA5-equivalent adapter (strict GEFS, with 0.5->0.25 regrid).

Drop-in monkey-patch for `prepare_complete_training_data` that serves ALL
required fields (T/Q/U/V/Z/MSL/SST/SP) out of GEFS per-storm GRIB2 archives
instead of ERA5 NetCDF. No ERA5 fallback.

Strategy:
  1. Index the per-case GEFS layout:
       {GFS_ROOT}/{CASE}/grib2/pgrb2a/ge{member}.t{HH}z.pgrb2a.0p50.f{fhh}.grib2
       {GFS_ROOT}/{CASE}/grib2/pgrb2b/ge{member}.t{HH}z.pgrb2b.0p50.f{fhh}.grib2
  2. For a given (member_id, valid_time), find the bracketing forecast hours
     of that member and linearly interpolate to valid_time.
  3. Cache each variable from each GRIB as a small NetCDF file on first read.
     We open each variable SEPARATELY with cfgrib (filter_by_keys shortName=...),
     because cfgrib fails when multiple variables in the same file have different
     level sets (HGT has 11 levels, U has 12 because U includes 400mb, etc).
  4. Bilinear-interpolate every GEFS field from 0.5deg to ERA5 0.25deg grid.
     Z (geopotential height in gpm) is multiplied by 9.80665 to m^2/s^2 to match
     ERA5 Z convention.
  5. Original `prepare_complete_training_data` is untouched; only names rebound.

Variables:
  PL (isobaricInhPa): t, q, u, v, gh (-> Z as gh*9.80665)
      pgrb2a: canonical set 1000/925/850/700/500/250/200/100/50/10
              U/V also have 300/400 mb
      pgrb2b: rest, including 600 mb, 975/950/900/800/750/650/550/450/...
  meanSea: prmsl                         -- MSL pressure
  surface: t                             -- skin temp -> SST
  surface: sp                            -- surface pressure
  BLH                                    -- not in GEFS, NaN; downstream falls back to 1400
"""
from __future__ import annotations
import os, threading
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xarray as xr

def _cfg_overrides():
    """Read path overrides from FHLO/config.txt (lowercase keys)."""
    import os
    cfg = {}
    p = Path(__file__).resolve().parent.parent / 'config.txt'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k, v = s.split('=', 1)
            cfg[k.strip().lower()] = v.strip()
    return cfg


_CFG = _cfg_overrides()
# GEFS GRIB2 archive, ERA5 archive, and the cfgrib->NetCDF cache dir.
# Precedence: config.txt (gefs_root / era5_root / gefs_cache_dir) > env > default.
GFS_ROOT = Path(_CFG.get('gefs_root') or os.environ.get('FHLO_GEFS_ROOT')
                or '/global/cfs/cdirs/m5011/Jay/ERA5/GFS')
ERA5_ROOT = Path(_CFG.get('era5_root') or os.environ.get('FHLO_ERA5_ROOT')
                 or '/global/cfs/cdirs/m5011/Jay/ERA5')
_NC_CACHE = Path(_CFG.get('gefs_cache_dir') or os.environ.get('FHLO_GEFS_CACHE')
                 or Path(__file__).resolve().parent.parent / 'data' / 'gefs_nc_cache')
_NC_CACHE.mkdir(parents=True, exist_ok=True)

# `PL_LEVELS` is for spatial_3d (7 levels consumed by ML/FAST models).
PL_LEVELS = [1000, 850, 700, 600, 500, 250, 200]

# Full set of isobaric levels we want for tcpyPI / CAPE calc. ERA5 has 30 levels
# (25-hPa steps); GEFS pgrb2a + pgrb2b together provide 31. T/Q/U/V/Z are
# returned with ALL these levels so tcpyPI has the dense vertical sampling it
# needs. (spatial_3d in create_spatial_fields still picks only PL_LEVELS via
# .sel(method='nearest'), so downstream ML models are unaffected.)
FULL_PL_LEVELS = [
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
    450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1,
]
TIME_TOLERANCE = pd.Timedelta(hours=3, minutes=30)

# ERA5 0.25deg grid (matches data/.../PL convention: 90 -> -90 step -0.25, 0 -> 359.75 step 0.25)
ERA5_LAT = np.arange(90, -90 - 1e-6, -0.25)
ERA5_LON = np.arange(0, 360, 0.25)

# --- per-process config (set by set_active_member before processing) ---
_CASE_DIR: Path | None = None
_MEMBER_CODE: str = 'c00'
_INIT_TIME: pd.Timestamp | None = None
_FHOURS: list[int] = list(range(0, 241, 3))


def set_active_member(case_dir: str | Path, member_code: str,
                       init_time: str | pd.Timestamp,
                       fhours: list[int] | None = None):
    """Configure the adapter to serve data for a specific ensemble member."""
    global _CASE_DIR, _MEMBER_CODE, _INIT_TIME, _FHOURS
    _CASE_DIR = Path(case_dir)
    _MEMBER_CODE = member_code
    _INIT_TIME = pd.Timestamp(init_time) if isinstance(init_time, str) else init_time
    _FHOURS = fhours if fhours is not None else list(range(0, 241, 3))
    _VAR_DS_CACHE.clear()
    _REGRIDDED_CACHE.clear()
    print(f'[gefs-adapter] case={_CASE_DIR.name} member={_MEMBER_CODE} '
          f'init={_INIT_TIME} fhours={len(_FHOURS)}', flush=True)


# ---------------- GEFS file index ----------------
class _GefsIndex:
    """For current (_CASE_DIR, _MEMBER_CODE), resolve (valid_time -> fhours)."""

    def bracket(self, valid_time: pd.Timestamp):
        """Return list of (fhour, (grib_a_path, grib_b_path)) bracketing valid_time."""
        if _CASE_DIR is None or _INIT_TIME is None:
            return []
        target_fh = (valid_time - _INIT_TIME).total_seconds() / 3600.0
        import bisect
        fhs = _FHOURS
        i = bisect.bisect_right(fhs, target_fh)
        if i == 0:
            return [(fhs[0], self._gpaths(fhs[0]))]
        if i >= len(fhs):
            return [(fhs[-1], self._gpaths(fhs[-1]))]
        return [(fhs[i - 1], self._gpaths(fhs[i - 1])),
                (fhs[i], self._gpaths(fhs[i]))]

    def _gpaths(self, fhour: int):
        fhh = f'{fhour:03d}'
        init_h = f'{_INIT_TIME.hour:02d}'
        a = _CASE_DIR / 'grib2' / 'pgrb2a' / f'ge{_MEMBER_CODE}.t{init_h}z.pgrb2a.0p50.f{fhh}.grib2'
        b = _CASE_DIR / 'grib2' / 'pgrb2b' / f'ge{_MEMBER_CODE}.t{init_h}z.pgrb2b.0p50.f{fhh}.grib2'
        return (a, b)


_GEFS = _GefsIndex()


# ---------------- Regrid-per-fhour cache (in-memory, sliding window) ----------------
# After the first time we open pgrb2a + pgrb2b for a given fhour and regrid
# everything to ERA5 0.25deg, cache the result so subsequent valid_times that
# share the same fhour are ~free. Keyed by (src, fhour) -> dict of DataArrays.
#
# Memory note: each (src, fhour) holds ~5 vars * 31 levs * 721*1440 * 8B = 1.3GB.
# With the sliding window below we keep at most MAX_CACHED_FHOURS=4 entries per
# src = 8 entries total = ~10GB peak. That fits Perlmutter CPU nodes (256GB)
# while giving time-interpolation full cache hits for typical brackets.
_REGRIDDED_CACHE: Dict[Tuple[str, int], Dict[str, xr.DataArray]] = {}
_REGRIDDED_ORDER: list[Tuple[str, int]] = []  # LRU order
# OOM fix (beryl 57170170: 16 workers * 8 fhours * 2 srcs * ~1.3GB = 320GB >
# 256GB node). 2/stream = ~5GB per worker; with 8 workers = ~40GB steady.
MAX_CACHED_FHOURS = 2  # per src; total across both srcs = 2x


def _load_fhour_regridded(src: str, fhour: int) -> Dict[str, xr.DataArray]:
    """Open all standard PL variables at this fhour, regrid to ERA5 grid.

    Cached by (src, fhour) with LRU eviction (MAX_CACHED_FHOURS per src).
    """
    key = (src, fhour)
    with _LOCK:
        if key in _REGRIDDED_CACHE:
            # bump LRU
            try: _REGRIDDED_ORDER.remove(key)
            except ValueError: pass
            _REGRIDDED_ORDER.append(key)
            return _REGRIDDED_CACHE[key]
    if _CASE_DIR is None or _INIT_TIME is None:
        return {}
    paths = _GEFS._gpaths(fhour)
    grib_path = paths[0] if src == 'a' else paths[1]
    out: Dict[str, xr.DataArray] = {}
    for short in ('t', 'q', 'u', 'v', 'gh'):
        ds = _grib_to_var_nc(grib_path, short, 'isobaricInhPa')
        if ds is None or short not in ds:
            continue
        da = ds[short]
        # A single-level field (b-stream U/V/gh @600mb) arrives with the level
        # as a SCALAR coord (not a dim); expand it back to a length-1 dim so
        # downstream level-merging can see it. squeeze() only on other scalar dims.
        lev_names = [c for c in ('isobaricInhPa', 'level') if c in da.coords]
        if lev_names:
            ln = lev_names[0]
            if ln not in da.dims:
                da = da.expand_dims(ln)
        drop_dims = [d for d in da.dims
                     if d not in ('isobaricInhPa', 'level', 'latitude', 'longitude')
                     and da.sizes[d] == 1]
        if drop_dims:
            da = da.squeeze(dim=drop_dims)
        if 'isobaricInhPa' not in da.coords and 'level' in da.coords:
            da = da.rename({'level': 'isobaricInhPa'})
        # Load into memory NOW so later time-interp arithmetic is fast (no
        # lazy dask graph), and so the underlying GRIB NC handle can be
        # safely reused. Without this, xr.concat below goes through dask.
        da = da.load()
        da = _regrid_to_era5(da)
        out[short] = da
    with _LOCK:
        # LRU eviction by src
        same_src_keys = [k for k in _REGRIDDED_ORDER if k[0] == src]
        while len(same_src_keys) >= MAX_CACHED_FHOURS:
            old = same_src_keys.pop(0)
            try: _REGRIDDED_ORDER.remove(old)
            except ValueError: pass
            _REGRIDDED_CACHE.pop(old, None)
        _REGRIDDED_CACHE[key] = out
        _REGRIDDED_ORDER.append(key)
    return out


# ---------------- GRIB -> NetCDF materialization ----------------
_VAR_DS_CACHE: Dict[Tuple[Path, str, str], xr.Dataset] = {}
_LOCK = threading.Lock()


def _var_nc_path(grib_path: Path, short: str, lvl_type: str) -> Path:
    cache_key = f'{_CASE_DIR.name}_{_MEMBER_CODE}_{grib_path.stem}_{short}_{lvl_type}'
    return _NC_CACHE / f'{cache_key}.nc'


def _grib_to_var_nc(grib_path: Path, short: str, lvl_type: str):
    """Open one variable from GRIB with a tight cfgrib filter and cache as NC.

    Concurrent-safe: the 16 prep workers (and multiple storm jobs) may target
    the same (case, member, fhour, var) cache file simultaneously. Writers go
    through a unique tmp file + os.replace (atomic), readers retry once on a
    torn file, and a corrupt final file is deleted so the next call rebuilds.
    """
    cache_key = (grib_path, short, lvl_type)
    with _LOCK:
        if cache_key in _VAR_DS_CACHE:
            return _VAR_DS_CACHE[cache_key]

    (_NC_CACHE / 'idx').mkdir(parents=True, exist_ok=True)
    nc_path = _var_nc_path(grib_path, short, lvl_type)
    for attempt in range(2):
        if not nc_path.exists():
            try:
                ds = xr.open_dataset(str(grib_path), engine='cfgrib',
                                     backend_kwargs={
                                         'filter_by_keys': {'typeOfLevel': lvl_type,
                                                            'shortName': short},
                                         # indexpath='' disables the on-disk idx:
                                         # cfgrib writes it non-atomically, and
                                         # concurrent workers sharing the path
                                         # read half-written pickles (EOFError).
                                         # The grib is read once per process and
                                         # immediately materialized to the NC
                                         # cache below, so a persistent index
                                         # buys nothing.
                                         'indexpath': '',
                                     })
                ds.load()
                drop = [c for c in ('step', 'valid_time') if c in ds.coords]
                if drop:
                    ds = ds.drop_vars(drop)
                # atomic publish: write unique tmp then rename over target
                tmp = nc_path.with_suffix(f'.tmp{os.getpid()}')
                ds.to_netcdf(tmp)
                os.replace(tmp, nc_path)
                ds.close()
            except Exception:
                return None
        try:
            ds = xr.open_dataset(nc_path)
            if not list(ds.data_vars):        # torn/empty cache from a race
                ds.close()
                nc_path.unlink(missing_ok=True)
                if attempt == 0:
                    continue                  # rebuild once
                return None
            break
        except Exception:
            nc_path.unlink(missing_ok=True)
            if attempt == 0:
                continue
            return None
    with _LOCK:
        _VAR_DS_CACHE[cache_key] = ds
    return ds


def _interp_var(bracket, valid_time: pd.Timestamp, short: str, lvl_type: str,
                src: str = 'a'):
    """Linearly interpolate one variable between bracketing fhours."""
    if not bracket:
        return None
    try:
        if len(bracket) == 1:
            fh, (a_path, b_path) = bracket[0]
            path = a_path if src == 'a' else b_path
            ds = _grib_to_var_nc(path, short, lvl_type)
            if ds is None or short not in ds:
                return None
            return ds[short].squeeze().copy()
        (fh0, (a0, b0)), (fh1, (a1, b1)) = bracket
        v0 = _INIT_TIME + pd.Timedelta(hours=fh0)
        v1 = _INIT_TIME + pd.Timedelta(hours=fh1)
        w = 0.0 if v1 == v0 else max(0.0, min(1.0, float((valid_time - v0) / (v1 - v0))))
        p0 = a0 if src == 'a' else b0
        p1 = a1 if src == 'a' else b1
        ds0 = _grib_to_var_nc(p0, short, lvl_type)
        ds1 = _grib_to_var_nc(p1, short, lvl_type)
        if ds0 is None or short not in ds0:
            return None
        if ds1 is None or short not in ds1:
            return ds0[short].squeeze().copy()
        return (1 - w) * ds0[short].squeeze() + w * ds1[short].squeeze()
    except Exception as e:
        print(f'[gefs-{short}] {valid_time} failed: {e}', flush=True)
        return None


# ---------------- 0.5deg -> 0.25deg regrid ----------------
# Pre-computed index pairs for bilinear interp from 0.5deg to 0.25deg.
# ERA5 grid: lat 90 -> -90 step -0.25 (721 pts); lon 0 -> 359.75 step 0.25 (1440 pts)
# GEFS grid: lat 90 -> -90 step -0.5 (361 pts); lon 0 -> 359.5 step 0.5  (720 pts)
# For each ERA5 cell, the four GEFS neighbors are fixed, so we precompute them.
_REGRID_I0 = None
_REGRID_I1 = None
_REGRID_J0 = None
_REGRID_J1 = None
_REGRID_W00 = None
_REGRID_W10 = None
_REGRID_W01 = None
_REGRID_W11 = None


def _init_regrid_indices():
    global _REGRID_I0, _REGRID_I1, _REGRID_J0, _REGRID_J1
    global _REGRID_W00, _REGRID_W10, _REGRID_W01, _REGRID_W11
    if _REGRID_I0 is not None:
        return
    # ERA5 lat: 90, 89.75, ..., -90  (descending); ERA5_LON: 0, 0.25, ..., 359.75
    # GEFS lat: 90, 89.5, ..., -90   (descending); GEFS lon: 0, 0.5, ..., 359.5
    n_lat_e = ERA5_LAT.size  # 721
    n_lon_e = ERA5_LON.size  # 1440
    # For each ERA5 lat, find bracketing GEFS lat indices.
    # GEFS lat[i] = 90 - 0.5*i  ->  i = (90 - lat) / 0.5
    lat_e = ERA5_LAT  # descending
    i_f = (90.0 - lat_e) / 0.5  # float index into GEFS lat array
    i0 = np.floor(i_f).astype(np.int32)
    w_lat = i_f - i0  # 0..1
    i0 = np.clip(i0, 0, 360)
    i1 = np.clip(i0 + 1, 0, 360)
    # broadcast to (n_lat_e, n_lon_e)
    I0 = np.broadcast_to(i0[:, None], (n_lat_e, n_lon_e))
    I1 = np.broadcast_to(i1[:, None], (n_lat_e, n_lon_e))
    W_LAT = np.broadcast_to(w_lat[:, None], (n_lat_e, n_lon_e))  # weight on i1

    # For each ERA5 lon, find bracketing GEFS lon indices.
    # GEFS lon[j] = 0.5*j  ->  j = lon / 0.5
    lon_e = ERA5_LON
    j_f = lon_e / 0.5
    j0 = np.floor(j_f).astype(np.int32)
    w_lon = j_f - j0
    j0 = np.clip(j0, 0, 719)
    j1 = np.clip(j0 + 1, 0, 719)
    J0 = np.broadcast_to(j0[None, :], (n_lat_e, n_lon_e))
    J1 = np.broadcast_to(j1[None, :], (n_lat_e, n_lon_e))
    W_LON = np.broadcast_to(w_lon[None, :], (n_lat_e, n_lon_e))  # weight on j1

    _REGRID_I0 = I0
    _REGRID_I1 = I1
    _REGRID_J0 = J0
    _REGRID_J1 = J1
    # Bilinear weights
    _REGRID_W00 = ((1 - W_LAT) * (1 - W_LON)).astype(np.float32)
    _REGRID_W10 = (W_LAT * (1 - W_LON)).astype(np.float32)
    _REGRID_W01 = ((1 - W_LAT) * W_LON).astype(np.float32)
    _REGRID_W11 = (W_LAT * W_LON).astype(np.float32)


def _regrid_to_era5(da: xr.DataArray) -> xr.DataArray:
    """Bilinear-interpolate a 0.5deg GEFS field to ERA5 0.25deg grid (fast numpy).

    Two-stage 1D linear interp: lat first (output: ..., 721, 720), then lon
    (output: ..., 721, 1440). This is ~5x faster than fancy indexing all four
    neighbors at once on large arrays.
    """
    _init_regrid_indices()
    if 'latitude' not in da.coords or 'longitude' not in da.coords:
        return da
    other_dims = [d for d in da.dims if d not in ('latitude', 'longitude')]
    da_t = da.transpose(*other_dims, 'latitude', 'longitude') if other_dims else da.transpose('latitude', 'longitude')
    arr = np.asarray(da_t.values, dtype=np.float32)

    # ---- Stage 1: interp along latitude (361 -> 721) ----
    # ERA5 lat[k] = 90 - 0.25*k. GEFS lat[i] = 90 - 0.5*i.
    # i_f = (90 - lat_e) / 0.5; for lat_e in 90, 89.75, ..., -90
    lat_e = ERA5_LAT.astype(np.float32)
    i_f = (90.0 - lat_e) / 0.5
    i0 = np.clip(np.floor(i_f).astype(np.int32), 0, 360)
    i1 = np.clip(i0 + 1, 0, 360)
    w_lat = (i_f - i0).astype(np.float32)  # (721,) weight on i1
    # gather: arr[..., i0[k], :] and arr[..., i1[k], :]
    # np.take along axis=-2
    a0 = np.take(arr, i0, axis=-2)  # (..., 721, 720)
    a1 = np.take(arr, i1, axis=-2)
    w_lat_b = w_lat.reshape([1] * (a0.ndim - 2) + [721, 1])
    arr_lat = a0 + w_lat_b * (a1 - a0)  # (..., 721, 720)

    # ---- Stage 2: interp along longitude (720 -> 1440) ----
    lon_e = ERA5_LON.astype(np.float32)
    j_f = lon_e / 0.5
    j0 = np.clip(np.floor(j_f).astype(np.int32), 0, 719)
    j1 = np.clip(j0 + 1, 0, 719)
    w_lon = (j_f - j0).astype(np.float32)
    b0 = np.take(arr_lat, j0, axis=-1)  # (..., 721, 1440)
    b1 = np.take(arr_lat, j1, axis=-1)
    w_lon_b = w_lon.reshape([1] * (b0.ndim - 2) + [1, 1440])
    out = b0 + w_lon_b * (b1 - b0)

    coords = {k: da_t[k] for k in other_dims}
    coords['latitude'] = ERA5_LAT
    coords['longitude'] = ERA5_LON
    dims = tuple(other_dims) + ('latitude', 'longitude')
    result = xr.DataArray(out.astype(np.float32), dims=dims, coords=coords)
    result.attrs.update(da.attrs)
    return result


def _merge_pl_var(bracket, t, short, target_levels=None):
    """Merge pgrb2a + pgrb2b for `short` and (optionally) select target levels.

    If `target_levels` is None, return ALL merged levels (no subsetting) - this
    is what we want for T/Q going into tcpyPI. If a list is passed (e.g.
    PL_LEVELS), keep only those levels.
    """
    da_a = _interp_var(bracket, t, short, 'isobaricInhPa', src='a')
    da_b = _interp_var(bracket, t, short, 'isobaricInhPa', src='b')
    if da_a is None and da_b is None:
        return None
    if da_a is None:
        merged = da_b
    elif da_b is None:
        merged = da_a
    else:
        coord = 'isobaricInhPa' if 'isobaricInhPa' in da_a.coords else 'level'
        a_levels = set(da_a[coord].values.tolist())
        b_levels = set(da_b[coord].values.tolist())
        b_only_levels = sorted(b_levels - a_levels, reverse=True)
        if b_only_levels:
            da_b_extra = da_b.sel({coord: b_only_levels})
            merged = xr.concat([da_a, da_b_extra], dim=coord)
            merged = merged.sortby(coord, ascending=False)
        else:
            merged = da_a
    coord = 'isobaricInhPa' if 'isobaricInhPa' in merged.coords else 'level'
    have = set(merged[coord].values.tolist())
    if target_levels is not None:
        want = [lv for lv in target_levels if lv in have]
        merged = merged.sel({coord: want})
    merged = _regrid_to_era5(merged)
    return merged


# ---------------- patched functions ----------------
def get_era5_config(basin='NA', era5_root=None):
    # No ERA5 used at all; return a stub.
    return {'basin': (basin or 'NA').upper(),
            'pl_root': str(ERA5_ROOT / '2025' / 'PL'),
            'sfc_root': str(ERA5_ROOT / '2025' / 'SFC')}


def _find_pl(var, date_str, cfg):
    return None


def _get_pl_slices(t, pl_cache, cfg):
    """All PL fields from GEFS, regridded to ERA5 0.25deg.

    Returns T/Q/U/V/Z with ALL available levels (pgrb2a + pgrb2b merged, ~31
    levels) so tcpyPI has dense vertical sampling for CAPE / To. spatial_3d
    (the 7-level ML input) is built downstream in create_spatial_fields via
    .sel(method='nearest') so ML models are unaffected.

    Optimization: load + regrid each fhour ONCE and cache; subsequent
    valid_times bracketing the same fhour reuse the cached arrays. This
    reduces per-step cost from ~7s to ~0.1s.
    """
    bracket = _GEFS.bracket(t)
    if not bracket:
        return {}

    # Helper: pick the right DataArrays from each fhour's cache and interpolate
    # in time. NOTE: q lives only in pgrb2b, so 2-file level mismatch only
    # matters across fh0/fh1 of the same src. We intersect levels to be safe.
    def _gather_interp(short):
        if len(bracket) == 1:
            fh, _ = bracket[0]
            # try src='a' first, fallback to 'b'
            da_dict = _load_fhour_regridded('a', fh)
            if short not in da_dict:
                da_dict = _load_fhour_regridded('b', fh)
            return da_dict.get(short)
        (fh0, _), (fh1, _) = bracket
        v0 = _INIT_TIME + pd.Timedelta(hours=fh0)
        v1 = _INIT_TIME + pd.Timedelta(hours=fh1)
        w = 0.0 if v1 == v0 else max(0.0, min(1.0, float((t - v0) / (v1 - v0))))
        d0_a = _load_fhour_regridded('a', fh0)
        d1_a = _load_fhour_regridded('a', fh1)
        d0_b = _load_fhour_regridded('b', fh0)
        d1_b = _load_fhour_regridded('b', fh1)
        # short may live in 'a' or 'b'
        da0 = d0_a.get(short) if short in d0_a else d0_b.get(short)
        da1 = d1_a.get(short) if short in d1_a else d1_b.get(short)
        if da0 is None and da1 is None:
            return None
        if da0 is None:
            return da1
        if da1 is None:
            return da0
        # Both present: align to level intersection in case fh0/fh1 have
        # different vertical coverage (mixed full/lean download boundary).
        coord = 'isobaricInhPa' if 'isobaricInhPa' in da0.coords else 'level'
        if coord in da0.coords and coord in da1.coords:
            l0 = set(int(v) for v in np.asarray(da0[coord].values))
            l1 = set(int(v) for v in np.asarray(da1[coord].values))
            if l0 != l1:
                common = sorted(l0 & l1, reverse=True)
                if common:
                    common_arr = np.asarray(common, dtype=np.float32)
                    da0 = da0.sel({coord: common_arr})
                    da1 = da1.sel({coord: common_arr})
        return (1 - w) * da0 + w * da1

    slices = {}
    # T, U, V: merge 'a' (mandatory levels) + 'b' (intermediate)
    for short, key in [('t', 'T'), ('u', 'U'), ('v', 'V')]:
        merged = _merge_pl_levels(bracket, t, short)
        if merged is not None:
            slices[key] = merged
    # Q lives only in pgrb2b - keep all its levels
    q = _gather_interp('q')
    if q is not None:
        slices['Q'] = q
    # Z from gh (gpm) -> m^2/s^2
    gh = _merge_pl_levels(bracket, t, 'gh')
    if gh is not None:
        slices['Z'] = gh * 9.80665
    return slices


def _merge_pl_levels(bracket, t, short):
    """Merge pgrb2a + pgrb2b for `short`, KEEPING ALL LEVELS, time-interpolated.

    Returns a single DataArray with all (a + b) levels. Since 'a' has 10
    canonical mandatory levels and 'b' has the complementary 21 intermediate
    levels, the union is 31 levels (no overlap for most short names).
    We build the merged array directly with numpy to keep this fast (~50ms
    per var instead of ~700ms with xr.concat/reindex).

    COMPATIBILITY: handles mixed full/lean downloads. If fh0 and fh1 have
    different level sets (e.g. fh0 from old full config, fh1 from new lean),
    the time interpolation uses the INTERSECTION of levels so shapes match.
    """
    def _get(src, fh):
        d = _load_fhour_regridded(src, fh)
        return d.get(short)

    def _align_to_levels(da, want_levs, coord_name):
        """Resample a DataArray onto want_levs (_nan if missing), keeping order."""
        cur_levs = np.asarray(da[coord_name].values)
        idx_map = {int(v): i for i, v in enumerate(cur_levs)}
        n_lat = da.values.shape[-2]; n_lon = da.values.shape[-1]
        out = np.full((len(want_levs), n_lat, n_lon), np.nan, dtype=np.float32)
        for k, v in enumerate(want_levs):
            iv = int(v)
            if iv in idx_map:
                out[k] = da.values[idx_map[iv]]
        coords = {coord_name: want_levs.astype(np.float32),
                  'latitude': ERA5_LAT, 'longitude': ERA5_LON}
        return xr.DataArray(out, dims=(coord_name, 'latitude', 'longitude'),
                            coords=coords)

    def _lev(da):
        if da is None:
            return None
        if 'isobaricInhPa' in da.coords:
            return 'isobaricInhPa'
        if 'level' in da.coords:
            return 'level'
        return None                 # squeezed single-level (2D): not mergeable

    if len(bracket) == 1:
        fh, _ = bracket[0]
        da_a = _get('a', fh)
        da_b = _get('b', fh)
        if _lev(da_a) is None:
            da_a = None
        if _lev(da_b) is None:
            da_b = None
    else:
        (fh0, _), (fh1, _) = bracket
        v0 = _INIT_TIME + pd.Timedelta(hours=fh0)
        v1 = _INIT_TIME + pd.Timedelta(hours=fh1)
        w = 0.0 if v1 == v0 else max(0.0, min(1.0, float((t - v0) / (v1 - v0))))
        a0 = _get('a', fh0); a1 = _get('a', fh1)
        b0 = _get('b', fh0); b1 = _get('b', fh1)
        # a and b may label the pressure dim differently (isobaricInhPa vs
        # level), and a variable may exist in one stream as a single squeezed
        # level (2D, no level coord at all). Resolve each side's own coord
        # name; None means that side cannot take part in level merging.
        def _coord_of(da):
            if da is None:
                return None
            if 'isobaricInhPa' in da.coords:
                return 'isobaricInhPa'
            if 'level' in da.coords:
                return 'level'
            return None            # 2D single-level field: unusable for merge
        ca = _coord_of(a0) or _coord_of(a1)
        cb = _coord_of(b0) or _coord_of(b1)
        if ca is None:
            a0 = a1 = None
        if cb is None:
            b0 = b1 = None
        # For each src, gather union of levels present at fh0 AND fh1; then align
        # both to the INTERSECTION so time-interp shapes match.
        da_a = None
        if a0 is not None or a1 is not None:
            levs0 = set(int(v) for v in np.asarray(a0[ca].values)) if a0 is not None else set()
            levs1 = set(int(v) for v in np.asarray(a1[ca].values)) if a1 is not None else set()
            # intersect (or just take whichever side exists when the other is None)
            if a0 is not None and a1 is not None:
                common = sorted(levs0 & levs1, reverse=True)
            elif a0 is not None:
                common = sorted(levs0, reverse=True)
            else:
                common = sorted(levs1, reverse=True)
            common_arr = np.asarray(common, dtype=np.float32)
            if a0 is not None and a1 is not None:
                a0a = _align_to_levels(a0, common_arr, ca)
                a1a = _align_to_levels(a1, common_arr, ca)
                arr = (1 - w) * a0a.values + w * a1a.values
                da_a = a0a.copy(data=arr)
            elif a0 is not None:
                da_a = a0
            else:
                da_a = a1
        da_b = None
        if b0 is not None or b1 is not None:
            levs0 = set(int(v) for v in np.asarray(b0[cb].values)) if b0 is not None else set()
            levs1 = set(int(v) for v in np.asarray(b1[cb].values)) if b1 is not None else set()
            if b0 is not None and b1 is not None:
                common = sorted(levs0 & levs1, reverse=True)
            elif b0 is not None:
                common = sorted(levs0, reverse=True)
            else:
                common = sorted(levs1, reverse=True)
            common_arr = np.asarray(common, dtype=np.float32)
            if b0 is not None and b1 is not None:
                b0a = _align_to_levels(b0, common_arr, cb)
                b1a = _align_to_levels(b1, common_arr, cb)
                arr = (1 - w) * b0a.values + w * b1a.values
                da_b = b0a.copy(data=arr)
            elif b0 is not None:
                da_b = b0
            else:
                da_b = b1

    if da_a is None and da_b is None:
        return None
    if da_a is None:
        return da_b
    if da_b is None:
        return da_a

    coord = 'isobaricInhPa' if 'isobaricInhPa' in da_a.coords else 'level'
    if 'isobaricInhPa' in da_a.coords and 'isobaricInhPa' not in da_b.coords:
        da_b = da_b.rename({'level': 'isobaricInhPa'})
    elif 'level' in da_a.coords and 'level' not in da_b.coords:
        da_b = da_b.rename({'isobaricInhPa': 'level'})
    a_levs = np.asarray(da_a[coord].values)
    b_levs = np.asarray(da_b[coord].values)
    merged_levs = np.unique(np.concatenate([a_levs, b_levs]))[::-1]  # descending
    n_lev = len(merged_levs)
    # Build merged numpy array directly
    a_vals = da_a.values; b_vals = da_b.values
    # Index arrays: where does each merged level live in a / b?
    a_idx = {int(v): i for i, v in enumerate(a_levs)}
    b_idx = {int(v): i for i, v in enumerate(b_levs)}
    # Lat/lon shape is the same for a and b (both regridded to ERA5 grid)
    n_lat = a_vals.shape[-2]; n_lon = a_vals.shape[-1]
    out = np.full((n_lev, n_lat, n_lon), np.nan, dtype=np.float32)
    for k, v in enumerate(merged_levs):
        iv = int(v)
        if iv in a_idx:
            out[k] = a_vals[a_idx[iv]]
        elif iv in b_idx:
            out[k] = b_vals[b_idx[iv]]
    coords = {coord: merged_levs.astype(np.float32),
              'latitude': ERA5_LAT, 'longitude': ERA5_LON}
    return xr.DataArray(out, dims=(coord, 'latitude', 'longitude'), coords=coords)


def _open_sfc(t, sfc_cache, cfg):
    """GEFS MSL (PRMSL) and SST (skin temp from surface)."""
    bracket = _GEFS.bracket(t)

    msl_da = _interp_var(bracket, t, 'prmsl', 'meanSea', src='a')
    sst_da = _interp_var(bracket, t, 't', 'surface', src='b')

    if msl_da is None or sst_da is None:
        return None

    msl_da = _regrid_to_era5(msl_da).rename('msl')
    sst_da = _regrid_to_era5(sst_da).rename('sst')

    merged = xr.merge([msl_da, sst_da], compat='override')
    return {'ds': merged, 'sfc_era5': None}


def _gfs_sel_time(ds, t, tol=None):
    """Datasets returned are already single-time snapshots."""
    if ds is None:
        return None
    return ds


def close_caches(sfc_cache, pl_cache):
    for e in sfc_cache.values():
        if e and isinstance(e, dict) and e.get('ds'):
            try: e['ds'].close()
            except Exception: pass
    for ds in _VAR_DS_CACHE.values():
        try: ds.close()
        except Exception: pass
    _VAR_DS_CACHE.clear()
    _REGRIDDED_CACHE.clear()


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
    print('[gefs-adapter] installed: STRICT GEFS (T/Q/U/V/Z/MSL/SST) regridded to 0.25deg')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/pscratch/sd/s/sixao74/Deepmind/PINN')
    set_active_member(
        case_dir='/global/cfs/cdirs/m5011/Jay/ERA5/GFS/2025_ERIN_NA',
        member_code='c00',
        init_time='2025-08-11 12:00',
    )
    install()
    t = pd.Timestamp('2025-08-11 15:00')
    print(f'\n=== {t} ===')
    sfc = _open_sfc(t, {}, get_era5_config('NA'))
    if sfc:
        ds = sfc['ds']
        print(f'  SST shape: {ds["sst"].shape}, lat: {ds["sst"].latitude.values[:3]}...{ds["sst"].latitude.values[-3:]}')
        print(f'  MSL shape: {ds["msl"].shape}')
    pl = _get_pl_slices(t, {}, get_era5_config('NA'))
    for k in ['T', 'Q', 'U', 'V', 'Z']:
        da = pl.get(k)
        if da is not None:
            coord = 'isobaricInhPa' if 'isobaricInhPa' in da.coords else 'level'
            print(f'  {k}: shape={da.shape} levels={da[coord].values.tolist()}')
        else:
            print(f'  {k}: MISSING')
