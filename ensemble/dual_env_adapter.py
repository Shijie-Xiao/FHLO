#!/usr/bin/env python3
"""Dual-source environment adapter: ERA5 replay segment + GEFS forecast segment.

FHLO Sec.2c initialization needs the ANALYSIS environment over the replay
window [fc_start - replay_hours, fc_start) (Lin et al. 2020 Sec.3e:
"environmental parameters ... from the analysis fields"), while the forecast
segment [fc_start, fc_start + duration) runs on the selected GEFS member's
forecast fields. This adapter monkey-patches prepare_complete_training_data
so that each environment query is routed by valid time:

    t <  fc_start  ->  ORIGINAL prep ERA5 functions (data/era5 analysis)
    t >= fc_start  ->  gefs_nc_adapter (member forecast nc)

Only the env-field getters are switched; everything else in prep (vortex
removal, chi, PI, ocean climatology, Cd) is untouched and operates
natively on whichever grid the routed source returns, exactly as each
source does in single-source mode.

The resulting pkl carries fc_start so the ODE (run_fast_reference.run_fast)
knows the forecast start and the plotting stage can shift the time origin.

Usage (inside an eprep worker):
    import dual_env_adapter
    dual_env_adapter.set_active_member('p07', fc_start='2025-06-30 12:00',
                                       init_time='2025-06-30 12:00',
                                       nc_dir='data/gefs_flossie')
    dual_env_adapter.install()
"""
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent / 'prep'))

import gefs_nc_adapter

_FC_START = None


def set_active_member(member_code, fc_start, init_time, nc_dir):
    """Point the worker at one GEFS member and set the forecast start time.

    fc_start = init_time in the standard chain (both are the GEFS cycle);
    they are separate args so a future chain can replay on a sub-selected
    valid time."""
    global _FC_START
    _FC_START = pd.Timestamp(fc_start)
    gefs_nc_adapter.set_active_member(member_code, init_time, nc_dir)


def _use_era5(t):
    return _FC_START is not None and pd.Timestamp(t) < _FC_START


def install():
    """Route prep's env getters by valid time: ERA5 before fc_start, GEFS
    at/after. Call AFTER importing prepare_complete_training_data (the
    module object is shared, so the GEFS patch order does not matter)."""
    import prepare_complete_training_data as P

    if getattr(P, '_dual_env_installed', False):
        return
    P._dual_env_installed = True

    era5_sfc = P._open_sfc
    era5_pl = P._get_pl_slices
    era5_sel_time = P._sel_time
    era5_find_pl = P._find_pl

    def _open_sfc(ts, cache, cfg):
        if _use_era5(ts):
            return era5_sfc(ts, cache, cfg)
        return gefs_nc_adapter._open_sfc(ts, cache, cfg)

    def _get_pl_slices(t, pl_cache, cfg):
        if _use_era5(t):
            return era5_pl(t, pl_cache, cfg)
        return gefs_nc_adapter._get_pl_slices(t, pl_cache, cfg)

    def _sel_time(ds, t, tol=None):
        # GEFS slabs arrive already time-interpolated (no time coord) ->
        # pass through, exactly like gefs_nc_adapter._gfs_sel_time; ERA5
        # datasets keep the real-time selection.
        if _use_era5(t):
            return era5_sel_time(ds, t)
        return ds

    def _find_pl(var, date_str, cfg):
        if _use_era5(pd.Timestamp(f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}')):
            return era5_find_pl(var, date_str, cfg)
        return gefs_nc_adapter._find_pl(var, date_str, cfg)

    P._open_sfc = _open_sfc
    P._get_pl_slices = _get_pl_slices
    P._sel_time = _sel_time
    P._find_pl = _find_pl
    P.close_caches = gefs_nc_adapter.close_caches
    P.TIME_TOLERANCE = gefs_nc_adapter.TIME_TOLERANCE
    print(f'[dual-env] installed: ERA5 (< {_FC_START}) | GEFS (>= {_FC_START})',
          flush=True)
