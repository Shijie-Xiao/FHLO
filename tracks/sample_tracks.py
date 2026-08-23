"""Sample synthetic ensemble tracks from per-step Markov params.

Case layout: tracks/processed/{storm}/{YYYYMMDDHH}/
    markov_params_6h.pkl -> synthetic_tracks_{n}members.nc
Sampling: per-step conditional Gaussians (FHLO paper sec.3a), init positions
and velocities bootstrapped from parent members, length capped by the 75%
survival horizon.
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Earth_Radius, PROCESSED_TRACKS_DIR, N_TRACKS, DURATION_DAYS

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


def _load_init_cloud(case_dir: Path):
    """Initial lon/lat and velocities of every parent member from raw.pkl."""
    raw_file = case_dir / "raw.pkl"
    if not raw_file.exists():
        return None
    with open(raw_file, "rb") as f:
        raw = pickle.load(f)
    lon_list, lat_list, u_list, v_list, parent_list = [], [], [], [], []
    for tr in raw.get("tracks", []):
        lon, lat, ta = tr.get("lon"), tr.get("lat"), tr.get("datetime")
        if lon is None or lat is None or len(lon) < 2 or not ta or len(ta) < 2:
            continue
        dt_sec = (ta[1] - ta[0]).total_seconds()
        if dt_sec <= 0:
            continue
        lon_list.append(float(lon[0]))
        lat_list.append(float(lat[0]))
        dx = (np.deg2rad(float(lon[1]) - float(lon[0]))
              * np.cos(np.deg2rad(float(lat[0]))))
        dy = np.deg2rad(float(lat[1]) - float(lat[0]))
        u_list.append(dx * Earth_Radius / dt_sec)
        v_list.append(dy * Earth_Radius / dt_sec)
        pm = tr.get("parent_member")
        if pm is None and tr.get("member_id") is not None:
            pm = f"e{int(tr['member_id']):02d}"
        parent_list.append(pm)
    if not lon_list:
        return None
    return {
        "lon": np.array(lon_list),
        "lat": np.array(lat_list),
        "u": np.array(u_list),
        "v": np.array(v_list),
        "parent": parent_list,
        "storm_dir": raw.get("storm_config", {}).get("storm_dir"),
    }


def sample_case(case_dir: Path, n_tracks=N_TRACKS,
                duration_days=DURATION_DAYS, seed=42):
    """markov_params_6h.pkl -> synthetic_tracks_*.nc for one case dir."""
    params_file = case_dir / "markov_params_6h.pkl"
    if not params_file.exists():
        return None
    with open(params_file, "rb") as f:
        data = pickle.load(f)

    mp = data.get("markov_params", {})
    step_params = mp.get("step_params")
    if not step_params:
        return None
    dt_hours = float(mp.get("dt_hours", 6.0))
    init_time = data.get("reference_time")
    storm_name = data.get("storm_config", {}).get("storm_name",
                                                  case_dir.parent.name)

    rng = np.random.default_rng(seed)
    init_cloud = _load_init_cloud(case_dir)

    lon_init = np.zeros(n_tracks)
    lat_init = np.zeros(n_tracks)
    u_init = v_init = None
    if init_cloud:
        idx = rng.integers(0, len(init_cloud["lon"]), size=n_tracks)
        lon_init = init_cloud["lon"][idx]
        lat_init = init_cloud["lat"][idx]
        if init_cloud.get("u") is not None:
            u_init = init_cloud["u"][idx]
            v_init = init_cloud["v"][idx]

    max_reliable_step = min(int(mp.get("max_reliable_step", 0)),
                            max(step_params.keys()))
    n_steps_req = int(duration_days * 24 / dt_hours) + 1
    n_steps = min(n_steps_req, max_reliable_step + 1)
    dt_seconds = dt_hours * 3600.0

    lon = np.zeros((n_tracks, n_steps))
    lat = np.zeros((n_tracks, n_steps))
    u = np.zeros((n_tracks, n_steps))
    v = np.zeros((n_tracks, n_steps))
    lon[:, 0], lat[:, 0] = lon_init, lat_init
    if u_init is not None:
        u[:, 0], v[:, 0] = u_init, v_init
    else:
        uv0 = rng.multivariate_normal(mp["mu_old"], mp["Sigma_oo"],
                                      size=n_tracks)
        u[:, 0], v[:, 0] = uv0[:, 0], uv0[:, 1]

    for i in range(1, n_steps):
        sp = step_params.get(i)
        if not sp:
            avail = [k for k in step_params if k <= i]
            sp = step_params[max(avail)] if avail else step_params[min(step_params)]
        prev_vec = np.stack([u[:, i - 1], v[:, i - 1]], axis=1)
        mean_cond = sp["mu_new"] + (sp["A"] @ (prev_vec - sp["mu_old"]).T).T
        curr = mean_cond + rng.multivariate_normal(
            [0, 0], sp["Sigma_cond"], size=n_tracks)
        u[:, i], v[:, i] = curr[:, 0], curr[:, 1]
        dlon_rad = ((u[:, i] * dt_seconds)
                    / (Earth_Radius * np.cos(np.deg2rad(lat[:, i - 1]))))
        dlat_rad = (v[:, i] * dt_seconds) / Earth_Radius
        lon[:, i] = ((lon[:, i - 1] + np.rad2deg(dlon_rad) + 180) % 360) - 180
        lat[:, i] = lat[:, i - 1] + np.rad2deg(dlat_rad)

    nc_path = case_dir / f"synthetic_tracks_{n_tracks}members.nc"
    if HAS_XARRAY:
        xr.Dataset(
            {
                "lon": (["track", "time"], lon),
                "lat": (["track", "time"], lat),
                "u": (["track", "time"], u),
                "v": (["track", "time"], v),
            },
            coords={
                "track": np.arange(n_tracks),
                "time": np.arange(n_steps) * dt_hours * 3600.0,
            },
            attrs={
                "storm": storm_name,
                "dt_hours": dt_hours,
                "init_time": str(init_time),
                "n_tracks": n_tracks,
                "duration_days": duration_days,
                "storm_dir": str(init_cloud.get("storm_dir") or ""),
            },
        ).to_netcdf(nc_path, format="NETCDF4")
    return nc_path


def run_sample_tracks(storm=None, year=None, n_tracks=N_TRACKS,
                      duration_days=DURATION_DAYS, plot=False):
    """Sample (and optionally plot) every case dir under one storm or all.

    storm: storm name (e.g. 'IRMA'); year optional but recommended to pick
    tracks/processed/{name}_{year}/ when several years share a name.
    """
    from plot_tracks import plot_case
    if storm:
        from config import storm_dir_name
        root = PROCESSED_TRACKS_DIR / (
            storm_dir_name(storm, year) if year else storm.lower())
        cases = sorted(root.glob("*/"))
    else:
        root = PROCESSED_TRACKS_DIR
        cases = sorted(root.glob("*/*/"))
    n = 0
    for case in cases:
        if not case.is_dir():
            continue
        nc = sample_case(case, n_tracks, duration_days)
        if nc:
            n += 1
            print(f"  {case.parent.name}/{case.name}: {nc.name}")
            plot_case(case, plot=plot,
                      bt_csv_dir=str(nc.parent))
    print(f"sample_tracks: {n} case(s)")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm", default="")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--n-tracks", type=int, default=N_TRACKS)
    ap.add_argument("--duration-days", type=float, default=DURATION_DAYS)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run_sample_tracks(args.storm or None, args.year, args.n_tracks,
                      args.duration_days, args.plot)
