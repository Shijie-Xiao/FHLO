"""Reusable track plotting for FHLO synthetic-track cases.

Single entry point:
    plot_case(case_dir, plot=True) -> png path or None
    plot_arrays(lon, lat, parents, bt, title, out_png)

Draws: gray synthetic members, light-blue parent ensemble members, blue
synthetic ensemble mean, black best track, red init point. Coastlines via
cartopy; falls back to plain axes if unavailable.
"""
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def _load_case_arrays(case_dir: Path):
    """Load synthetic NC + parent raw.pkl for one case dir."""
    import pickle
    import xarray as xr
    nc_file = next(case_dir.glob("synthetic_tracks_*.nc"), None)
    if nc_file is None:
        return None
    ds = xr.open_dataset(nc_file)
    lon, lat = ds["lon"].values, ds["lat"].values

    parents = None
    raw_file = case_dir / "raw.pkl"
    if raw_file.exists():
        with open(raw_file, "rb") as f:
            raw = pickle.load(f)
        parents = [(np.asarray(t["lon"], float), np.asarray(t["lat"], float))
                   for t in raw["tracks"]
                   if t.get("lon") is not None and len(t["lon"]) >= 2]
    init_time = ds.attrs.get("init_time", "")
    ds.close()
    return {"lon": lon, "lat": lat, "parents": parents, "init_time": init_time}


def load_best_track(storm_dir, init_time, n_steps, dt_hours=6.0):
    """Interpolated best track on the case time grid (or None)."""
    import pandas as pd
    csv = Path(storm_dir) / "track_intensity_6h.csv"
    if not csv.exists():
        return None
    bt = pd.read_csv(csv)
    bt["time"] = pd.to_datetime(bt["time"])
    init = pd.Timestamp(str(init_time))
    if not isinstance(init_time, str):
        init = pd.Timestamp(init_time)
    grid = init + pd.to_timedelta(np.arange(n_steps) * dt_hours, unit="h")
    t0 = bt["time"].iloc[0]
    s = (bt["time"] - t0).dt.total_seconds().values
    sg = (grid - t0).total_seconds().values
    return (np.interp(sg, s, bt["lon"]), np.interp(sg, s, bt["lat"]))


def plot_arrays(lon, lat, parents, bt, title, out_png, max_draw=1000):
    """Core drawing routine. bt: (lon, lat) tuple or None."""
    if not HAS_MPL:
        return None
    n_tracks = lon.shape[0]

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        transform = None

    pool_lon = [lon.flatten()]
    pool_lat = [lat.flatten()]
    if parents:
        for pl, pa in parents:
            pool_lon.append(np.asarray(pl))
            pool_lat.append(np.asarray(pa))
    if bt is not None:
        pool_lon.append(np.asarray(bt[0]))
        pool_lat.append(np.asarray(bt[1]))
    pool_lon = np.concatenate(pool_lon)
    pool_lat = np.concatenate(pool_lat)
    lon_min, lon_max = np.nanmin(pool_lon), np.nanmax(pool_lon)
    lat_min, lat_max = np.nanmin(pool_lat), np.nanmax(pool_lat)
    lon_b = max((lon_max - lon_min) * 0.10, 4.0)
    lat_b = max((lat_max - lat_min) * 0.10, 3.0)

    if HAS_CARTOPY:
        ax.set_extent([lon_min - lon_b, lon_max + lon_b,
                       lat_min - lat_b, lat_max + lat_b],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=4)
        ax.add_feature(cfeature.LAND, facecolor="0.95", alpha=0.8, zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="0.98", alpha=0.8, zorder=1)

    if parents:
        for pl, pa in parents:
            ax.plot(pl, pa, color="#9ecae1", linewidth=1.0, alpha=0.55,
                    transform=transform, zorder=2)
    sel = np.arange(min(max_draw, n_tracks))
    for i in sel:
        ax.plot(lon[i], lat[i], color="#B0B0B0", linewidth=0.6, alpha=0.5,
                transform=transform, zorder=3)
    ax.plot(lon.mean(0), lat.mean(0), color="#1f77b4", linewidth=2.6,
            transform=transform, zorder=7, label="Ensemble mean")
    if bt is not None:
        ax.plot(bt[0], bt[1], "k-", linewidth=2.6, transform=transform,
                zorder=8, label="Best track")
        ax.plot(bt[0][0], bt[1][0], "k*", markersize=13, zorder=9,
                transform=transform)
    ax.plot(lon[0, 0], lat[0, 0], marker="o", color="#d62728", markersize=6,
            transform=transform, zorder=9, label="Init")

    ax.set_title(title, fontsize=12, fontweight="bold")
    if HAS_CARTOPY:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_case(case_dir, plot=True, bt_csv_dir=None):
    """Plot one case dir (synthetic + parents + best track).

    plot=False: no-op (batch mode without plotting).
    Returns png path (or None when not plotting / nothing to plot).
    """
    if not plot:
        return None
    data = _load_case_arrays(Path(case_dir))
    if data is None:
        return None
    storm_dir = bt_csv_dir or data.get("storm_dir")
    bt = None
    if storm_dir:
        try:
            bt = load_best_track(storm_dir, data["init_time"],
                                 data["lon"].shape[1])
        except Exception:
            bt = None
    title = (f"{Path(case_dir).parent.name} {Path(case_dir).name} – "
             f"synthetic tracks ({data['lon'].shape[0]})")
    return plot_arrays(data["lon"], data["lat"], data["parents"], bt,
                       title, Path(case_dir) / "tracks.png")
