#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the FAST-ODE ensemble (1000 members) from ensemble_fast.nc.

Adapted from plot_vs_google_vmax.py, FAST-only (no ML branch): spaghetti +
mean/top-10% + obs (v_obz) in the top panel, vent (chi*s) in the bottom.
Optional overlays: IBTrACS best-track CSV and the Google WeatherLab FNV3
paired CSV for the same init.

Usage:
  python plot_ensemble_fast.py --ens-nc data/ensemble/beryl_gefs/ensemble_fast.nc \
      --out_png data/ensemble/beryl_gefs/ensemble_fast.png --storm BERYL
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ens_stats(arr, top_frac=0.10):
    n = arr.shape[0]
    nt = max(1, int(n * top_frac))
    top = np.argsort(np.nanmax(arr, 1))[-nt:]
    return np.nanmean(arr, 0), np.nanmean(arr[top], 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ens-nc", required=True, help="ensemble_fast.nc")
    p.add_argument("--out_png", required=True)
    p.add_argument("--storm", default=None)
    p.add_argument("--init_time", default=None)
    p.add_argument("--fc_start", default=None,
                   help="forecast start (GEFS init); with a replay window "
                        "the hour axis is re-origin'ed here: negative hours "
                        "= ERA5 replay segment, 0+ = GEFS forecast")
    p.add_argument("--google_csv", default=None,
                   help="optional Google FNV3 paired CSV overlay")
    p.add_argument("--google_id", default=None)
    p.add_argument("--google_lead_shift_h", type=float, default=0.0)
    p.add_argument("--max_hours", type=float, default=None)
    args = p.parse_args()

    ds = xr.open_dataset(args.ens_nc)
    fast = ds["fast_vmax_kts"].values
    obs = ds["v_obz_kts"].values
    sl = ds["seq_len"].values if "seq_len" in ds else None
    chi = ds["fast_chi"].values if "fast_chi" in ds else None
    s = ds["fast_s"].values if "fast_s" in ds else None
    init = args.init_time or str(ds.attrs.get("init_time", ""))
    env = str(ds.attrs.get("env", ""))
    mode = str(ds.attrs.get("ode_mode", "cold"))
    rep_h = int(ds.attrs.get("replay_hours", 0) or 0)
    nmem = ds.sizes["member"]
    time0 = ds["time"].values[0] if "time" in ds.coords else None
    ds.close()

    T = fast.shape[1]
    h = np.arange(T, dtype=float)
    # re-origin the hour axis to the forecast start when replaying
    replayed = bool(args.fc_start) and rep_h > 0
    if replayed and time0 is not None:
        h = h - float((np.datetime64(pd.Timestamp(args.fc_start)) - time0)
                      / np.timedelta64(1, "h"))
    # mask beyond each member's valid length
    if sl is not None:
        for m in range(fast.shape[0]):
            n = int(sl[m])
            if n < T:
                fast[m, n:] = np.nan
                obs[m, n:] = np.nan
                if chi is not None:
                    chi[m, n:] = np.nan
                if s is not None:
                    s[m, n:] = np.nan

    fa_mean, fa_top = _ens_stats(fast)
    obs0 = obs[0]
    Tobs = int(np.isfinite(obs0).sum())

    # Google overlay (optional; CSV from ensemble/download_fnv3.py)
    g_lead = g_mean = g_topm = g_p10 = g_p90 = g_mat = None
    nG = 0
    g_init = ""
    if args.google_csv and Path(args.google_csv).exists():
        g = pd.read_csv(args.google_csv, comment="#")
        if args.google_id:
            g = g[g["track_id"] == args.google_id]
        gcol = "maximum_sustained_wind_speed_knots"
        g = g.copy()
        if "init_time" in g.columns and len(g):
            g_init = str(pd.Timestamp(g["init_time"].iloc[0]))
        g["lead_h"] = (pd.to_timedelta(g["lead_time"]).dt.total_seconds()
                       / 3600.0 + args.google_lead_shift_h)
        g_lead = np.sort(g["lead_h"].unique())
        g_by = {smp: grp.set_index("lead_h")[gcol]
                for smp, grp in g.groupby("sample")}
        nG = len(g_by)
        g_mat = np.full((nG, len(g_lead)), np.nan)
        for i, (_, ser) in enumerate(g_by.items()):
            g_mat[i] = np.interp(g_lead, ser.index.values, ser.values,
                                 left=np.nan, right=np.nan)
        g_mean = np.nanmean(g_mat, 0)
        g_p10 = np.nanpercentile(g_mat, 10, 0)
        g_p90 = np.nanpercentile(g_mat, 90, 0)
        gt = np.argsort(np.nanmax(g_mat, 1))[-max(1, int(nG * 0.10)):]
        g_topm = np.nanmean(g_mat[gt], 0)

    # vent panel needs both chi and s
    two = chi is not None and s is not None
    if two:
        vent = chi * s
        v_mean = np.nanmean(vent, 0)
        v_lo = np.nanpercentile(vent, 10, 0)
        v_hi = np.nanpercentile(vent, 90, 0)

    if two:
        fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 9), dpi=170,
                                      sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(11, 6), dpi=180)

    # ---- top: intensity ----
    if g_mat is not None:
        ax.fill_between(g_lead, g_p10, g_p90, color="#dc2626", alpha=0.10,
                        zorder=1)
        for i in range(nG):
            ax.plot(g_lead, g_mat[i], color="#fecaca", lw=0.3, alpha=0.28,
                    zorder=1)
    for i in range(fast.shape[0]):
        ax.plot(h, fast[i], color="#86efac", lw=0.2, alpha=0.18, zorder=2)
    ax.plot(h, obs0, "k-", lw=3.0, zorder=9,
            label=f"IBTrACS (peak {np.nanmax(obs0[:Tobs]):.0f})")
    ax.plot(h, fa_mean, color="#16a34a", lw=2.6, zorder=8,
            label=f"FAST mean (peak {np.nanmax(fa_mean):.0f})")
    ax.plot(h, fa_top, color="#16a34a", lw=2.0, ls="--", zorder=8,
            label=f"FAST top10% (peak {np.nanmax(fa_top):.0f})")
    if g_mean is not None:
        ax.plot(g_lead, g_mean, color="#dc2626", lw=2.2, marker="o", ms=4,
                zorder=8,
                label=f"Google FNV3 mean, init {pd.Timestamp(g_init):%m-%d %HZ} "
                      f"(peak {np.nanmax(g_mean):.0f}, {nG} mem)")
        ax.plot(g_lead, g_topm, color="#dc2626", lw=1.8, ls="--", marker="s",
                ms=3.5, mfc="none", zorder=8,
                label=f"Google FNV3 top10% (peak {np.nanmax(g_topm):.0f})")
        ax.plot(g_lead, g_p90, color="#dc2626", lw=1.2, ls=":",
                zorder=7, label="Google FNV3 p90")
    xmax = args.max_hours if args.max_hours is not None else \
        max(Tobs, float(g_lead.max()) if g_lead is not None else 0)
    xmin = float(h[0]) if replayed else 0.0
    ax.set_xlim(xmin, xmax)
    if replayed:
        ax.axvline(0.0, color="#2563eb", lw=1.4, ls="-.", alpha=0.8)
        ax.text(0.985, 0.02,
                f"forecast start {pd.Timestamp(args.fc_start):%m-%d %HZ} "
                f"(h<0: ERA5 obs replay, not forecast)",
                transform=ax.transAxes, fontsize=8.5, color="#2563eb",
                ha="right")
    ax.set_ylim(0, max(160, np.nanmax(obs0[:Tobs]) + 30))
    ax.grid(alpha=0.25)
    ax.set_ylabel("Intensity (kt)")
    sname = (args.storm or "").strip().upper()
    envlab = {"gefs": "GEFS fcst env", "era5": "ERA5 analysis env"}.get(env, env)
    gtxt = "" if g_mean is None else \
        f" vs Google FNV3 (init {g_init})"
    mtxt = "" if mode in ("", "cold") else \
        f", {mode} init ({rep_h}h obs replay)"
    fc_txt = f", fc start {pd.Timestamp(args.fc_start):%m-%d %HZ}" \
        if replayed else f", init {init}" if init else ""
    ax.set_title((f"{sname}: " if sname else "")
                 + f"FAST ensemble ({nmem} members, {envlab}{mtxt}"
                   f"{fc_txt}){gtxt} vs IBTrACS",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    # ---- bottom: vent chi*s ----
    if replayed:
        xlab = (f"Hours since forecast start {pd.Timestamp(args.fc_start):%m-%d %HZ} "
                f"(GEFS init: {init}; h<0: ERA5 obs replay)")
    else:
        xlab = "Hours since init" + (f"   (init: {init})" if init else "")
    if two:
        axv.fill_between(h, v_lo, v_hi, color="#16a34a", alpha=0.15)
        axv.plot(h, v_mean, color="#16a34a", lw=2.2,
                 label=f"vent = chi*s  (mean {np.nanmean(v_mean):.1f})")
        axv.grid(alpha=0.25)
        axv.set_ylabel("vent (chi*S)")
        axv.set_xlabel(xlab)
        axv.legend(loc="upper right", fontsize=9, framealpha=0.9)
        axv.set_ylim(0, None)
    else:
        ax.set_xlabel(xlab)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    peaks = np.nanmax(fast, axis=1)
    if replayed:
        m_fc = h >= 0
        fa_mean_pk = np.nanmax(fa_mean[m_fc])
        fa_top_pk = np.nanmax(fa_top[m_fc])
        peaks = np.nanmax(np.where(m_fc[None, :], fast, np.nan), axis=1)
    else:
        fa_mean_pk = np.nanmax(fa_mean)
        fa_top_pk = np.nanmax(fa_top)
    print(f"  saved {out.with_suffix('.svg')}")
    print(f"  FAST mean peak={fa_mean_pk:.0f} "
          f"top10%={fa_top_pk:.0f} | "
          f"member peaks p10={np.nanpercentile(peaks,10):.0f} "
          f"p50={np.nanpercentile(peaks,50):.0f} "
          f"p90={np.nanpercentile(peaks,90):.0f} | "
          f"obs={np.nanmax(obs0[:Tobs]):.0f}")
    if g_mat is not None:
        gpk = np.nanmax(g_mat, axis=1)
        print(f"  Google FNV3 ({nG} mem) mean peak={np.nanmean(gpk):.1f} "
              f"med={np.nanmedian(gpk):.1f} p10={np.nanpercentile(gpk,10):.0f} "
              f"p90={np.nanpercentile(gpk,90):.0f} max={np.nanmax(gpk):.0f}")


if __name__ == "__main__":
    main()
