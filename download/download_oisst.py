#!/usr/bin/env python3
"""Download NOAA OISST v2.1 daily SST (0.25°, AVHRR-only) for a storm window.

The FHLO paper (Lin et al. 2020, section 2) uses the NCEI 0.25° Optimum
Interpolation SST dataset (Reynolds et al. 2008) for real-time SST estimates.
The current production version is v2.1 (AVHRR-only, final files ~2 weeks
behind real time; use --nrt for the 1-day-latency preliminary version).

Two access modes:
  direct (default): per-day NetCDF files from the NCEI HTTPS archive
      https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/
  erddap: server-side subset via the NOAA CoastWatch ERDDAP griddap service
      (recommended when a regional box is enough — much smaller downloads)

Usage:
    python download_oisst.py --storm irma
    python download_oisst.py --start 2017-09-05 --end 2017-09-12 \
        --lat 15 30 --lon 280 320
    python download_oisst.py --storm irma --mode erddap --pad 5
    python download_oisst.py --storm beryl --nrt

Output:
    OISST/{storm}/oisst-avhrr-v02r01.YYYYMMDD.nc      (direct mode)
    OISST/{storm}/oisst_subset_YYYYMMDD_YYYYMMDD.nc   (erddap mode)

Dependencies:
    requests, xarray, netCDF4 (erddap mode also benefits from dask)
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# NCEI direct archive (full-globe daily files, ~1.6 MB/day compressed)
# ---------------------------------------------------------------------------
NCEI_BASE = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-"
             "optimum-interpolation/v2.1/access/{variant}")
FILE_TMPL = "oisst-avhrr-v02r01.{yyyymmdd}.nc"

# ---------------------------------------------------------------------------
# CoastWatch ERDDAP griddap (server-side spatial/temporal subsetting)
#   final: ncdcOisst21Agg   (final product, ~2 week latency)
#   nrt:   ncdcOisst21NrtAgg (preliminary, ~1 day latency)
# ---------------------------------------------------------------------------
ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/{dataset}.nc"
ERDDAP_DATASET = {"final": "ncdcOisst21Agg", "nrt": "ncdcOisst21NrtAgg"}

# Storm presets: best-track time windows and a generous regional box.
# lon in 0-360 degE. Times follow the FHLO reforecast convention
# (init cycle at 00/12 UTC).
STORMS = {
    "irma": dict(
        start=date(2017, 9, 5), end=date(2017, 9, 12),
        lat=(10.0, 30.0), lon=(275.0, 320.0), desc="Irma 2017 NA"),
    "maria": dict(
        start=date(2017, 9, 18), end=date(2017, 9, 25),
        lat=(10.0, 30.0), lon=(285.0, 325.0), desc="Maria 2017 NA"),
    "beryl": dict(
        start=date(2024, 6, 28), end=date(2024, 7, 10),
        lat=(8.0, 35.0), lon=(280.0, 330.0), desc="Beryl 2024 NA"),
    "erin": dict(
        start=date(2025, 8, 14), end=date(2025, 8, 22),
        lat=(15.0, 40.0), lon=(290.0, 330.0), desc="Erin 2025 NA"),
}


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def http_get(url: str, dst: Path, retries: int = 4, timeout: int = 300) -> bool:
    """Stream a URL to dst with retry + resume; True on success."""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[FOUND] {dst.name}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "FHLO-OISST-downloader/1.0"}
    for attempt in range(1, retries + 1):
        try:
            h = dict(headers)
            if resume_from > 0:
                h["Range"] = f"bytes={resume_from}-"
            with requests.get(url, headers=h, stream=True, timeout=timeout) as r:
                if r.status_code == 416:            # resume point already EOF
                    break
                r.raise_for_status()
                mode = "ab" if resume_from > 0 and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.rename(dst)
            print(f"[OK] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"[RETRY {attempt}/{retries}] {dst.name}: {e}")
            time.sleep(5 * attempt)
            resume_from = tmp.stat().st_size if tmp.exists() else 0
    print(f"[FAIL] {dst.name}")
    return False


def download_direct(out_dir: Path, d0: date, d1: date, nrt: bool):
    """Per-day files from the NCEI HTTPS archive."""
    variant = "avhrr-nrt" if nrt else "avhrr"
    base = NCEI_BASE.format(variant=variant)
    n_ok = 0
    days = list(daterange(d0, d1))
    for d in days:
        url = f"{base}/{d.strftime('%Y%m')}/{FILE_TMPL.format(yyyymmdd=d.strftime('%Y%m%d'))}"
        dst = out_dir / FILE_TMPL.format(yyyymmdd=d.strftime("%Y%m%d"))
        n_ok += http_get(url, dst)
    print(f"[Done] direct: {n_ok}/{len(days)} days -> {out_dir}")


def download_erddap(out_dir: Path, d0: date, d1: date,
                    lat: tuple, lon: tuple, nrt: bool, pad: float):
    """Server-side subset via ERDDAP griddap; one combined NetCDF."""
    dataset = ERDDAP_DATASET["nrt" if nrt else "final"]
    lat0, lat1 = lat[0] - pad, lat[1] + pad
    lon0, lon1 = lon[0] - pad, lon[1] + pad
    lat0, lat1 = max(-89.875, lat0), min(89.875, lat1)
    lon0 = lon0 % 360.0
    lon1 = lon0 + (lon1 - lon0)            # keep span, normalise origin
    query = (f"?sst[({d0.isoformat()}T00:00:00Z):1:({d1.isoformat()}T00:00:00Z)]"
             f"[(0.0):1:(0.0)]"
             f"[({lat0:.3f}):1:({lat1:.3f})]"
             f"[({lon0:.3f}):1:({lon1:.3f})]")
    url = ERDDAP_URL.format(dataset=dataset) + query
    dst = out_dir / f"oisst_subset_{d0.strftime('%Y%m%d')}_{d1.strftime('%Y%m%d')}.nc"
    ok = http_get(url, dst)
    if ok:
        print(f"[Done] erddap ({dataset}): {dst}")
    else:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--storm", choices=sorted(STORMS), help="storm preset")
    g.add_argument("--start", help="start date YYYY-MM-DD (with --end)")
    p.add_argument("--end", help="end date YYYY-MM-DD (inclusive)")
    p.add_argument("--lat", nargs=2, type=float, metavar=("S", "N"),
                   help="latitude box (direct mode ignores this)")
    p.add_argument("--lon", nargs=2, type=float, metavar=("W", "E"),
                   help="longitude box in 0-360 degE (direct mode ignores this)")
    p.add_argument("--mode", choices=["direct", "erddap"], default="direct",
                   help="direct: per-day NCEI files (global); erddap: regional subset")
    p.add_argument("--pad", type=float, default=2.0,
                   help="degree pad added around the box (erddap mode)")
    p.add_argument("--nrt", action="store_true",
                   help="use the near-real-time preliminary product (1-day latency)")
    p.add_argument("--out", default=None, help="output root (default download/OISST)")
    args = p.parse_args()

    if args.storm:
        cfg = STORMS[args.storm]
        d0, d1 = cfg["start"], cfg["end"]
        lat, lon = cfg["lat"], cfg["lon"]
        tag = args.storm
    elif args.start and args.end:
        d0 = datetime.strptime(args.start, "%Y-%m-%d").date()
        d1 = datetime.strptime(args.end, "%Y-%m-%d").date()
        lat = tuple(args.lat) if args.lat else (0.0, 60.0)
        lon = tuple(args.lon) if args.lon else (260.0, 360.0)
        tag = f"{d0.strftime('%Y%m%d')}_{d1.strftime('%Y%m%d')}"
    else:
        p.error("specify --storm or both --start and --end")

    out_root = Path(args.out) if args.out else Path(__file__).parent / "OISST"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    if not args.nrt and d1 > today - timedelta(days=16):
        print(f"[WARN] final product lags ~2 weeks; {d1} may not exist yet "
              f"(consider --nrt)")

    print("=" * 70)
    print(f"OISST v2.1 (AVHRR-only, 0.25°, daily) | mode={args.mode} "
          f"| {'NRT' if args.nrt else 'final'}")
    print(f"window: {d0} .. {d1}  | box: lat {lat}, lon {lon} (degE)")
    print(f"output: {out_dir}")
    print("=" * 70)

    if args.mode == "direct":
        download_direct(out_dir, d0, d1, args.nrt)
    else:
        download_erddap(out_dir, d0, d1, lat, lon, args.nrt, args.pad)


if __name__ == "__main__":
    main()
