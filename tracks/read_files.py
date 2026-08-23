"""Read parent ensemble TC tracks from TIGGE for arbitrary (storm, cycle).

Two sources (selected via config.txt track_source):
  ECMWF: TIGGE XML cyclone products, 51 members
         /global/cfs/cdirs/m5011/Jay/TIGGE/ecmf/{year}/{YYYYMMDD}/*.xml
  GEFS : vortex tracking from pgrb2a GRIB2 (tracks/gefs_tracks.py)

A "case" is one storm at one ensemble cycle (base time). For each case we
write tracks/processed/{storm}/{YYYYMMDDHH}/raw.pkl.

Init-time policy (FHLO paper): run every 0000/1200 UTC cycle during the
storm's best-track lifetime that yields >= MIN_MEMBERS parent members.
"""
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ECMWF_BASE_DIR, PROCESSED_TRACKS_DIR, MIN_MEMBERS,
    discover_storms, storm_dir_name,
)


def _parse_dt(text):
    """Parse CXML time; tolerant of optional trailing 'Z' (2017 has Z)."""
    if not text:
        return None
    s = text.strip().rstrip("Z")
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def parse_tigge_xml(fp: Path, storm_name: str, base_time_filter: datetime = None):
    """Parse one TIGGE XML; return ensemble-member tracks of the named storm.

    Longitude/latitude conventions differ between TIGGE generations:
      2023+ : units='N'/'W' with SIGNED values
      2017-22: units='deg N'/'deg W' with unsigned magnitudes
    Hemisphere indicated by units OR a trailing letter forces the sign.
    """
    tracks = []
    try:
        root = ET.parse(fp).getroot()
        header = root.find("header")
        base_time = None
        if header is not None:
            bt = header.find("baseTime")
            if bt is not None and bt.text:
                base_time = _parse_dt(bt.text)
        if base_time_filter and base_time and \
                abs((base_time - base_time_filter).total_seconds()) > 3601:
            return [], base_time

        for data_elem in root.findall("data"):
            if data_elem.get("type", "") != "ensembleForecast":
                continue
            ms = data_elem.get("member")
            if ms is None:
                continue
            member_id = int(ms)
            if member_id < 0 or member_id > 50:
                continue
            for dist in data_elem.findall("disturbance"):
                ne = dist.find("cycloneName")
                cyclone_name = ne.text.strip() if ne is not None and ne.text else None
                if not cyclone_name or storm_name.upper() not in cyclone_name.upper():
                    continue
                num_e = dist.find("cycloneNumber")
                cyclone_num = None
                if num_e is not None and num_e.text:
                    digits = re.sub(r"\D", "", num_e.text)
                    cyclone_num = int(digits) if digits else None
                bas_e = dist.find("basin")
                basin = bas_e.text.strip() if bas_e is not None and bas_e.text else None

                lon_list, lat_list, time_list = [], [], []
                for fix in dist.findall("fix"):
                    lat_e = fix.find("latitude")
                    lon_e = fix.find("longitude")
                    time_e = fix.find("validTime")
                    if lat_e is None or lon_e is None:
                        continue
                    try:
                        lat_txt = lat_e.text.strip()
                        lat_units = (lat_e.get("units") or "").upper()
                        south = lat_txt.upper().endswith("S") or "S" in lat_units
                        lat_v = float(lat_txt.rstrip("NSns"))
                        if south and lat_v > 0:
                            lat_v = -lat_v
                        lon_txt = lon_e.text.strip()
                        lon_units = (lon_e.get("units") or "").upper()
                        west = lon_txt.upper().endswith("W") or "W" in lon_units
                        lon_v = float(lon_txt.rstrip("EWew"))
                        if west and lon_v > 0:
                            lon_v = -lon_v
                        lon_v = ((lon_v + 180) % 360) - 180
                        fix_time = base_time
                        if time_e is not None and time_e.text:
                            fix_time = _parse_dt(time_e.text) or base_time
                        if fix_time:
                            lon_list.append(lon_v)
                            lat_list.append(lat_v)
                            time_list.append(fix_time)
                    except Exception:
                        continue
                if len(lon_list) >= 2:
                    tracks.append({
                        "ensemble_system": "ecmwf", "member_id": member_id,
                        "storm_name": cyclone_name, "storm_number": cyclone_num,
                        "basin": basin, "init_time": base_time,
                        "lon": np.array(lon_list), "lat": np.array(lat_list),
                        "datetime": time_list,
                    })
    except Exception:
        return [], None
    return tracks, base_time


def candidate_cycles(storm, source="ecmwf", base_dir=None):
    """All 00/12 UTC cycles during the storm's BT lifetime, newest-first not
    needed; returns sorted list of datetimes."""
    t0, t1 = storm["genesis"], storm["last_time"]
    cycles = []
    t = datetime(t0.year, t0.month, t0.day)
    while t <= t1:
        for hh in (0, 12):
            cyc = t + timedelta(hours=hh)
            if t0 <= cyc <= t1:
                cycles.append(cyc)
        t += timedelta(days=1)
    return cycles


def _ecmwf_xml_for_cycle(year, cycle, base_dir=None):
    ddir = (base_dir or ECMWF_BASE_DIR) / str(year) / cycle.strftime("%Y%m%d")
    if not ddir.is_dir():
        return []
    return sorted(ddir.glob("*.xml"))


def read_case_ecmwf(storm, cycle, min_members=None, save=True):
    """Load one (storm, cycle) case from TIGGE XML; write raw.pkl."""
    min_members = min_members or MIN_MEMBERS
    members = {}
    for fp in _ecmwf_xml_for_cycle(storm["year"], cycle):
        trks, base_time = parse_tigge_xml(fp, storm["storm_name"],
                                          base_time_filter=cycle)
        if base_time is None or not trks:
            continue
        for tr in trks:
            mid = tr["member_id"]
            if mid not in members or len(tr["lon"]) > len(members[mid]["lon"]):
                members[mid] = tr
    if len(members) < min_members:
        return None
    tracks = []
    for mid in sorted(members):
        tr = dict(members[mid])
        tr["parent_member"] = f"e{mid:02d}"
        tracks.append(tr)
    return _package_case(storm, cycle, tracks, "ecmwf", save)


def _package_case(storm, cycle, tracks, source, save=True):
    times = [t["datetime"] for t in tracks if t.get("datetime")]
    result = {
        "storm_config": {**storm, "forced_init_time": cycle},
        "tracks": tracks,
        "n_tracks": len(tracks),
        "time_range": {
            "init_time": min(min(tt) for tt in times) if times else cycle,
            "end_time": max(max(tt) for tt in times) if times else cycle,
        },
        "forced_init_time": cycle,
        "ensemble_systems": [source],
    }
    if save:
        out_dir = case_dir(storm["storm_name"], storm["year"], cycle)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "raw.pkl", "wb") as f:
            pickle.dump(result, f)
    return result


def case_dir(storm_name: str, year, cycle: datetime) -> Path:
    """tracks/processed/{name}_{year}/{YYYYMMDDHH}"""
    return PROCESSED_TRACKS_DIR / storm_dir_name(storm_name, year) \
        / cycle.strftime("%Y%m%d%H")


# ── GEFS source ───────────────────────────────────────────────────────────────
def read_case_gefs(storm, cycle, min_members=None, save=True):
    """Load one (storm, cycle) from GEFS GRIB2 vortex tracking."""
    import gefs_tracks as gt
    min_members = min_members or MIN_MEMBERS
    # gefs_tracks.extract_gefs_tracks expects a case_dir + bt csv; locate by date
    case_root = gt.GFS_ROOT / str(storm["year"]) / cycle.strftime("%Y%m%d%H")
    if not case_root.is_dir():
        return None
    bt_csv = Path(storm["storm_dir"]) / "track_intensity_6h.csv"
    result = gt.extract_gefs_tracks(case_root, bt_csv,
                                    min_members=min_members)
    if result is None or result.get("n_tracks", 0) < min_members:
        return None
    # normalize to the shared package layout
    tracks = result["tracks"]
    for tr in tracks:
        tr["init_time"] = cycle
    return _package_case(storm, cycle, tracks, "gefs", save)


def read_case(storm, cycle, source="ecmwf", **kw):
    """Dispatch to the configured source for one (storm, cycle) case."""
    if source == "gefs":
        return read_case_gefs(storm, cycle, **kw)
    return read_case_ecmwf(storm, cycle, **kw)


def run_read_cases(storms=None, source="ecmwf", init_times=None):
    """Read all cycles for each storm. init_times: {storm_name: [dt,...]}
    to override the auto cycle list (e.g. paper case Irma 2017-09-05 00Z)."""
    storms = storms or discover_storms()
    n_ok = 0
    for storm in storms:
        cycles = (init_times or {}).get(storm["storm_name"]) \
            or candidate_cycles(storm, source)
        for cyc in cycles:
            r = read_case(storm, cyc, source=source)
            if r:
                n_ok += 1
                print(f"  {storm['storm_name']} {cyc:%Y-%m-%d %HZ} [{source}]: "
                      f"{r['n_tracks']} members")
            else:
                print(f"  [skip] {storm['storm_name']} {cyc:%Y-%m-%d %HZ} "
                      f"[{source}]: <{MIN_MEMBERS} members")
    print(f"read_cases: {n_ok} case(s) saved")
    return n_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecmwf", choices=["ecmwf", "gefs"])
    ap.add_argument("--storms", default="", help="comma-separated storm dir names")
    args = ap.parse_args()
    flt = [s for s in args.storms.split(",") if s.strip()] or None
    run_read_cases(discover_storms(storms_filter=flt), source=args.source)
