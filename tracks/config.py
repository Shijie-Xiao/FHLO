"""Shared configuration for the FHLO synthetic-track pipeline.

Everything is driven by the project-level config.txt (see FHLO/config.txt):
    storms       = comma-separated  best-track storm dir names (blank = all)
    n_synth      = 1000             synthetic members per case
    duration_days= 11               requested forecast length (capped by the
                                    75% survival horizon of the parent set)

Directory layout (per forecast case = one ensemble cycle):
    tracks/processed/{storm_name}/{YYYYMMDDHH}/
        raw.pkl                parent ensemble member tracks (ECMWF TIGGE)
        pairs_6h.pkl           6h velocity pairs + step indices
        markov_params_6h.pkl   per-step k=1 Gaussian conditional fits
        synthetic_tracks_*.nc  sampled synthetic tracks (with parent_track)
        tracks.png             (optional) track plot

Data sources:
    ECMWF: /global/cfs/cdirs/m5011/Jay/TIGGE/ecmf/{year}/{YYYYMMDD}/*.xml
    Best track: data/ibtracs/{basin}/{year}/{STORM}/track_intensity_6h.csv
"""
from pathlib import Path
from datetime import datetime, timedelta
import os
import re

# ── Base directories ──────────────────────────────────────────────────────────
TRACKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACKS_DIR.parent
PROCESSED_TRACKS_DIR = TRACKS_DIR / "processed"


def _cfg_overrides():
    """Read path overrides from FHLO/config.txt (lowercase keys)."""
    cfg = {}
    p = PROJECT_ROOT / "config.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            cfg[k.strip().lower()] = v.strip()
    return cfg


_CFG = _cfg_overrides()
# Community archives (read-only inputs). Precedence: config.txt > env > default.
#   ecmwf_root : TIGGE ECMWF XML archive  ({year}/{YYYYMMDD}/*.xml)
#   output_dir : best-track root          ({basin}/{year}/{STORM}/)
ECMWF_BASE_DIR = Path(_CFG.get("ecmwf_root")
                      or os.environ.get("FHLO_ECMWF_ROOT")
                      or "/global/cfs/cdirs/m5011/Jay/TIGGE/ecmf")
BEST_TRACK_DIR = PROJECT_ROOT / _CFG.get("output_dir", "data/ibtracs")

PROCESSED_TRACKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
Earth_Radius = 6_378_100.0  # meters (WGS84 mean)
DT_HOURS = 6.0
N_TRACKS = 1000
DURATION_DAYS = 11.0
MIN_MEMBERS = 20             # parent members required to keep a cycle
SURVIVAL_FRAC = 0.75         # FHLO paper 75% survival rule


def storm_dir_name(storm_name: str, year) -> str:
    """Case root dir name: '{name}_{year}' (e.g. 'irma_2017')."""
    return f"{storm_name.lower()}_{year}"


# ── config.txt loading ────────────────────────────────────────────────────────
def load_project_config(path=None):
    """Parse FHLO/config.txt into a lowercase-key dict."""
    path = Path(path or PROJECT_ROOT / "config.txt")
    cfg = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        cfg[k.strip().lower()] = v.strip()
    return cfg


# ── Storm discovery ───────────────────────────────────────────────────────────
def _name_from_dir(dirname: str):
    """'2023232N13300_FRANKLIN' -> 'FRANKLIN' (None for UNNAMED/STORMxxx)."""
    m = re.match(r"^\d{4}\d+[NS]\d+_(.+)$", dirname)
    if not m:
        return None
    name = m.group(1).strip().upper()
    if name.startswith("UNNAMED") or name.startswith("STORM"):
        return None
    return name


def _bt_from_dir(storm_dir: Path):
    csv = storm_dir / "track_intensity_6h.csv"
    if not csv.exists():
        return None
    rows = []
    with open(csv) as f:
        f.readline()
        for ln in f:
            parts = ln.strip().split(",")
            if not parts or not parts[0]:
                continue
            try:
                rows.append(datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
    return rows or None


def discover_storms(storms_filter=None, years=None, basins=("NA",)):
    """All named storms with best-track data.

    storms_filter: list of storm dir names (e.g. ['2024181N09320_BERYL']);
                   None = every discovered storm.
    Returns [{storm_name, ibtracs_id, storm_dir, year, basin,
              genesis, last_time}]
    """
    want = set(storms_filter) if storms_filter else None
    out = []
    for basin in basins:
        bdir = BEST_TRACK_DIR / basin
        if not bdir.is_dir():
            continue
        for ydir in sorted(bdir.iterdir()):          # data/ibtracs/{basin}/{year}
            if not (ydir.is_dir() and ydir.name.isdigit()):
                continue
            for sdir in sorted(ydir.iterdir()):      # .../{year}/{STORM}/
                if not sdir.is_dir():
                    continue
                if want is not None and sdir.name not in want:
                    continue
                name = _name_from_dir(sdir.name)
                if not name:
                    continue
                times = _bt_from_dir(sdir)
                if not times:
                    continue
                out.append({
                    "storm_name": name,
                    "ibtracs_id": sdir.name,
                    "storm_dir": str(sdir),
                    "year": int(ydir.name),
                    "basin": basin,
                    "genesis": times[0],
                    "last_time": times[-1],
                })
    return out
