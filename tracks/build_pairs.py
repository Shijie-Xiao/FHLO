"""Build 6h velocity pairs for each case dir (raw.pkl -> pairs_6h.pkl).

Case layout: tracks/processed/{storm}/{YYYYMMDDHH}/
First velocity uses backward extrapolation (2*u1 - u2), never a copy.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Earth_Radius, PROCESSED_TRACKS_DIR, SURVIVAL_FRAC

DT_HOURS = 6.0


def time_to_step_index(time_val, reference_time, dt_hours=DT_HOURS,
                       tolerance_hours=3.0):
    """Convert absolute time to discrete forecast step index."""
    if reference_time is None or time_val is None:
        return None
    delta_hours = (time_val - reference_time).total_seconds() / 3600.0
    step_idx = int(round(delta_hours / dt_hours))
    if abs(delta_hours - step_idx * dt_hours) > tolerance_hours:
        return None
    return step_idx if step_idx >= 0 else None


def _compute_6h_velocities(track):
    """6h velocities; first point backward-extrapolated (2*u1 - u2)."""
    lon, lat, time_arr = track.get("lon"), track.get("lat"), track.get("datetime")
    if lon is None or lat is None or time_arr is None:
        return {}
    if min(len(lon), len(lat), len(time_arr)) < 2:
        return {}
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    time_np = np.array(time_arr, "datetime64[s]")
    n, dt_sec = len(lon), DT_HOURS * 3600.0
    u, v = np.zeros(n, float), np.zeros(n, float)
    lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
    for i in range(1, n):
        dx = (lon_rad[i] - lon_rad[i - 1]) * np.cos(lat_rad[i - 1]) * Earth_Radius
        dy = (lat_rad[i] - lat_rad[i - 1]) * Earth_Radius
        u[i], v[i] = dx / dt_sec, dy / dt_sec
    # Backward-extrapolate first velocity; a copied u[0]=u[1] makes the
    # step-1 pair perfectly correlated and degenerates the conditional fit.
    if n >= 3:
        u[0], v[0] = 2 * u[1] - u[2], 2 * v[1] - v[2]
    else:
        u[0], v[0] = u[1], v[1]
    return {"lon": lon, "lat": lat, "u": u, "v": v, "time": time_np}


def _build_pairs(u, v, step_times, ref_time):
    """Velocity pairs (u_{t-1}, v_{t-1}, u_t, v_t) with step indices."""
    if len(u) < 2:
        return np.zeros((0, 4)), np.array([], int)
    pairs = np.stack([u[:-1], v[:-1], u[1:], v[1:]], axis=1)
    steps = []
    for t in step_times[1:]:
        s = time_to_step_index(t, ref_time, DT_HOURS, 3.0)
        if s and s > 0:
            steps.append(int(s))
    if len(steps) != len(pairs):
        steps = list(range(1, len(pairs) + 1))
    return pairs, np.array(steps, int)


def survival_horizon(processed):
    """FHLO 75% rule: last step index where >=75% of members survive."""
    max_len = max(len(r["lon"]) for _, r in processed)
    min_req = int(np.ceil(SURVIVAL_FRAC * len(processed)))
    return next(
        (i - 1 for i in range(max_len)
         if sum(1 for _, r in processed if i < len(r["lon"])) < min_req),
        max_len - 1
    )


def build_case_pairs(case_dir: Path):
    """raw.pkl -> pairs_6h.pkl for one case dir. Returns n_pairs or None."""
    raw_file = case_dir / "raw.pkl"
    if not raw_file.exists():
        return None
    with open(raw_file, "rb") as f:
        raw = pickle.load(f)
    forced_init = raw.get("forced_init_time")
    if not forced_init:
        return None

    processed = [(tr, r) for tr in raw["tracks"]
                 if (r := _compute_6h_velocities(tr))]
    if not processed:
        return None

    max_valid_step = survival_horizon(processed)

    kept_pairs, kept_steps, kept_members = [], [], []
    for tr, r in processed:
        truncate_len = min(len(r["lon"]), max_valid_step + 1)
        if truncate_len < 2:
            continue
        pairs, steps = _build_pairs(
            r["u"][:truncate_len], r["v"][:truncate_len],
            r["time"][:truncate_len], forced_init
        )
        valid = [(pairs[i], steps[i])
                 for i, s in enumerate(steps) if 0 < s <= max_valid_step]
        if valid:
            kept_pairs.append(np.array([v[0] for v in valid]))
            kept_steps.append(np.array([v[1] for v in valid]))
            kept_members.append(int(tr["member_id"]))

    if not kept_pairs:
        return None

    all_pairs = np.vstack(kept_pairs)
    all_steps = np.concatenate(kept_steps)
    with open(case_dir / "pairs_6h.pkl", "wb") as f:
        pickle.dump({
            "storm_config": raw["storm_config"],
            "forced_init_time": forced_init,
            "members": kept_members,
            "velocity_pairs": all_pairs,
            "step_indices": all_steps,
            "dt_hours": DT_HOURS,
            "n_pairs": len(all_pairs),
            "n_tracks": len(kept_pairs),
            "max_valid_step": max_valid_step,
        }, f)
    return len(all_pairs)


def run_build_pairs(storm=None):
    """Build pairs for every case dir (optionally under one storm)."""
    root = PROCESSED_TRACKS_DIR / storm.lower() if storm else PROCESSED_TRACKS_DIR
    n = 0
    for case in sorted(root.glob("*/") if storm else root.glob("*/*/")):
        if not case.is_dir():
            continue
        r = build_case_pairs(case)
        if r:
            n += 1
            print(f"  {case.parent.name}/{case.name}: {r} pairs")
    print(f"build_pairs: {n} case(s)")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm", default="")
    args = ap.parse_args()
    run_build_pairs(args.storm or None)
