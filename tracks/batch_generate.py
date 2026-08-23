"""Batch FHLO synthetic-track generation: storm list -> cases -> sampling.

One command drives the whole chain for any number of storms:
    read cases (ECMWF/GEFS per config.txt) -> pairs -> per-step Markov fit
    -> 1000-member sampling -> optional per-case track plots.

Usage:
    python batch_generate.py --storms 2024181N09320_BERYL
    python batch_generate.py --storms 2024181N09320_BERYL,2017242N16333_IRMA \
        --init 2017242N16333_IRMA:2017090500 --plot
    python batch_generate.py --all --plot          # every discovered storm
    python batch_generate.py --stage sample,plot --storm IRMA   # rerun tail

Init-time policy (FHLO paper): every 0000/1200 UTC cycle during the storm's
best-track lifetime with >= MIN_MEMBERS parent ensemble members. Override
per storm with --init NAME:YYYYMMDDHH[,YYYYMMDDHH...].
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    load_project_config, discover_storms, N_TRACKS, DURATION_DAYS,
    PROCESSED_TRACKS_DIR,
)


def parse_init_overrides(spec):
    """"NAME:YYYYMMDDHH[+YYYYMMDDHH...] -> {NAME: [datetime,...]}"""
    out = {}
    if not spec:
        return out
    for tok in spec.split(","):
        if ":" not in tok:
            continue
        name, times = tok.split(":", 1)
        dts = []
        for t in times.split("+"):
            t = t.strip()
            if t:
                dts.append(datetime.strptime(t, "%Y%m%d%H"))
        if dts:
            out[name.strip().upper()] = dts
    return out


def read_cycles_for(storm, cycles_mode="genesis", source="ecmwf",
                    init_overrides=None):
    """Resolve which cycles to read for one storm.

    cycles_mode:
      'genesis'  (default) – the first 00/12 UTC cycle at/after the IBTrACS
                  genesis time; falls back to the next cycle if the ensemble
                  has < MIN_MEMBERS members there
      'all'      – every 00/12 UTC cycle in the storm's lifetime (paper-scale
                  reforecast mode)
    init_overrides[name] – explicit list, wins over both modes
    """
    import read_files as rf
    if init_overrides and storm["storm_name"] in init_overrides:
        return init_overrides[storm["storm_name"]]
    if cycles_mode == "all":
        return rf.candidate_cycles(storm, source)
    # genesis mode: first viable 00/12Z cycle at/after genesis
    cyc = storm["genesis"]
    cyc = cyc.replace(hour=0) if cyc.hour < 12 else cyc.replace(hour=12)
    out = []
    for _ in range(4):  # try up to 4 consecutive cycles
        out.append(cyc)
        cyc += __import__("datetime").timedelta(hours=12)
    return out


def run(storms_filter=None, source=None, init_overrides=None,
        cycles="genesis", n_tracks=N_TRACKS, duration_days=DURATION_DAYS,
        plot=False, stages=("read", "pairs", "fit", "sample")):
    cfg = load_project_config()
    source = (source or cfg.get("track_source", "ECMWF")).upper()
    storms_filter = storms_filter or [
        s for s in cfg.get("storms", "").split(",") if s.strip()]

    storms = discover_storms(storms_filter=storms_filter)
    if not storms:
        print("No storms found (check config.txt storms= / data/ibtracs)")
        return

    import read_files
    import build_pairs
    import train_markov
    import sample_tracks
    from plot_tracks import plot_case

    print(f"Storms: {[s['storm_name'] for s in storms]} | source={source} | "
          f"cycles={cycles} | stages={list(stages)} | plot={plot}")

    # ── stage 1: read parent ensembles per (storm, cycle) ────────────────────
    if "read" in stages:
        n_read = 0
        for storm in storms:
            for cyc in read_cycles_for(storm, cycles, source.lower(),
                                       init_overrides):
                r = read_files.read_case(storm, cyc, source=source.lower())
                if r:
                    n_read += 1
                    print(f"  [read] {storm['storm_name']} {cyc:%Y-%m-%d %HZ}: "
                          f"{r['n_tracks']} members")
        print(f"[read] {n_read} case(s)")

    # ── stages 2-4 per storm (case dirs live under tracks/processed/{name}_{year}) ──
    total = 0
    for storm in storms:
        from config import storm_dir_name
        sdir = PROCESSED_TRACKS_DIR / storm_dir_name(
            storm["storm_name"], storm["year"])
        if not sdir.is_dir():
            continue
        for case in sorted(sdir.glob("*/")):
            if not case.is_dir():
                continue
            label = f"{storm['storm_name']}/{case.name}"
            if "pairs" in stages:
                n = build_pairs.build_case_pairs(case)
                if n:
                    print(f"  [pairs] {label}: {n} pairs")
            if "fit" in stages:
                n = train_markov.train_case(case)
                if n:
                    print(f"  [fit] {label}: {n} steps")
            if "sample" in stages:
                nc = sample_tracks.sample_case(case, n_tracks, duration_days)
                if nc:
                    total += 1
                    print(f"  [sample] {label}: {nc.name}")
                    png = plot_case(case, plot=plot)
                    if png:
                        print(f"  [plot] {label}: {png.name}")
    print(f"batch_generate: {total} sampled case(s) "
          f"({time.strftime('%H:%M:%S')})")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storms", default="", help="comma-separated storm dir names")
    ap.add_argument("--all", action="store_true", help="run every discovered storm")
    ap.add_argument("--source", default="", choices=["", "ECMWF", "GEFS"])
    ap.add_argument("--init", default="",
                    help="per-storm init override NAME:YYYYMMDDHH[+...]")
    ap.add_argument("--n-tracks", type=int, default=N_TRACKS)
    ap.add_argument("--duration-days", type=float, default=DURATION_DAYS)
    ap.add_argument("--stage", default="read,pairs,fit,sample",
                    help="comma list: read,pairs,fit,sample,plot")
    ap.add_argument("--cycles", default="genesis", choices=["genesis", "all"],
                    help="genesis: IBTrACS genesis 00/12Z cycle (default); "
                         "all: every 00/12Z cycle in the storm lifetime")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    stages = tuple(s.strip().lower() for s in args.stage.split(",") if s.strip())
    plot = args.plot or "plot" in stages
    storms_filter = None if args.all else (
        [s for s in args.storms.split(",") if s.strip()] or None)

    run(storms_filter=storms_filter,
        source=args.source or None,
        init_overrides=parse_init_overrides(args.init),
        cycles=args.cycles,
        n_tracks=args.n_tracks,
        duration_days=args.duration_days,
        plot=plot,
        stages=stages)


if __name__ == "__main__":
    main()
