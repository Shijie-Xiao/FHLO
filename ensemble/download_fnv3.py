#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the Google WeatherLab FNV3 paired-CSV matching THIS library's
ensemble init time, for direct overlay comparison.

Simplified single-case port of the PINN bulk downloader:
  - init time comes from config.txt `gefs_init_{tag}` (the SAME init the
    GEFS ensemble uses, so the two forecasts line up hour-for-hour), no
    ecmwf_raw/genesis heuristic;
  - one CSV per call: FNV3 files are GLOBAL (all storms active at that
    init), so a single fetch covers the case;
  - output lands in data/google/ next to the rest of the demo data.

Usage:
    python ensemble/download_fnv3.py                     # config storms + init
    python ensemble/download_fnv3.py --init '2025-06-29 06:00' --out-dir data/google
    python ensemble/download_fnv3.py --dry-run

After downloading, `run.py --ensemble --stage plot` overlays it automatically
(keyed by `google_id_{tag}` in config.txt).
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FNV3_BASE = ('https://deepmind.google.com/science/weatherlab/download/'
             'cyclones/FNV3/ensemble/paired/csv')


def load_cfg(path=None):
    path = Path(path or PROJECT_ROOT / 'config.txt')
    cfg = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k, v = [x.strip() for x in s.split('=', 1)]
            cfg[k.lower()] = v
    return cfg


def snap_to_cycle(ts: datetime) -> datetime:
    """Snap to the nearest 6h FNV3 init cycle (00/06/12/18Z)."""
    best = None
    for day_off in (0, 1):
        for c in (0, 6, 12, 18):
            cand = datetime(ts.year, ts.month, ts.day, c) + \
                __import__('datetime').timedelta(days=day_off)
            d = abs((cand - ts).total_seconds())
            if best is None or d < best[0]:
                best = (d, cand)
    return best[1]


def curl_one(url: str, dst: Path, timeout: int = 120):
    try:
        subprocess.run(['curl', '-sSL', '-m', str(timeout), '-o', str(dst), url],
                       capture_output=True, text=True)
        if not dst.exists() or dst.stat().st_size < 1000:
            sz = dst.stat().st_size if dst.exists() else 0
            if dst.exists():
                dst.unlink(missing_ok=True)
            return False, f'size={sz}B'
        head = dst.read_text(errors='ignore')[:60]
        if head.lstrip().startswith('<'):
            dst.unlink(missing_ok=True)
            return False, 'html_error_page'
        return True, f'{dst.stat().st_size // 1024}KB'
    except Exception as e:
        return False, repr(e)[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init', default='',
                    help='FNV3 init time (default: config gefs_init_{tag}, '
                         'snapped to the nearest 6h cycle)')
    ap.add_argument('--out-dir', default=str(PROJECT_ROOT / 'data' / 'google'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = load_cfg()
    tag = 'flossie'
    storms = [s.strip() for s in cfg.get('storms', '').split(',') if s.strip()]
    if storms:
        tag = storms[0].split('_', 1)[-1].lower()

    init_str = args.init or cfg.get(f'gefs_init_{tag}', '')
    if not init_str:
        print('no init: pass --init or set gefs_init_%s in config.txt' % tag)
        return
    init = snap_to_cycle(datetime.fromisoformat(init_str))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'FNV3_{init:%Y_%m_%dT%H_00}_paired.csv'
    url = f'{FNV3_BASE}/FNV3_{init:%Y_%m_%dT%H_00}_paired.csv'

    print(f'=== FNV3 CSV download ({tag}) ===')
    print(f'  config init: {init_str}  ->  cycle: {init:%Y-%m-%d %HZ}')
    print(f'  url: {url}')
    print(f'  dst: {dst}')

    if dst.exists() and dst.stat().st_size > 1000 and not args.force:
        print('  [skip] already exists (use --force to re-download)')
        return
    if args.dry_run:
        print('  [dry-run] no fetch performed')
        return
    ok, msg = curl_one(url, dst)
    print(f'  [{"OK" if ok else "FAIL"}] {msg}')
    if ok:
        print(f'\nOverlay it with:')
        print(f'  python run.py --ensemble --stage plot')
        print(f'  (config google_csv_{tag} / google_id_{tag})')


if __name__ == '__main__':
    main()
