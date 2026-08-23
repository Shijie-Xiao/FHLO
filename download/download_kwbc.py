#!/usr/bin/env python
"""
download_kwbc.py
Author: Shijie Xiao
Email: sxiao73@gatech.edu
Date: 2025-11-15

Download GEFS (kwbc) TIGGE TC track data from NCAR GDEX
Downloads data for:
- Hurricane Irma (2017): August-September 2017, Atlantic basin
- Hurricane Maria (2017): September 2017, Atlantic basin

File types:
- GEFS_glob_prod_esttr_glo.xml: GEFS ensemble tracks (primary)
- GFS_glob_prod_sttr_glo.xml: GFS single track
- CENS_glob_prod_esttr_glo.xml: Ensemble tracks (control)
- CMC_glob_prod_sttr_glo.xml: CMC single track

Usage:
    python download_kwbc.py
"""

import sys
import os
from pathlib import Path
from calendar import monthrange
from urllib.request import build_opener
from typing import Iterable, List, Optional, Sequence, Tuple

# Base directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "tracks" / "tigge" / "gefs"

# Basin mapping for different storms
BASIN_MAP = {
    (2017, 8): "al",  # Hurricane Irma genesis (late August)
    (2017, 9): "al",  # Hurricane Irma and Maria - Atlantic
}

# Base URL for NCAR GDEX
BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d330003/kwbc"

# File types and their suffixes
# (type, suffix) pairs for kwbc files
FILE_TYPES = [
    ("GEFS", "esttr"),   # GEFS ensemble tracks (most important)
    ("GFS", "sttr"),     # GFS single track
    ("CENS", "esttr"),   # Ensemble tracks (control)
    ("CMC", "sttr"),     # CMC single track
]

# Hours per day (4 times: 00:00, 06:00, 12:00, 18:00)
HOURS = [0, 6, 12, 18]


def _normalize_days(year: int, month: int, days: Optional[Sequence[int]]) -> List[int]:
    """Return the list of valid days for a month (filtered if provided)."""
    _, num_days = monthrange(year, month)
    if not days:
        return list(range(1, num_days + 1))
    valid_days = []
    for day in days:
        if 1 <= day <= num_days:
            valid_days.append(day)
    return sorted(set(valid_days))


def generate_filelist(year: int,
                      month: int,
                      days: Optional[Sequence[int]] = None,
                      hours: Optional[Iterable[int]] = None,
                      file_types: Optional[Sequence[Tuple[str, str]]] = None) -> list:
    """
    Generate list of GEFS (kwbc) TIGGE file URLs for a given year and month
    
    Args:
        year: Year (4 digits)
        month: Month (1-12)
    
    Returns:
        List of file URLs
    
    Note: Not all file types may exist for all hours/days.
    Script will try to download all combinations and skip missing ones.
    """
    hour_list = list(hours) if hours is not None else HOURS
    type_list = list(file_types) if file_types is not None else FILE_TYPES
    day_list = _normalize_days(year, month, days)
    
    filelist = []
    for day in day_list:
        for hour in hour_list:
            for file_type, suffix in type_list:
                url = (
                    f"{BASE_URL}/{year}/{year}{month:02d}{day:02d}/"
                    f"z_tigge_c_kwbc_{year}{month:02d}{day:02d}{hour:02d}0000_"
                    f"{file_type}_glob_prod_{suffix}_glo.xml"
                )
                filelist.append(url)
    return filelist


def download_files(filelist: list, output_dir: Path, basin: str) -> int:
    """
    Download files from the filelist to the output directory
    
    Args:
        filelist: List of file URLs to download
        output_dir: Directory to save files
        basin: Basin name (al or ep)
    
    Returns:
        Number of successfully downloaded files
    """
    # Create basin subdirectory
    basin_dir = output_dir / basin
    basin_dir.mkdir(parents=True, exist_ok=True)
    
    opener = build_opener()
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    not_found_count = 0
    
    print(f"\nDownloading to: {basin_dir}")
    print(f"Total file URLs to try: {len(filelist)}\n")
    
    for url in filelist:
        ofile = os.path.basename(url)
        filepath = basin_dir / ofile
        
        # Skip if file already exists and is not empty
        if filepath.exists() and filepath.stat().st_size > 1000:
            skipped_count += 1
            continue
        
        try:
            sys.stdout.write(f"Downloading {ofile} ... ")
            sys.stdout.flush()
            
            infile = opener.open(url)
            outfile = open(filepath, "wb")
            outfile.write(infile.read())
            outfile.close()
            infile.close()
            
            # Verify file was downloaded successfully
            if filepath.stat().st_size > 1000:
                sys.stdout.write("done\n")
                downloaded_count += 1
            else:
                sys.stdout.write("failed (file too small)\n")
                filepath.unlink()  # Delete empty file
                not_found_count += 1
        except Exception as e:
            # Check if it's a 404 (file not found) - this is expected for some combinations
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                sys.stdout.write("not found (skipped)\n")
                not_found_count += 1
            else:
                sys.stdout.write(f"failed: {e}\n")
                failed_count += 1
    
    print(f"\nSummary:")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Not found (expected): {not_found_count}")
    print(f"  Failed (errors): {failed_count}")
    
    return downloaded_count


def main():
    """Main download function"""
    print("=" * 70)
    print("GEFS (kwbc) TIGGE TC Track Data Download")
    print("=" * 70)
    
    # Download configurations
    downloads = [
        # Irma genesis (late August 2017) – only need Aug 30-31
        {"year": 2017, "month": 8, "basin": "al", "storm": "Hurricane Irma (Genesis)", "days": [30, 31]},
        {"year": 2017, "month": 9, "basin": "al", "storm": "Hurricanes Irma and Maria"},
    ]
    
    total_downloaded = 0
    
    for config in downloads:
        year = config["year"]
        month = config["month"]
        basin = config["basin"]
        storm = config["storm"]
        
        print(f"\n{'=' * 70}")
        print(f"{storm} ({year}) - {year}-{month:02d}, Basin: {basin.upper()}")
        print(f"{'=' * 70}")
        
        # Generate file list
        filelist = generate_filelist(year, month, days=config.get("days"))
        day_info = f"days {config['days']}" if config.get("days") else "entire month"
        print(f"Generated {len(filelist)} file URLs for {year}-{month:02d} ({day_info})")
        print(f"Note: Not all combinations exist. Script will skip missing files.")
        
        # Download files
        downloaded = download_files(filelist, DATA_DIR, basin)
        total_downloaded += downloaded
    
    print(f"\n{'=' * 70}")
    print(f"Download complete!")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Data saved to: {DATA_DIR}")
    print(f"\nNote: GEFS_glob_prod_esttr_glo.xml files contain ensemble track data.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()