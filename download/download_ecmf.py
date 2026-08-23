#!/usr/bin/env python
"""
download_ecmf.py
Author: Shijie Xiao
Email: sxiao73@gatech.edu
Date: 2025-11-15

Download ECMWF TIGGE TC track data from NCAR GDEX
Downloads data for:
- Hurricane Irma (2017): August-September 2017, Atlantic basin
- Hurricane Maria (2017): September 2017, Atlantic basin

Usage:
    python download_ecmf.py
"""

import sys
import os
from pathlib import Path
from calendar import monthrange
from urllib.request import build_opener
from typing import Optional, Sequence

# Base directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "tracks" / "tigge" / "ecmwf"

# Basin mapping for different storms
BASIN_MAP = {
    (2017, 8): "al",  # Hurricane Irma genesis (late August)
    (2017, 9): "al",  # Hurricane Irma and Maria - Atlantic
}

# Base URL for NCAR GDEX
BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d330003/ecmf"


def _normalize_days(year: int, month: int, days: Optional[Sequence[int]]) -> list:
    """Return valid days for the month, filtered if provided."""
    _, num_days = monthrange(year, month)
    if not days:
        return list(range(1, num_days + 1))
    valid = [day for day in days if 1 <= day <= num_days]
    return sorted(set(valid))


def generate_filelist(year: int,
                      month: int,
                      days: Optional[Sequence[int]] = None) -> list:
    """
    Generate list of ECMWF TIGGE file URLs for a given year and month
    
    Args:
        year: Year (4 digits)
        month: Month (1-12)
    
    Returns:
        List of file URLs
    """
    filelist = []
    day_list = _normalize_days(year, month, days)
    
    for day in day_list:
        # Two times per day: 00:00 and 12:00 UTC
        for hour in [0, 12]:
            url = f"{BASE_URL}/{year}/{year}{month:02d}{day:02d}/z_tigge_c_ecmf_{year}{month:02d}{day:02d}{hour:02d}0000_ifs_glob_prod_all_glo.xml"
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
    
    print(f"\nDownloading to: {basin_dir}")
    print(f"Total files: {len(filelist)}\n")
    
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
                failed_count += 1
        except Exception as e:
            sys.stdout.write(f"failed: {e}\n")
            failed_count += 1
    
    print(f"\nSummary:")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Failed: {failed_count}")
    
    return downloaded_count


def main():
    """Main download function"""
    print("=" * 70)
    print("ECMWF TIGGE TC Track Data Download")
    print("=" * 70)
    
    # Download configurations
    downloads = [
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
        
        # Download files
        downloaded = download_files(filelist, DATA_DIR, basin)
        total_downloaded += downloaded
    
    print(f"\n{'=' * 70}")
    print(f"Download complete!")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Data saved to: {DATA_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()