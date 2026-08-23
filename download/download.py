"""
download.py
Author: Shijie Xiao
Email: sxiao73@gatech.edu
Date: 2025-11-15

Download TC track data for Hurricanes Irma (2017) and Maria (2017)

Data Sources:
1. HURDAT2: https://www.nhc.noaa.gov/data/hurdat/ (Public, no account needed)
   - Best track (post-analysis) data
2. ATCF: https://ftp.nhc.noaa.gov/atcf/ (Public, no account needed)
   - a-deck: Ensemble forecasts from multiple models (including ECMWF/GEFS)
   - b-deck: Best track data
3. TIGGE TC Track Dataset (NCAR GDEX):
   - Preprocessed ensemble TC track forecasts from ECMWF and GEFS
   - Downloaded via download_ecmf.py (ECMWF) and download_kwbc.py (GEFS)
   - Note: ECMWF/GEFS ensemble tracks are also available in ATCF a-deck files

Output paths:
- HURDAT2: data/tracks/hurdat2/{filename}
- ATCF: data/tracks/atcf/{adeck,bdeck}/{year}/{filename}
- TIGGE: data/tracks/tigge/{ecmwf,gefs}/{basin}/{filename}
"""

import urllib.request
import urllib.error
import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "tracks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URLS = {
    "hurdat2": "https://www.nhc.noaa.gov/data/hurdat/",
    "atcf": "https://ftp.nhc.noaa.gov/atcf/",
}

# Latest HURDAT2 files (cumulative format containing all years)
# Atlantic: hurdat2-1851-2024-040425.txt
# East Pacific: hurdat2-nepac-1949-2024-031725.txt
HURDAT2_FILES = {
    "atl": "hurdat2-1851-2024-040425.txt",
    "nepac": "hurdat2-nepac-1949-2024-031725.txt",
}


class TrackDataDownloader:
    """
    Download TC track data from various sources
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.hurdat2_dir = self.data_dir / "hurdat2"
        self.atcf_dir = self.data_dir / "atcf"
        self.tigge_dir = self.data_dir / "tigge"
        
        self.hurdat2_dir.mkdir(exist_ok=True)
        self.atcf_dir.mkdir(exist_ok=True)
        self.tigge_dir.mkdir(exist_ok=True)
    
    def download_hurdat2(self, year: int):
        """
        Download HURDAT2 best track data (cumulative files containing all years)
        
        Args:
            year: Year (4 digits) - used only for filtering/extracting specific year
        
        Output: 
        - data/tracks/hurdat2/hurdat2-1851-2024-040425.txt (Atlantic)
        - data/tracks/hurdat2/hurdat2-nepac-1949-2024-031725.txt (East Pacific)
        """
        print(f"[HURDAT2] Downloading cumulative files...", end=" ")
        
        downloaded = 0
        for basin, filename in HURDAT2_FILES.items():
            url = f"{BASE_URLS['hurdat2']}{filename}"
            filepath = self.hurdat2_dir / filename
            
            if filepath.exists() and filepath.stat().st_size > 1000:
                print(f"{filename} exists", end=" ")
                downloaded += 1
                continue
            
            try:
                urllib.request.urlretrieve(url, filepath)
                if filepath.stat().st_size > 1000:
                    print(f"{filename} ✓", end=" ")
                    downloaded += 1
                else:
                    filepath.unlink()
                    print(f"{filename} ✗", end=" ")
            except Exception as e:
                print(f"{filename} ✗", end=" ")
        
        print()
    
    def download_atcf(self,
                      year: int,
                      basin: str = "al",
                      months: Optional[List[int]] = None):
        """
        Download ATCF a-deck and b-deck data
        
        Args:
            year: Year (4 digits)
            basin: 'al' (Atlantic) or 'ep' (East Pacific)
            months: Optional subset of months to download (1-12). If None,
                    downloads the full year.
        
        Output: 
        - data/tracks/atcf/adeck/{year}/a{basin}{month:02d}{year}.dat
        - data/tracks/atcf/bdeck/{year}/b{basin}{month:02d}{year}.dat
        
        Note: 
        - Filename format: aal/aep (a-deck) or bal/bep (b-deck) + month + year
        - Files from 2018+ may be compressed (.gz)
        """
        month_list = months or list(range(1, 13))
        month_list = sorted(set(m for m in month_list if 1 <= m <= 12))
        month_desc = ",".join(f"{m:02d}" for m in month_list)
        print(f"[ATCF] {basin.upper()} {year} (months:{month_desc})...", end=" ")
        
        # Convert basin code: al -> al, ep -> ep
        basin_prefix = {"al": "aal", "ep": "aep"}.get(basin, f"a{basin}")
        bdeck_prefix = {"al": "bal", "ep": "bep"}.get(basin, f"b{basin}")
        
        adeck_dir = self.atcf_dir / "adeck" / str(year)
        bdeck_dir = self.atcf_dir / "bdeck" / str(year)
        adeck_dir.mkdir(parents=True, exist_ok=True)
        bdeck_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_count = 0
        
        # Download a-deck (monthly files)
        for month in month_list:
            # Try compressed first (for recent years), then uncompressed
            for suffix in [".dat.gz", ".dat"]:
                filename = f"{basin_prefix}{month:02d}{year:04d}{suffix}"
                url = f"{BASE_URLS['atcf']}archive/{year}/{filename}"
                filepath = adeck_dir / filename
                final_filepath = adeck_dir / filename.replace(".gz", "")
                
                if final_filepath.exists():
                    downloaded_count += 1
                    break
                
                if filepath.exists():
                    continue
                
                try:
                    urllib.request.urlretrieve(url, filepath)
                    if filepath.stat().st_size > 0:
                        # Decompress if needed
                        if suffix == ".gz":
                            with gzip.open(filepath, 'rb') as f_in:
                                with open(final_filepath, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            filepath.unlink()
                            downloaded_count += 1
                        else:
                            downloaded_count += 1
                        break
                except urllib.error.HTTPError:
                    if suffix == ".gz":
                        continue  # Try uncompressed
                    pass
                except Exception:
                    pass
        
        # Download b-deck (monthly files in archive directory, same as a-deck)
        # Note: Historical years are in archive/{year}/, current year may be in btk/
        for month in month_list:
            # Try archive directory first (for historical years)
            for suffix in [".dat.gz", ".dat"]:
                filename = f"{bdeck_prefix}{month:02d}{year:04d}{suffix}"
                url = f"{BASE_URLS['atcf']}archive/{year}/{filename}"
                filepath = bdeck_dir / filename
                final_filepath = bdeck_dir / filename.replace(".gz", "")
                
                if final_filepath.exists():
                    downloaded_count += 1
                    break
                
                if filepath.exists():
                    continue
                
                try:
                    urllib.request.urlretrieve(url, filepath)
                    if filepath.stat().st_size > 0:
                        # Decompress if needed
                        if suffix == ".gz":
                            with gzip.open(filepath, 'rb') as f_in:
                                with open(final_filepath, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            filepath.unlink()
                            downloaded_count += 1
                        else:
                            downloaded_count += 1
                        break
                except urllib.error.HTTPError:
                    if suffix == ".gz":
                        continue  # Try uncompressed
                    # If archive directory fails, try btk directory (for current year)
                    try:
                        btk_filename = f"{bdeck_prefix}{month:02d}{year:04d}.dat"
                        btk_url = f"{BASE_URLS['atcf']}btk/{btk_filename}"
                        btk_filepath = bdeck_dir / btk_filename
                        if not btk_filepath.exists():
                            urllib.request.urlretrieve(btk_url, btk_filepath)
                            if btk_filepath.stat().st_size > 0:
                                downloaded_count += 1
                                break
                    except:
                        pass
                    break
                except Exception:
                    pass
        
        if downloaded_count > 0:
            print(f"✓ ({downloaded_count} files)")
        else:
            print("✗")
    
    def download_tigge_via_scripts(self, download_ecmwf: bool = True, download_gefs: bool = True):
        """
        Download TIGGE data by calling download_ecmf.py and download_kwbc.py
        
        Args:
            download_ecmwf: If True, call download_ecmf.py
            download_gefs: If True, call download_kwbc.py
        """
        base_dir = Path(__file__).parent
        
        if download_ecmwf:
            print("\n[TIGGE] Calling download_ecmf.py for ECMWF data...")
            ecmf_script = base_dir / "download_ecmf.py"
            if ecmf_script.exists():
                try:
                    subprocess.run([sys.executable, str(ecmf_script)], check=False)
                except Exception as e:
                    print(f"  ⚠ Error running download_ecmf.py: {e}")
            else:
                print(f"  ⚠ Script not found: {ecmf_script}")
        
        if download_gefs:
            print("\n[TIGGE] Calling download_kwbc.py for GEFS data...")
            kwbc_script = base_dir / "download_kwbc.py"
            if kwbc_script.exists():
                try:
                    subprocess.run([sys.executable, str(kwbc_script)], check=False)
                except Exception as e:
                    print(f"  ⚠ Error running download_kwbc.py: {e}")
            else:
                print(f"  ⚠ Script not found: {kwbc_script}")
    
    def download_irma_data(self, download_tigge: bool = True):
        """
        Download all data for Hurricane Irma (2017)
        
        Args:
            download_tigge: If True, call download_ecmf.py and download_kwbc.py for TIGGE data
        """
        print("\n[Hurricane Irma 2017]")
        self.download_hurdat2(2017)
        # Only grab August + September to cover genesis and early evolution
        self.download_atcf(2017, basin="al", months=[8, 9])
        
        if download_tigge:
            self.download_tigge_via_scripts(download_ecmwf=True, download_gefs=True)
    
    def download_maria_data(self, download_tigge: bool = True):
        """
        Download all data for Hurricane Maria (2017)
        
        Args:
            download_tigge: If True, call download_ecmf.py and download_kwbc.py for TIGGE data
        """
        print("\n[Hurricane Maria 2017]")
        self.download_hurdat2(2017)
        # Download September data for Maria (genesis and evolution)
        self.download_atcf(2017, basin="al", months=[9])
        
        if download_tigge:
            self.download_tigge_via_scripts(download_ecmwf=True, download_gefs=True)


def main():
    """Main download function"""
    downloader = TrackDataDownloader()
    downloader.download_irma_data()
    downloader.download_maria_data()
    print(f"\n[Complete] Data saved to: {DATA_DIR}")
    print("\nNote: HURDAT2 files are cumulative (contain all years).")
    print("      Use preprocess_tracks.py to extract specific storms/years.")


if __name__ == "__main__":
    main()