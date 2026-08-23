"""
download_era5_irma.py
Download ERA5 6-hourly reanalysis data for Hurricane Irma (2017)
- Time range: 2017-08-30 to 2017-09-14 (with buffer)
- Spatial domain: North Atlantic basin region (NA basin)
- Variables: SST, surface pressure, T/Q/U/V at pressure levels
- Grid: 0.25° x 0.25°
- Output: ERA5/irma/ directory structure

Usage:
    python download_era5_irma.py

Dependencies:
    pip install cdsapi xarray netCDF4
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import cdsapi
from basin_regions import get_basin_area, get_basin_info

# --------------------
# Configuration for Irma
# --------------------
# Time range (from track analysis)
START_DATE = datetime(2017, 8, 30, 0, 0)  # Include buffer before TIGGE init
END_DATE = datetime(2017, 9, 14, 23, 59)   # Include buffer after track end

# Spatial domain: Use North Atlantic basin region
BASIN_CODE = 'NA'  # North Atlantic basin
basin_info = get_basin_info(BASIN_CODE)
AREA = basin_info['area']  # [N, W, S, E] in 0-360°E format for CDS API

# Output directory
BASE_DIR = Path(__file__).parent / "ERA5" / "irma"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Grid resolution
GRID = "0.25/0.25"

# Pressure levels for different variables
# Full set for temperature & specific humidity
PL_LEVELS_TQ = [
    "70", "100", "125", "150", "175", "200", "225", "250", "300", "350", "400", "450",
    "500", "550", "600", "650", "700", "750", "775", "800", "825", "850", "875", "900",
    "925", "950", "975", "1000"
]

# Wind levels (250 and 850 hPa as per original paper)
PL_LEVELS_UV = ["250", "850"]

# 6-hourly times
TIMES6 = ["00:00", "06:00", "12:00", "18:00"]

# --------------------
# Helper functions
# --------------------
def get_date_range(start_date, end_date):
    """Generate list of dates between start and end (inclusive)"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def get_months_days(dates):
    """Extract unique months and days from date list"""
    months = sorted(set(d.strftime("%m") for d in dates))
    days = sorted(set(d.strftime("%d") for d in dates))
    return months, days

def request_file(fn: Path, req_type: str, req: dict):
    """Request file from CDS API"""
    fn.parent.mkdir(parents=True, exist_ok=True)
    if fn.exists():
        print(f"[FOUND] {fn.name}")
        return
    print(f"[CDS] Requesting {fn.name} ...")
    try:
        c = cdsapi.Client()
        c.retrieve(req_type, req, str(fn))
        print(f"[OK] Downloaded {fn.name}")
    except Exception as e:
        print(f"[ERROR] Failed to download {fn.name}: {e}")
        raise

# --------------------
# Download functions
# --------------------
def download_single_level(output_dir: Path, year: int, var: str, short: str, dates):
    """Download single-level variable (SST or surface pressure)"""
    months, days_list = get_months_days(dates)
    
    for month in months:
        # Filter days for this month
        month_dates = [d for d in dates if d.strftime("%m") == month]
        month_days = sorted(set(d.strftime("%d") for d in month_dates))
        
        out_file = output_dir / f"era5_{short}_6h_{year}{month}.nc"
        
        req = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var,
            "year": str(year),
            "month": month,
            "day": month_days,
            "time": TIMES6,
            "area": AREA,  # Spatial subset
            "grid": GRID,
        }
        request_file(out_file, "reanalysis-era5-single-levels", req)

def download_pressure_level(output_dir: Path, year: int, var: str, short: str, levels, dates):
    """Download pressure-level variable (T, Q, U, or V)"""
    months, days_list = get_months_days(dates)
    
    for month in months:
        # Filter days for this month
        month_dates = [d for d in dates if d.strftime("%m") == month]
        month_days = sorted(set(d.strftime("%d") for d in month_dates))
        
        out_file = output_dir / f"era5_{short}_6h_{year}{month}.nc"
        
        req = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var,
            "pressure_level": levels,
            "year": str(year),
            "month": month,
            "day": month_days,
            "time": TIMES6,
            "area": AREA,  # Spatial subset
            "grid": GRID,
        }
        request_file(out_file, "reanalysis-era5-pressure-levels", req)

# --------------------
# Main
# --------------------
def main():
    print("="*70)
    print("Downloading ERA5 data for Hurricane Irma (2017)")
    print("="*70)
    print(f"Time range: {START_DATE} to {END_DATE}")
    print(f"Basin: {basin_info['name']} ({BASIN_CODE})")
    print(f"Spatial domain: {basin_info['description']}")
    print(f"Area bounds: {AREA} [N, W, S, E] in 0-360°E format")
    print(f"  Latitude: {AREA[2]}°N to {AREA[0]}°N")
    print(f"  Longitude: {AREA[1]}°E to {AREA[3]}°E (0-360° format)")
    print(f"Output directory: {BASE_DIR}")
    print("="*70)
    
    # Generate date list
    dates = get_date_range(START_DATE, END_DATE)
    year = 2017
    
    # Create subdirectories
    single_dir = BASE_DIR / "single"
    press_dir = BASE_DIR / "press"
    single_dir.mkdir(parents=True, exist_ok=True)
    press_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/6] Downloading sea surface temperature (SST)...")
    download_single_level(single_dir, year, "sea_surface_temperature", "sst", dates)
    
    print("\n[2/6] Downloading surface pressure...")
    download_single_level(single_dir, year, "surface_pressure", "sp", dates)
    
    print("\n[3/6] Downloading temperature (pressure levels)...")
    download_pressure_level(press_dir, year, "temperature", "t", PL_LEVELS_TQ, dates)
    
    print("\n[4/6] Downloading specific humidity (pressure levels)...")
    download_pressure_level(press_dir, year, "specific_humidity", "q", PL_LEVELS_TQ, dates)
    
    print("\n[5/6] Downloading u-component of wind (pressure levels)...")
    download_pressure_level(press_dir, year, "u_component_of_wind", "u", PL_LEVELS_UV, dates)
    
    print("\n[6/6] Downloading v-component of wind (pressure levels)...")
    download_pressure_level(press_dir, year, "v_component_of_wind", "v", PL_LEVELS_UV, dates)
    
    print("\n" + "="*70)
    print("[COMPLETE] All downloads finished!")
    print(f"Output directory: {BASE_DIR}")
    print("\nSample output files:")
    print(f"  {single_dir / 'era5_sst_6h_201708.nc'}")
    print(f"  {single_dir / 'era5_sp_6h_201708.nc'}")
    print(f"  {press_dir / 'era5_t_6h_201708.nc'}")
    print(f"  {press_dir / 'era5_q_6h_201708.nc'}")
    print(f"  {press_dir / 'era5_u_6h_201708.nc'}")
    print(f"  {press_dir / 'era5_v_6h_201708.nc'}")
    print("="*70)

if __name__ == "__main__":
    main()

