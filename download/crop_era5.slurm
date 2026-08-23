#!/bin/bash
#SBATCH -J fhlo_crop
#SBATCH -A m5011
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -c 16
#SBATCH -t 120
#SBATCH -o logs/crop_%j.out
#SBATCH -e logs/crop_%j.err

# Crop ERA5 (+OISST) by date, full spatial domain, for one storm.
# Usage: sbatch crop_era5.slurm [BASIN YEAR STORM]
# NOTE: submit from the repo root (paths are relative to the checkout dir).

module load conda
source activate fast_ml

cd "$(dirname "$0")/.."
mkdir -p logs data/era5 data/oisst

BASIN=${1:-NA}
YEAR=${2:-2024}
STORM=${3:-2024181N09320_BERYL}

echo "[$(date)] Cropping ERA5/OISST for $BASIN $YEAR $STORM"
python download/crop_beryl_sample.py "$BASIN" "$YEAR" "$STORM" --workers 16
echo "[$(date)] Crop finished"
