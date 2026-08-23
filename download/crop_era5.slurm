#!/bin/bash
#SBATCH -J fhlo_crop
#SBATCH -A m5011
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -c 16
#SBATCH -t 120
#SBATCH -o /pscratch/sd/s/sixao74/Deepmind/FHLO/logs/crop_%j.out
#SBATCH -e /pscratch/sd/s/sixao74/Deepmind/FHLO/logs/crop_%j.err

# Crop ERA5 (+OISST) by date, full spatial domain, for one storm.
# Usage: sbatch crop_era5.slurm [BASIN YEAR STORM]

module load conda
source activate fast_ml

cd /pscratch/sd/s/sixao74/Deepmind/FHLO
mkdir -p logs data/era5 data/oisst

BASIN=${1:-NA}
YEAR=${2:-2024}
STORM=${3:-2024181N09320_BERYL}

echo "[$(date)] Cropping ERA5/OISST for $BASIN $YEAR $STORM"
python download/crop_beryl_sample.py "$BASIN" "$YEAR" "$STORM" --workers 16
echo "[$(date)] Crop finished"
