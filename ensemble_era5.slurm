#!/bin/bash
#SBATCH -J fhlo_era5
#SBATCH -A m5011
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -c 64
#SBATCH -t 120
#SBATCH -o logs/era5_%j.out
#SBATCH -e logs/era5_%j.err

# 1000-member ensemble, analysis mode: ECMWF-sampled tracks x ERA5 env.
# Usage: sbatch ensemble_era5.slurm [extra run.py --ensemble args]
# NOTE: submit from the repo root (paths are relative to the checkout dir).

module load conda
source activate fast_ml

cd "$(dirname "$0")"
mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export FHLO_TQDM=0

ARGS=${@:---ensemble --env era5 \
  --synth-nc tracks/processed/beryl_2024/2024062900/synthetic_tracks_1000members.nc \
  --members 1000 --workers 64 \
  --out-root data/ensemble/beryl_era5_1000}

echo "[$(date)] FHLO ERA5 ensemble start: run.py $ARGS"
srun python run.py $ARGS
echo "[$(date)] FHLO ERA5 ensemble end"
