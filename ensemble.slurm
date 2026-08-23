#!/bin/bash
#SBATCH -J fhlo_ens
#SBATCH -A m5011
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -c 64
#SBATCH -t 120
#SBATCH -o /pscratch/sd/s/sixao74/Deepmind/FHLO/logs/ens_%j.out
#SBATCH -e /pscratch/sd/s/sixao74/Deepmind/FHLO/logs/ens_%j.err

# Full 1000-member GEFS ensemble forecast (eprep + FAST ODE).
# Usage: sbatch ensemble.slurm [extra run.py --ensemble args]
# (default args below run the beryl 2024-06-28 12Z GEFS self-consistent case)

module load conda
source activate fast_ml

cd /pscratch/sd/s/sixao74/Deepmind/FHLO
mkdir -p logs

# 1 BLAS thread per worker process (64 workers on 64 cores)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export FHLO_TQDM=0

ARGS=${@:---ensemble \
  --synth-nc tracks/processed/beryl_2024/2024062812_gefs/synthetic_tracks_1000members.nc \
  --members 1000 --assign gefs --workers 64 \
  --out-root data/ensemble/beryl_gefs_1000}

echo "[$(date)] FHLO ensemble start: run.py $ARGS"
srun python run.py $ARGS
echo "[$(date)] FHLO ensemble end"
