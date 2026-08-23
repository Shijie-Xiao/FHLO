#!/bin/bash
#SBATCH -J fhlo_run
#SBATCH -A m5011
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -c 8
#SBATCH -t 120
#SBATCH -o /pscratch/sd/s/sixao74/Deepmind/FHLO/logs/run_%j.out
#SBATCH -e /pscratch/sd/s/sixao74/Deepmind/FHLO/logs/run_%j.err

# Full pipeline (prep with vortex surgery + FAST ODE) for storms in config.txt.
# Usage: sbatch run.slurm [extra run.py args]

module load conda
source activate fast_ml

cd /pscratch/sd/s/sixao74/Deepmind/FHLO
mkdir -p logs

echo "[$(date)] FHLO pipeline start"
python run.py "$@"
echo "[$(date)] FHLO pipeline end"
