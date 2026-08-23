"""FHLO synthetic-track pipeline.

  read_files   – (storm, cycle) -> parent ensemble raw.pkl (ECMWF/GEFS)
  build_pairs  – raw -> 6h velocity pairs + 75% survival horizon
  train_markov – pairs -> per-step k=1 Gaussian conditional fits
  sample_tracks– params -> 1000-member synthetic tracks NC
  plot_tracks  – optional per-case track figure
  batch_generate – single entry point driving all stages
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
