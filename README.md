# FHLO — Forecasts of Hurricanes Using Large-Ensemble Outputs

A self-contained implementation of **FHLO** (Lin, Emanuel & Vigh 2020,
*Wea. Forecasting*, 35, 1713–1731), a physics-based ensemble hurricane
intensity forecasting system. This repository reproduces the full
pipeline from operational forecast fields to probabilistic intensity and
wind-exceedance products, with all constants calibrated to
[linjonathan/tropical_cyclone_risk](https://github.com/linjonathan/tropical_cyclone_risk).

## What is reproduced

| Component | Reference |
|---|---|
| Synthetic track ensemble (Markov resampling of TIGGE parents) | Lin et al. 2020, Sec. 3a–b |
| Environmental field extraction + vortex removal | Lin et al. 2020, Sec. 3c |
| FAST intensity ODE (Emanuel 2012 coupled air–sea) | Lin et al. 2020, Sec. 3d |
| χ calibration, (Ck, h_bl) constants, ocean feedback α | Lin et al. 2020 + Lin namelist |
| ODE initialization (cold / obs-replay / KL perturbation) | Lin et al. 2020, Sec. 2c + appendix B |
| Surface wind field (CLE15 + shape-k + asymmetry) | Lin et al. 2020, Sec. 3d–e |
| 34/50/64-kt wind-exceedance probability | DeMaria 2009 |

## Physics chain

```
TIGGE parents ──Markov──> 1000 synthetic tracks
                                │
GEFS / ERA5 fields ──vortex removal──> env winds, χ, s, shear, Vp
                                │
                     FAST ODE:  dV/dt = (Vp−V)·|χ_eff·s| / (h_bl·V)
                                │
                     CLE15 wind field -> 34/50/64-kt probability maps
```

- **FAST ODE** (Emanuel 2012): intensity evolves toward potential
  intensity Vp under ventilation χ_eff·s; Heun integration, 4 sub-steps/h.
- **Dynamic ocean feedback**: α recomputed from current V every 15-min
  sub-step using monthly MLD/strat climatology + bathymetry; γ = ε + ακ.
- **Wind model**: CLE15 axisymmetric profile (Chavas, Lin & Emanuel 2015,
  official code ported) + shape parameter k + translation/shear asymmetry;
  (r0, k) fitted to IBTrACS quadrant wind radii at initialization.

## Quick start

```bash
conda env create -f environment.yml && conda activate fast_ml

# full pipeline (prep -> ode -> plot -> wind probability)
python run.py --ensemble --stage eprep,ode,plot,wind

# re-run a single stage on existing outputs
python run.py --ensemble --stage wind        # wind probability only
python run.py --ensemble --stage plot        # intensity plots only
```

All paths and experiment knobs resolve from `config.txt`. Output lands in
`data/ensemble/{storm}_{env}/`. On Slurm: `sbatch ens_flossie_gefs_default.slurm`.

## Pipeline stages (`run.py --stage`)

| stage | does | output |
|---|---|---|
| `ibtracs` | download best-track CSVs | `data/ibtracs/` |
| `eprep` | per-member env extraction (vortex removal, scalars) | `{STORM}_M{NNN}/_dataset.pkl` |
| `ode` | FAST intensity ODE per member | `ensemble_fast.nc`, `ensemble_summary.nc` |
| `plot` | intensity spaghetti + mean plots | `ensemble_fast.png/svg` |
| `wind` | CLE15 wind field -> exceedance probability | `wind_prob.nc`, `wind_prob.png/svg` |

## Configuration (`config.txt`)

| key | meaning |
|---|---|
| `storms`, `basins`, `year_start/end` | storm selection |
| `gefs_dir_{tag}` / `gefs_init_{tag}` | local GEFS forecast dir + init cycle |
| `synth_gefs_nc_{tag}` / `synth_ecmwf_nc_{tag}` | synthetic-track NC |
| `era5_dir` | local ERA5 analysis crop |
| `env` | environment source: `gefs` (forecast) or `era5` (analysis) |
| `vortex_mode` | `annulus` (200–800 km mean) or `surgery` (global fields only) |
| `ode_mode` | `cold` / `fhlo` (obs replay + KL + F) / `free` (replay + KL) |
| `replay_hours`, `kl_perturb` | obs-replay window, appendix-B KL perturbation |
| `vp_comp_gefs` / `vp_comp_era5` | Vp bias compensation (1.1 / 1.0) |
| `n_workers` | process-pool size |

## Repository layout

```
run.py          pipeline orchestrator (stages: eprep/ode/plot/wind)
config.txt      all paths + experiment knobs
prep/           IBTrACS download, env extraction (dataset.pkl)
physics/        run_fast_reference.py — FAST ODE
tracks/         TIGGE -> Markov -> synthetic tracks
ensemble/       GEFS/ERA5 adapters, ensemble plotting
wind/           CLE15 port + shape-k + asymmetry + (r0,k) fit + probability
common/         thermo tables, vortex-inversion library
data/           bundled local data (ship separately, ~47 GB)
```

## Reproducibility

- Each ensemble run records its exact configuration in `run_config.txt`.
- Per-member outputs are resumable; existing pkls are skipped unless
  `--overwrite`.
- KL perturbation seeds are deterministic (`100000 + member id`).
- All data used by `run.py` lives under the local `data/` tree; no
  external paths or network access needed after setup.
