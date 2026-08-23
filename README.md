# FHLO — Fast Hurricane Large-ensemble ODE

A physics-based hurricane intensity forecasting framework that strictly follows
Lin, Emanuel & Vigh (2020), "Forecasts of Hurricanes Using Large-Ensemble
Outputs" (Wea. Forecasting, 35, 1713–1731), with the χ calibration and
(Ck, h_bl) constants aligned to [linjonathan/tropical_cyclone_risk](https://github.com/linjonathan/tropical_cyclone_risk).

Three pipelines share one codebase:

1. **Deterministic reference** — IBTrACS best track → 1 h interpolation →
   ERA5/OISST environment extraction (vortex surgery) → FAST ODE
   (`run.py`, one command).
2. **Synthetic track ensembles** — parent ensemble tracks (ECMWF TIGGE XML /
   GEFS GRIB2 vortex tracking) → Markov chain on translational velocity →
   1000-member sampling (`tracks/batch_generate.py`).
3. **Full 1000-member ensemble forecast** — synthetic tracks × GEFS forecast
   environment fields → per-member FAST ODE with FHLO initialization
   (48 h replay + KL intensity perturbation) (`ensemble/`).

---

## Repository layout

```
FHLO/
├── run.py / run.slurm / config.txt      # deterministic pipeline entry + config
├── prep/                                # data preparation
│   ├── IBtracs_datasets.py              #   IBTrACS download -> 6h best-track CSV
│   ├── prepare_complete_training_data.py#   1h interp + env extraction -> dataset.pkl
│   └── prepare_ensemble_storm.py        #   synthetic-track NC loaders + member splice
├── physics/                             # FAST model
│   ├── run_fast_reference.py            #   single-track ODE (48h init + F decay)
│   ├── run_fast_physics_torch.py        #   batched/GPU version
│   ├── Fast.py / env.py / track.py      #   ODE core, env providers, track providers
│   └── constants.py / utils.py
├── tracks/                              # synthetic-track module (see tracks/README.md)
│   ├── batch_generate.py                #   THE entry: read->pairs->fit->sample->plot
│   ├── read_files.py / build_pairs.py   #   TIGGE XML / GEFS -> 6h velocity pairs
│   ├── train_markov.py / sample_tracks.py
│   └── config.py / plot_tracks.py / gefs_tracks.py
├── ensemble/                            # 1000-member forecast pipeline
│   ├── gefs_ens_adapter.py              #   GEFS GRIB2 -> ERA5-equivalent fields
│   ├── run_prep_gefs_ens1000.py         #   (track × GEFS member) -> dataset.pkl
│   ├── run_prep_div1000.py              #   (synthetic track × ERA5) -> dataset.pkl
│   ├── predict_chi_s_div1000.py         #   ML chi/s extraction (optional stage)
│   └── run_ode_from_chis.py             #   per-member FAST ODE -> ensemble NC
├── common/                              # shared libraries
│   ├── thermo_table/                    #   entropy-table thermodynamics (χ)
│   ├── vortex_inversion/                #   vortex surgery (pyamg + pyshtools)
│   └── util/                            #   spherical geometry, basins, constants
├── download/                            # data download & cropping
├── precalc_data/                        # static fields (Cd, land, MLD, strat, bathy)
└── data/                                # local data (gitignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate fast_ml
```

Key dependencies: `pyamg` + `pyshtools` (vortex surgery), `xarray`/`netcdf4`,
`scipy`, `cfgrib` (GEFS GRIB2), `tcpyPI` (potential intensity, optional).

---

## Pipeline 1 — Deterministic reference from IBTrACS (`run.py`)

```
IBTrACS online CSV ──prep/IBtracs_datasets.py──▶ data/ibtracs/{basin}/{year}/{STORM}/track_intensity_6h.csv
                                                        │
        ERA5 (data/era5)  OISST (data/oisst)            ▼
              └──────────┬─────────────▶ prep/prepare_complete_training_data.py
                                        (6h→1h cubic spline; vortex surgery on
                                         25° domain; χ from 600hPa annulus 200–800km;
                                         tcpyPI v_pot; scalars α/β/γ/vp)
                                                        │
                                                        ▼
                                     data/ibtracs/{basin}/{year}/{STORM}/{STORM}_dataset.pkl
                                                        │
                                                        ▼
                                        physics/run_fast_reference.py
                                        (48h init: V tracks Vtarget, F accumulated;
                                         forecast: F·exp[-(lead/24h)²])
                                                        │
                                                        ▼
                                     fast_reference.csv / fast_reference.png
```

```bash
# 0) local demo data (time-cropped, full spatial domain, ~90 s)
sbatch download/crop_era5.slurm NA 2024 2024181N09320_BERYL

# 1) best track (skip if data/ibtracs already populated)
python prep/IBtracs_datasets.py

# 2) full pipeline (prep + ODE)
python run.py --storms 2024181N09320_BERYL          # foreground
sbatch run.slurm --storms 2024181N09320_BERYL       # Slurm (premium QOS)

# variants
python run.py --sst OISST                # SST source override
python run.py --stage prep,ode           # skip IBTrACS download
python run.py --overwrite --list --workers N
```

`run.py` stages: `ibtracs` (download) → `prep` (pkl) → `ode` (csv/png).
Storms run in a `ProcessPoolExecutor`; within a storm the time loop is
sequential (vortex-surgery caches reuse across steps). ~1.5–2 s per 1 h step;
Beryl (313 steps) ≈ 8 min single-process.

`config.txt` fields: `basins`, `year_start/year_end`, `storms`, `sst_source`
(ERA5|OISST), `env_source` (ERA5|GEFS), `track_source` (ECMWF|GEFS),
`n_workers`, and all data paths — every path is overridable in config.txt
(precedence: CLI flag > env var > config.txt > built-in default):

| key | used by | default |
|---|---|---|
| `output_dir` | best-track root (`{basin}/{year}/{STORM}/`) | `data/ibtracs` |
| `era5_dir` | local ERA5 crop read by prep | `data/era5` |
| `oisst_dir` | local OISST crop read by prep | `data/oisst` |
| `era5_root` | CFS ERA5 archive (crop scripts, adapter) | CFS path |
| `oisst_root` | CFS OISST archive (crop scripts) | CFS path |
| `gefs_root` | GEFS GRIB2 archive (tracks, adapter, ens prep) | CFS path |
| `ecmwf_root` | TIGGE ECMWF XML archive (tracks) | CFS path |
| `gefs_cache_dir` | GEFS cfgrib→NetCDF cache (optional) | `data/gefs_nc_cache` |
| `gefs_case_dir` | specific GEFS case for `run_prep_gefs_ens1000.py` (optional) | `{gefs_root}/2025_ERIN_NA` |

`{STORM}_dataset.pkl` keys:

| key | shape | content |
|---|---|---|
| `spatial_3d` | (1, T, 5, 7, 72, 72) | T/Q/U/V/Z @ 7 levels, 72×72 window |
| `spatial_2d` | (1, T, 2, 72, 72) | SSTK, MSL |
| `scalars` | (1, T, 4) | alpha, beta, gamma, v_pot |
| `chi_ref` / `s_ref` / `xs_ref` | (1, T, 1) | saturation deficit, shear, product |
| `env_wnds` | (1, T, 4) | u/v @ 250/850 hPa (after vortex surgery) |
| `v_gt` / `lats` / `lons` / `times` | (T,) | observed intensity, position, time |
| `cd_ref` / `blh_ref` / `utran` / `vtran` | (1, T, 1) | drag, BL height, translation |

Beryl 2024 reference: MAE ≈ 30 kts, peak ~116 vs 145 kts observed
(after the Lin χ/Ck calibration; the previous tuned combination gave 25.8).

---

## Pipeline 2 — Synthetic track sampling from parent ensembles (`tracks/`)

```
ECMWF TIGGE XML (per cycle)   or   GEFS pgrb2a GRIB2 vortex tracking
        │                                   │
        └───────────┬───────────────────────┘
                    ▼
     tracks/read_files.py   (per storm & init cycle; lon-sign normalization)
                    │  raw.pkl — parent member tracks
                    ▼
     tracks/build_pairs.py  (6h velocities; u0 = 2u1−u2 backward extrapolation;
                    │        75% survival horizon caps the lead times)
                    │  pairs_6h.pkl
                    ▼
     tracks/train_markov.py (per-lead-time k=1 conditional Gaussian)
                    │  markov_params_6h.pkl
                    ▼
     tracks/sample_tracks.py (1000 members, horizon-capped)
                    │  synthetic_tracks_1000members.nc
                    ▼
     tracks/plot_tracks.py --plot (parents + synthetics + best track)
```

```bash
# one storm, default init = first 00/12Z cycle at/after IBTrACS genesis
python tracks/batch_generate.py --storms 2024181N09320_BERYL

# paper Irma case: explicit init cycle + plots
python tracks/batch_generate.py --storms 2017242N16333_IRMA \
    --init 2017242N16333_IRMA:2017090500 --plot

# every 00/12Z cycle over the lifetime (paper evaluation mode)
python tracks/batch_generate.py --storms 2024181N09320_BERYL --cycles all

# batch many storms / re-run tail stages only
python tracks/batch_generate.py --storms A,B,C --plot
python tracks/batch_generate.py --stage sample,plot --storms A
```

Output layout:

```
tracks/processed/{storm_lowercase}_{year}/{YYYYMMDDHH}/
    raw.pkl                        # parent ensemble member tracks
    pairs_6h.pkl                   # velocity pairs + survival horizon
    markov_params_6h.pkl           # per-lead-time conditional Gaussians
    synthetic_tracks_1000members.nc# sampled tracks (downstream interface)
    tracks.png                     # when --plot
```

The NC (`lon/lat/u/v (track, time)`, attrs `init_time`, `dt_hours`) is the
exact interface consumed by `prep/prepare_ensemble_storm._load_synthetic`.
Details and paper-conformance table: `tracks/README.md`.

---

## Pipeline 3 — Full 1000-member ensemble forecast (`ensemble/`)

Two environment backends, same downstream ODE:

**(a) GEFS forecast fields** (`run_prep_gefs_ens1000.py`) — the forecast
configuration: each synthetic track is paired with a GEFS ensemble member
(c00, p01–p30) and the environment along the track is read from that member's
GRIB2 forecast via `gefs_ens_adapter.py` (0.5°→0.25° regrid, pgrb2a+b level
merge; BLH absent → 1400 m fallback).

**(b) ERA5 analysis fields** (`run_prep_div1000.py`) — fully-divergent mode:
`init_time = reference_time`, no best-track prepend, members diverge from h=0.

```
synthetic_tracks_1000members.nc ──┐
 (Pipeline 2 output)              │  prep/prepare_ensemble_storm.py
 best-track dataset.pkl ──────────┤  (hourly splice; vmax from best track)
                                  ▼
                    {OUT_ROOT}/{HID}_M{NNN}/{HID}_M{NNN}_dataset.pkl   ×1000
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
   ensemble/predict_chi_s_div1000.py        ensemble/run_ode_from_chis.py
   (ML TwoStream χ/S extraction;             (FAST ODE per member:
    optional — FAST path reads                48h replay + KL(n=10) intensity
    physics χ from the pkl directly)          perturbation, F·exp[-(t/24h)²])
              │                                       │
              └────────────► chi_s.nc ◄───────────────┘
                                                  │
                                                  ▼
                                        ensemble ODE output NC
                              (V(t) per member; ensemble stats downstream)
```

```bash
# (a) GEFS-driven prep: 1000 members, track×member parent-paired
GEFS_CASE_DIR=/global/cfs/cdirs/m5011/Jay/ERA5/GFS/2025_ERIN_NA \
GEFS_INIT_TIME='2025-08-11 12:00' \
PREP_SYNTH_NC=tracks/processed/erin_2025/2025081112/synthetic_tracks_1000members.nc \
PREP_BEST_TRACK=data/ibtracs/NA/2025/2025223N17337_ERIN/2025223N17337_ERIN_dataset.pkl \
PREP_OUT_ROOT=data/ensemble/erin_gefs_1000 \
PREP_N_MEMBERS=1000 PREP_WORKERS=32 \
python ensemble/run_prep_gefs_ens1000.py

# (b) ERA5 fully-divergent prep
PREP_SYNTH_NC=tracks/processed/irma_2017/2017090500/synthetic_tracks_1000members.nc \
PREP_BEST_TRACK=data/ibtracs/NA/2017/2017242N16333_IRMA/2017242N16333_IRMA_dataset.pkl \
PREP_OUT_ROOT=data/ensemble/irma_div1000 \
PREP_REF_TIME='2017-09-05 00:00' PREP_DURATION_H=264 \
python ensemble/run_prep_div1000.py

# ODE ensemble (FAST physics χ; KL perturbation on; F forcing on)
python ensemble/run_ode_from_chis.py \
    --in_nc  data/ensemble/irma_div1000/irma_chi_s.nc \
    --out_nc data/ensemble/irma_div1000/irma_ode.nc \
    --bt_pkl data/ibtracs/NA/2017/2017242N16333_IRMA/2017242N16333_IRMA_dataset.pkl \
    --modes fast --workers 32

# useful ODE flags
#   --init_mode fhlo|free      fhlo = 48h replay + KL; free = t=0 KL only
#   --no_kl_perturb            disable the KL intensity perturbation
#   --no_forcing               free physics after replay (F diagnostic only)
#   --replay_until_h H         glue V to obs through hour H, then release
#   --vent_scale / --vp_scale  ventilation / v_pot tuning
#   --env_mode era5 --era5_dir DIR   live ERA5EnvProvider (exact reproduction)
```

Environment variables (`run_prep_*.py`): `PREP_SYNTH_NC`, `PREP_BEST_TRACK`,
`PREP_OUT_ROOT`, `PREP_HID`, `PREP_YEAR`, `PREP_REF_TIME`, `PREP_DURATION_H`,
`PREP_N_MEMBERS`, `PREP_SEED`, `PREP_VORTEX`, `PREP_WORKERS`,
`PREP_START/END` (chunked reruns); GEFS-specific: `GEFS_CASE_DIR`,
`GEFS_INIT_TIME`, `PREP_ASSIGN_MODE` (`parent_paired` — FHLO-style, the
synthetic track inherits the environment of the parent member it was seeded
from — `round_robin`, or `random`).

---

## Data sources & directories

| use | source | local dir |
|---|---|---|
| SST | ERA5 SSTK / OISST (GHRSST) | `data/era5`, `data/oisst` |
| env fields (T/Q/U/V/Z/MSL) | ERA5 analysis / GEFS forecast | `data/era5`, CFS `gefs_root` |
| parent ensemble tracks | ECMWF TIGGE XML / GEFS GRIB2 | CFS `ecmwf_root` / `gefs_root` |
| best track | IBTrACS | `data/ibtracs` |

Local ERA5/OISST crops are **time-cropped, full spatial domain** (NA archive
0–80°N / 0–360°) so vortex surgery never hits a data edge. Demo data is
self-contained. Crop with `download/crop_beryl_sample.py NA 2024 <STORM>`.

## Physics conventions

- **χ (saturation entropy deficit)**: entropy-table `sat_deficit` at 600 hPa,
  annulus 200–800 km around the storm; calibrated downstream with the official
  Lin formula `χ_eff = clip(exp(ln(χ+1e-3)+0.5)+1.3, 1e-5, 5)` (replaces the
  legacy `clip(χ×5, 0, 4)`).
- **(Ck, h_bl) = (1.2e-3, 1400 m)** everywhere (Lin namelist; formerly split
  1.5e-3/1000 in one path — spinup coefficient +75%, now unified).
- **Environment winds/shear**: strict `vortex_lib.vortex_surgery` (Helmholtz
  decomposition removing the vortex), not an annulus mean.
- **FAST ODE**: 48 h initialization (V tracks the V_axisym target inverted
  from observed V_max; F = obs − physics accumulated, last-12-h mean), then
  forecast with F decaying as `exp[-(lead/24h)²]`; 4 sub-steps/h.
- **KL(n=10) intensity perturbation** of the past-24 h observed history,
  σ² = 5/10 m²s⁻² piecewise per observation (paper appendix B).
- **75% survival rule**: training and sampling of the Markov track model stop
  at the first lead time where <75% of parent members still track the storm.

## Reproduction & extension

- Change storms: edit `storms =` in `config.txt` or `--storms <NAME>`
- Switch SST source: `python run.py --sst OISST`
- Track sampling for another storm: `tracks/batch_generate.py --storms <NAME>`
- Full ensemble with GEFS environment: `ensemble/run_prep_gefs_ens1000.py`
  (set `track_source = GEFS` in config.txt to sample from GEFS parents)
