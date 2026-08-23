# FHLO — Forecasts of Hurricanes Using Large-Ensemble Outputs

A physics-based hurricane intensity forecasting framework that strictly follows
Lin, Emanuel & Vigh (2020), "Forecasts of Hurricanes Using Large-Ensemble
Outputs" (Wea. Forecasting, 35, 1713–1731), with the χ calibration and
(Ck, h_bl) constants aligned to [linjonathan/tropical_cyclone_risk](https://github.com/linjonathan/tropical_cyclone_risk).

Three pipelines share one codebase:

1. **Deterministic reference** — IBTrACS best track → 1 h interpolation →
   ERA5/OISST environment extraction (vortex surgery) → FAST ODE
   (`run.py`, one command).
2. **Synthetic track ensembles** — parent ensemble tracks (ECMWF TIGGE XML)
   → Markov chain on translational velocity → 1000-member sampling with
   member-paired bootstrap (`tracks/batch_generate.py`).
3. **Full 1000-member ensemble forecast** — synthetic tracks × GEFS forecast
   environment fields (member-paired or round-robin) → per-member FAST ODE
   (`run.py --ensemble`).

---

## Repository layout

```
FHLO/
├── run.py / run.slurm / config.txt      # pipeline entries + config
│                                        #   (deterministic AND ensemble modes)
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
│   ├── read_files.py / build_pairs.py   #   TIGGE XML -> 6h velocity pairs
│   ├── train_markov.py / sample_tracks.py  # per-step Gaussian fit + member-paired sampling
│   └── config.py / plot_tracks.py
├── ensemble/                            # ensemble env adapter (consumed by run.py)
│   └── gefs_nc_adapter.py               #   GEFS local nc -> env fields (a+b merge,
│                                         #   MSL+skt SST, native 0.5-deg, fhour cache)
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

## Pipeline 3 — Full 1000-member ensemble forecast (`run.py --ensemble`)

One command drives the whole chain: synthetic tracks (Pipeline 2) are spliced
hourly with the best-track intensity, each track is assigned a GEFS ensemble
member's forecast environment, prep extracts per-member scalars with strict
vortex surgery, and the FAST ODE produces per-member V(t) — all in a process
pool.

```
synthetic_tracks_1000members.nc ──┐
 (Pipeline 2 output, with          │  prep/prepare_ensemble_storm.py
  parent_track)                    │  (hourly splice; vmax from best track)
 best-track dataset.pkl ──────────┤
                                   ▼
        run.py --ensemble (stage eprep)
          resolve_member_assignment():
            ecmwf       parent_track[i] % 31  (51 ECMWF parents -> 31 GEFS members)
            gefs        parent GEFS code directly (self-consistent)
            round_robin i % 31
          ensemble/gefs_nc_adapter.py (pgrb2a+b level merge, MSL + skt SST,
            native 0.5-deg grid, fhour cache, vortex surgery incl. edge
            degradation)  ->  {OUT}/{STORM}_M{NNN}/{STORM}_M{NNN}_dataset.pkl
                                   │
        run.py --ensemble (stage ode)
          physics/run_fast_reference.py per member
            -> fast_reference.csv / fast_reference.png + ensemble summary
```

```bash
# quick test: 5 members
python run.py --ensemble \
    --synth-nc tracks/processed/beryl_2024/2024062900/synthetic_tracks_1000members.nc \
    --members 5 --assign ecmwf --workers 5

# full 1000-member run (slurm: run.slurm passes args through)
python run.py --ensemble \
    --synth-nc tracks/processed/beryl_2024/2024062900/synthetic_tracks_1000members.nc \
    --members 1000 --assign ecmwf --workers 32

# key flags
#   --assign ecmwf|gefs|round_robin   env-member assignment mode
#   --gefs-init / --gefs-dir          GEFS forecast init / local nc dir
#                                     (defaults from config.txt)
#   --members N                       subset for testing
#   --duration-h H                    per-member forecast length (default 240)
#   --out-root DIR                    default data/ensemble/beryl_gefs_1000
```

Output layout: `data/ensemble/beryl_gefs_1000/{STORM}_M{NNN}/` with
`_track.csv`, `member_assignment.txt`, `_dataset.pkl`, `fast_reference.csv`,
`fast_reference.png`; the run finishes with an ensemble peak-intensity
summary (mean/median/min/max/sd across members).

---

## Data sources & directories

| use | source | local dir |
|---|---|---|
| SST (deterministic) | ERA5 SSTK / OISST (GHRSST) | `data/era5`, `data/oisst` |
| SST (ensemble) | GEFS skt skin temperature | `data/gefs_beryl/skt_*.nc` |
| env fields (T/Q/U/V/Z/MSL) | ERA5 analysis / GEFS forecast | `data/era5`, `data/gefs_beryl` |
| parent ensemble tracks | ECMWF TIGGE XML | CFS `ecmwf_root` |
| best track | IBTrACS | `data/ibtracs` |

Local ERA5/OISST crops are **time-cropped, full spatial domain** (NA archive
0–80°N / 0–360°) so vortex surgery never hits a data edge. The GEFS ensemble
crop is regional (0.5°, 3.5–48.5°N / 258.5–326°E); the surgery wrapper
degrades gracefully near those edges (unfiltered center wind instead of a
crash). Crop GEFS with `download/crop_gefs_finish.py` + `download/crop_gefs_skt.py`.

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
- Full ensemble with GEFS environment: `python run.py --ensemble`
  (see Pipeline 3; assignment modes `ecmwf` / `gefs` / `round_robin`)
