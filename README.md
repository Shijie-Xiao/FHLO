# FHLO — Forecasts of Hurricanes Using Large-Ensemble Outputs

Physics-based hurricane intensity ensemble forecasting following
Lin, Emanuel & Vigh (2020), *Wea. Forecasting*, 35, 1713–1731, with the χ
calibration and (Ck, h_bl) constants aligned to
[linjonathan/tropical_cyclone_risk](https://github.com/linjonathan/tropical_cyclone_risk).

**This repository is a self-contained demo**: Hurricane Flossie 2025 (EP),
GEFS 06-29 06Z init, 1000-member **cold-start** forecast. All code and all
data needed to reproduce the results live inside this directory — no
external paths, no external packages beyond the conda env, no observation
nudging anywhere in the forecast (the only observation ever used is the
initial vmax at t=0).

## Quick start

```bash
conda env create -f environment.yml && conda activate fast_ml

# ---- THE one-command full reproduction ----
python run.py --ensemble
#   eprep (1000 members x GEFS env) -> ode (FAST cold start) -> plot
#   everything (synth NC, GEFS dir, init time, Vp compensation) resolves
#   from config.txt; output lands in data/ensemble/flossie_gefs/

# quick sanity check (5 members, foreground)
python run.py --ensemble --members 5 --workers 5

# re-plot only (reads the saved ensemble_fast.nc)
python run.py --ensemble --stage plot
# or standalone:
python ensemble/plot_ensemble_fast.py --ens-nc data/ensemble/flossie_gefs/ensemble_fast.nc \
    --out_png data/ensemble/flossie_gefs/ensemble_fast.png --storm FLOSSIE
```

On an HPC with Slurm, `sbatch ens_flossie_gefs_default.slurm` runs the same
command (1000 members, 128 workers).

### Optional chains (kept selectable; GEFS is the default)

```bash
# ERA5 analysis env instead of GEFS forecast env (ECMWF-sampled tracks)
python run.py --ensemble --env era5
```

`--env gefs` (default) pairs each synthetic track with its parent GEFS
member's forecast fields; `--env era5` uses the local ERA5 analysis for every
member. `--assign` controls the track→member mapping (`gefs`, `ecmwf` hash,
`round_robin`; `auto` picks by `--env`).

### ODE initialization modes (`--ode-mode`, config `ode_mode`)

```bash
# FHLO Sec.2c init: obs replay + KL(n=10) + F*exp(-(t/24h)^2) forcing
python run.py --ensemble --ode-mode fhlo

# replay + KL, no forecast-phase forcing (free physics from replayed state)
python run.py --ensemble --ode-mode free
```

With a later GEFS init than the IBTrACS record start (e.g. `--gefs-init
"2025-06-30 12:00"`), the environment chain becomes **dual-source**: the
replay window `[fc_start - replay_hours, fc_start)` runs on ERA5 analysis
fields (Lin et al. 2020 Sec.3e convention) while the forecast segment runs
on the selected GEFS member. The replay window is always clipped to the
IBTrACS record start — never extrapolated pre-genesis — and degenerates to
a cold start automatically when the record starts at/after `fc_start`
(the shipped 06-29 06Z demo case). `--no-kl` disables the appendix-B KL
perturbation; `--replay-hours` sets the window (default 48 h).

## Configuration (`config.txt`)

| key | meaning |
|---|---|
| `storms`, `basins`, `year_start/end` | storm selection (demo: Flossie EP 2025) |
| `gefs_dir_flossie` | local GEFS forecast nc dir (`data/gefs_flossie`, 93 files) |
| `gefs_init_flossie` | GEFS init time `2025-06-29 06:00` |
| `synth_gefs_nc_flossie` | pre-generated synthetic-track NC (1000 members) |
| `synth_ecmwf_nc_flossie` | optional ECMWF-parent NC for `--env era5` |
| `era5_dir` | local ERA5 analysis crop (SST/BLH/MSL fallback chain) |
| `vp_comp_gefs` / `vp_comp_era5` | Vp bias compensation (1.1 / 1.0) |
| `ode_mode` | ODE init: `cold` (default) / `fhlo` (obs replay + KL + F) / `free` |
| `replay_hours` | obs replay window before fc_start (48; clipped to IBTrACS start) |
| `kl_perturb` | 1 = KL(n=10) observed-history perturbation (appendix B) |
| `n_workers` | process-pool size |

`era5_root` / `oisst_root` / `ecmwf_root` / `gefs_grib_root` point at the
original community archives and are used **only** by `download/` crop scripts
and `tracks/` generation — never by `run.py` prep/ode stages, which read
exclusively from the local `data/` tree shipped here.

## Bundled data (all inside this workspace)

```
data/ibtracs/EP/2025/2025180N13261_FLOSSIE/   best-track pkl + 1h/6h CSVs
data/gefs_flossie/                            31 members x (pgrb2a+pgrb2b+skt) nc, 0.5° regional
data/era5/                                    T/Q/U/V/Z 2025-06-27..07-08 + SSTK/MSL/BLH/SP monthly (0.25° global)
tracks/processed/flossie_2025/2025062906_gefs/ raw.pkl + markov + synthetic_tracks_1000members.nc
precalc_data/                                 Cd.nc, mld/strat climatology, bathymetry, land (dynamic alpha)
```

Total ≈ 47 GB. Everything `run.py` touches is local; after unzipping the
Google Drive archive the one command above reproduces the full ensemble with
zero additional downloads.

## Ensemble outputs (`data/ensemble/flossie_gefs/`)

| file | content |
|---|---|
| `ensemble_fast.nc` | **final result**: `fast_vmax_kts (member, hour)` + `fast_chi`/`fast_s` (vent), `vp/v_obz/m`, `seq_len`, member codes |
| `ensemble_fast.png/.svg` | spaghetti + mean/top10% + obs + vent panel (auto after ode) |
| `ensemble_winds.csv` | long-format V(t) per member (easy pandas) |
| `ensemble_summary.nc` | per-member coefficients: chi, u/v 250/850, shear, peak_kts |
| `{STORM}_M{NNN}/` | per-member slim `_dataset.pkl`, `fast_reference.csv`, `_track.csv`, `member_assignment.txt` |
| `run_config.txt` | exact config used (env, vortex mode, members, vp_comp) |

## How the ensemble works

```
TIGGE parents (GEFS kwbc)  [done once, NC shipped in tracks/processed/]
   │ tracks/read_files.py -> build_pairs -> train_markov -> sample_tracks
   │   (1000 members, parent_track kept for member-paired bootstrap)
   ▼
synthetic_tracks_1000members.nc ──┐ hourly splice w/ best-track vmax
 best-track dataset.pkl ──────────┤        (prep/prepare_ensemble_storm)
                                  ▼
              run.py --ensemble  (stage eprep, process pool)
                --assign gefs     track's parent GEFS member directly
                env: gefs -> ensemble/gefs_nc_adapter.py
                     (pgrb2a+b merge, skt SST, fhour cache; p25-p30
                      u/v/gh from a-stream, missing t/q levels log-p filled)
                     era5 -> data/era5 analysis directly
                                  ▼
              run.py --ensemble  (stage ode)
                physics/run_fast_reference.py per member — COLD START:
                V(0) = first observed vmax (axisymmetric inversion),
                m(0) = official _init_m inversion at dv/dt=0, then pure
                physics; NO replay, NO F forcing, no obs afterwards
                                  ▼
              ensemble_fast.nc + ensemble_winds.csv + ensemble_summary.nc
                                  ▼
              ensemble_fast.png/svg  (ensemble/plot_ensemble_fast.py)
```

- **Vortex removal**: `annulus` (200–800 km mean, default) on the regional
  GEFS grid; strict Lin vortex `surgery` needs full-global fields and is
  therefore only available with `--env era5`.
- **Ocean coupling is always dynamic** (official `coupled_fast` Eq. 4-5):
  α recomputed from the current V every 15-min sub-step via `precalc_data/`
  monthly MLD/strat climatology + bathymetry; γ = ε + ακ.
- **Cd chain** (official `geo.read_drag`): 10-m drag → gradient-height
  correction `Cd/(1+250·Cd)` → normalized to open-ocean 1.2e-3; h_bl per
  basin from the namelist `atm_bl_depth` (EP/NA 1400 m → coeff 1.543e-3/h).
- **Vp compensation** 1.1 for GEFS (systematic 5-10% low bias vs ERA5
  analysis; a maintained correction, not case tuning), 1.0 for ERA5.
- Workers: jobs batched per GEFS member (fhour cache locality), BLAS threads
  pinned to 1.

## Deterministic single-track path (`run.py`, no ensemble)

```bash
python prep/IBtracs_datasets.py                    # IBTrACS -> 6h CSV (uses _cache)
python run.py --storms 2025180N13261_FLOSSIE       # prep + FAST ODE
# -> data/ibtracs/EP/2025/{STORM}/fast_reference.csv/.png
```

`{STORM}_dataset.pkl` keys: `scalars (1,T,4)` = α/β/γ/v_pot;
`chi_ref/s_ref (1,T,1)`; `env_wnds (1,T,4)`; `v_gt/times/lats/lons`;
`cd_ref/blh_ref/utran/vtran/hm_ref/strat_ref/bathy_ref`.

## Synthetic-track regeneration (`tracks/`, optional)

The shipped NC was generated from the GEFS TIGGE `kwbc` XML archive
(`gefs_grib_root` in config); regenerating needs that archive:

```bash
python tracks/batch_generate.py --storms 2025180N13261_FLOSSIE --source gefs --init FLOSSIE:2025062906 --plot
```

Details: `tracks/README.md`. The demo never requires this step.

## Repository layout

```
run.py                       single entry: prep/eprep/ode/plot orchestration
config.txt                   all paths + experiment knobs
ens_flossie_gefs_default.slurm  HPC batch wrapper (1000 members)
prep/      IBTrACS download; env extraction + scalars (dataset.pkl)
physics/   run_fast_reference.py — FAST ODE (cold / fhlo replay / free)
tracks/    TIGGE -> Markov -> 1000-member synthetic tracks
ensemble/  gefs_nc_adapter.py (GEFS env), dual_env_adapter.py (ERA5 replay
           + GEFS forecast routing), plot_ensemble_fast.py, download_fnv3.py
common/    thermo tables, vortex-inversion surgery library, spherical utils
download/  crop scripts that BUILT data/era5 + data/gefs_flossie (optional)
data/      local data (gitignored; ship via Google Drive)
```

## Physics conventions

- **χ**: entropy-table `sat_deficit` at 600 hPa, 200–800 km annulus (EP:
  50th percentile, 900 km renv); Lin calibration
  `χ_eff = clip(exp(ln(χ+1e-3)+0.5)+1.3, 1e-5, 5)`; vent = χ_eff·s.
- **(Ck, h_bl) = (1.2e-3, 1400 m)** for EP/NA (Lin namelist `atm_bl_depth`;
  WP/AU 1800, SI 1600, SP 2000, NI 1500).
- **Integration**: Heun, 4 sub-steps per hour; V_axisym → V_max via the
  translation + 0.1·s·V/15 shear-tilt conversion at the end.
- **75% survival rule** for Markov track training/sampling horizon.

## Reproducibility notes

- `run_config.txt` next to each ensemble output records the exact
  configuration (storm, env source, vortex mode, members, vp_comp, init,
  ode_mode, replay_hours, KL).
- Reruns are resumable: existing per-member `_dataset.pkl`s are skipped
  unless `--overwrite`.
- In `cold` mode the ODE is deterministic per member; spread comes entirely
  from the 1000 synthetic tracks x 31 GEFS parent members. In `fhlo`/`free`
  modes the KL(n=10) observed-history perturbation (seed `100000 + member
  id`, deterministic) adds IC spread on top, exactly per FHLO appendix B.
