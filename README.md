# FHLO — Forecasts of Hurricanes Using Large-Ensemble Outputs

Physics-based hurricane intensity ensemble forecasting following
Lin, Emanuel & Vigh (2020), *Wea. Forecasting*, 35, 1713–1731, with the χ
calibration and (Ck, h_bl) constants aligned to
[linjonathan/tropical_cyclone_risk](https://github.com/linjonathan/tropical_cyclone_risk).

## Quick start

```bash
conda env create -f environment.yml && conda activate fast_ml

# ---- THE two ensemble run commands (paths/init all from config.txt) ----
# 1) GEFS forecast ensemble: GEFS-sampled tracks x GEFS member env, 1000 members
#    (eprep -> ode -> summary -> ensemble_fast.png/svg, all in one job)
sbatch ensemble.slurm

# 2) ERA5 analysis ensemble: ECMWF-sampled tracks x ERA5 env, 1000 members
sbatch ensemble_era5.slurm

# quick sanity check (5 members, foreground)
python run.py --ensemble --env gefs --members 5 --workers 5

# re-plot only (reads the saved ensemble_fast.nc)
python run.py --ensemble --env gefs --stage plot
# or standalone:
python ensemble/plot_ensemble_fast.py --ens-nc data/ensemble/flossie_gefs/ensemble_fast.nc \
    --out_png data/ensemble/flossie_gefs/ensemble_fast.png --storm FLOSSIE
```

Everything each command needs is in `config.txt`. Per-storm keys use the
lowercase storm tag (`{id}_{NAME}` -> `flossie`):

| key | meaning |
|---|---|
| `gefs_dir_{tag}` / `gefs_init_{tag}` | local GEFS forecast nc dir + init time |
| `synth_gefs_nc_{tag}` / `synth_ecmwf_nc_{tag}` | synthetic-track NCs each mode reads |
| `era5_dir` / `oisst_dir` | local analysis crops (env fields, SST) |
| `storms`, `n_workers` | storm selection, process-pool size |

Legacy Beryl keys (`gefs_beryl_dir`, `synth_gefs_nc`, ...) still work as
fallbacks when no per-storm key exists.

## Ensemble outputs (`data/ensemble/beryl_{gefs\|era5}/`)

| file | content |
|---|---|
| `ensemble_fast.nc` | **final result**: `fast_vmax_kts (member, hour)` + `fast_chi`/`fast_s` (vent), `vp/v_obz/m`, `seq_len`, member codes |
| `ensemble_fast.png/.svg` | ensemble plot: spaghetti + mean/top10% + obs + vent panel (auto after ode) |
| `ensemble_winds.csv` | long-format V(t) per member (easy pandas) |
| `ensemble_summary.nc` | per-member coefficients: chi, u/v 250/850, shear, peak_kts |
| `{STORM}_M{NNN}/` | per-member: slim `_dataset.pkl` (coefficients only, no env grids), `fast_reference.csv`, `_track.csv`, `member_assignment.txt` |

`ensemble_fast.nc` + `ensemble/plot_ensemble_fast.py` are the plotting pair
(FAST-only; the old `plot_vs_google_vmax.py` ML branch was removed). Optional
Google FNV3 overlay via `--google_csv`.

## How the ensemble works

```
TIGGE parents (GEFS kwbc / ECMWF ecmf)
   │ tracks/read_files.py -> build_pairs -> train_markov -> sample_tracks
   │   (1000 members, parent_track kept for member-paired bootstrap)
   ▼
synthetic_tracks_1000members.nc ──┐ hourly splice w/ best-track vmax
 best-track dataset.pkl ──────────┤        (prep/prepare_ensemble_storm)
                                  ▼
              run.py --ensemble  (stage eprep, process pool)
                --assign gefs     track's parent GEFS member directly
                --assign ecmwf    parent_track % 31 -> GEFS member
                env: gefs -> ensemble/gefs_nc_adapter.py
                     (pgrb2a+b merge, skt SST, fhour cache; p25-p30
                      u/v/gh from a-stream, missing t/q levels log-p filled)
                     era5 -> data/era5 analysis directly
                                  ▼
              run.py --ensemble  (stage ode)
                physics/run_fast_reference.py per member (FAST ODE only)
                                  ▼
              ensemble_fast.nc + ensemble_winds.csv + ensemble_summary.nc
                                  ▼
              ensemble_fast.png/svg  (ensemble/plot_ensemble_fast.py,
                                     auto-invoked as the final stage)
```

- **Vortex surgery** on all env winds (25° box, Helmholtz decomposition);
  near the GEFS regional-grid edge it degrades to the unfiltered center
  wind instead of crashing.
- **FAST ODE** only in this repo path (no ML): 48 h init (V tracks V_axisym
  from observed V_max; F = obs − physics), forecast F·exp[−(lead/24h)²],
  4 sub-steps/h.
- Workers: jobs are batched per GEFS member (fhour cache locality), BLAS
  threads pinned to 1, tqdm off under multiprocessing.

## Deterministic reference (`run.py`, single best track)

```bash
python prep/IBtracs_datasets.py                    # IBTrACS -> 6h CSV
python run.py --storms 2024181N09320_BERYL         # prep + FAST ODE
# -> data/ibtracs/{basin}/{year}/{STORM}/fast_reference.csv/.png
```

`{STORM}_dataset.pkl` keys: `scalars (1,T,4)` = α/β/γ/v_pot;
`chi_ref/s_ref/xs_ref (1,T,1)`; `env_wnds (1,T,4)`; `v_gt/times/lats/lons`;
`cd_ref/blh_ref/utran/vtran`. Beryl 2024: MAE ≈ 30 kts, peak ~116 vs 145
observed.

## Synthetic-track sampling (`tracks/`)

```bash
python tracks/batch_generate.py --storms 2024181N09320_BERYL [--source gefs] [--plot]
```

`tracks/processed/{storm}/{cycle}/synthetic_tracks_1000members.nc` carries
`parent_track` + `parent_members` attrs (member-paired bootstrap). Pre-genesis
GEFS cycles fall back to matching unnamed 'Invest' disturbances against the
best-track genesis position. Details: `tracks/README.md`.

## Repository layout

```
run.py / ensemble.slurm / ensemble_era5.slurm / config.txt
prep/      IBTrACS download; env extraction + scalars (dataset.pkl)
physics/   FAST ODE (run_fast_reference.py, Fast.py, env.py)
tracks/    TIGGE -> Markov -> 1000-member synthetic tracks
ensemble/  gefs_nc_adapter.py (GEFS nc -> env fields), plot_vs_google_vmax.py
common/    thermo tables, vortex surgery, spherical utils
download/  data download & crop scripts (ERA5/OISST/GEFS)
data/      local data (gitignored): era5, gefs_beryl, oisst, ibtracs, ensemble
```

## Physics conventions

- **χ**: entropy-table `sat_deficit` at 600 hPa, 200–800 km annulus; Lin
  calibration `χ_eff = clip(exp(ln(χ+1e-3)+0.5)+1.3, 1e-5, 5)`.
- **(Ck, h_bl) = (1.2e-3, 1400 m)** everywhere (Lin namelist).
- **Env winds/shear**: strict vortex surgery, not annulus means.
- **KL(n=10)** perturbation of past-24 h obs (σ² = 5/10 m²s⁻²).
- **75% survival rule** for Markov track training/sampling horizon.

## Data notes

- GEFS crop: 0.5° regional (3.5–48.5°N / 258.5–326°E), `pgrb2a`+`pgrb2b`
  merged per member; SST from `skt_{member}.nc` (skin temperature) — no OISST
  in ensemble mode.
- ERA5/OISST crops: time-cropped, full spatial domain (vortex surgery never
  hits an edge).
- Sources: `era5_root` / `oisst_root` / `ecmwf_root` (+ kwbc GEFS TIGGE) in
  `config.txt`.
