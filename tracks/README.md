# FHLO Synthetic Track Module (`tracks/`)

Implementation of the ensemble synthetic-track sampler from
Lin, Emanuel & Vigh (2020), "Forecasts of Hurricanes Using Large-Ensemble
Outputs", Wea. Forecasting, 35, 1713-1731 (section 3a). Parent ensemble
cyclone tracks (ECMWF TIGGE / GEFS) are modeled as a Markov chain in
translational velocity; a per-lead-time conditional Gaussian is fitted and
sampled to generate 1000 statistically indistinguishable synthetic tracks.

## Quick start

```bash
# One storm, default init = the 00/12Z cycle at/after IBTrACS genesis
python tracks/batch_generate.py --storms 2024181N09320_BERYL

# Explicit init time (paper Irma case) + plotting
python tracks/batch_generate.py --storms 2017242N16333_IRMA \
    --init IRMA:2017090500 --plot

# Every 00/12Z cycle over the storm lifetime (paper evaluation mode)
python tracks/batch_generate.py --storms 2024181N09320_BERYL --cycles all

# Batch: several storms / all discovered storms
python tracks/batch_generate.py --storms A,B,C --plot
python tracks/batch_generate.py --all

# Re-run only sampling and plotting (inputs already read)
python tracks/batch_generate.py --stage sample,plot --storms ...
```

Without `--storms`, the storm list comes from `FHLO/config.txt` (`storms =`).
The parent-ensemble source is set by `track_source = ECMWF | GEFS` in
config.txt (overridable with `--source`).

## Output layout

```
tracks/processed/{storm_lowercase}_{year}/{YYYYMMDDHH}/
    raw.pkl                           # parent ensemble member tracks (this cycle)
    pairs_6h.pkl                      # 6h velocity pairs + 75% survival horizon
    markov_params_6h.pkl              # per-lead-time k=1 conditional Gaussians
    synthetic_tracks_1000members.nc   # sampled tracks (consumed downstream)
    tracks.png                        # written when --plot
```

NC variables: `lon/lat/u/v (track, time)`; attrs include
`init_time`, `dt_hours`, `storm_dir` — the exact interface consumed by
`prep/prepare_ensemble_storm._load_synthetic` and the ensemble pipeline
(`ensemble/run_prep_div1000.py`).

## Modules

| file | role |
|---|---|
| `config.py` | constants, config.txt parsing, storm discovery, dir naming |
| `read_files.py` | (storm, cycle) -> raw.pkl; ECMWF TIGGE XML or GEFS GRIB2 vortex tracking |
| `build_pairs.py` | raw -> velocity pairs; 75% survival horizon |
| `train_markov.py` | per-lead-time k=1 Gaussian fit (authoritative Reproduce implementation) |
| `sample_tracks.py` | conditional-Gaussian chain, 1000 members, horizon-capped |
| `plot_tracks.py` | single plotting interface `plot_case(case_dir, plot=True/False)` |
| `batch_generate.py` | the one CLI entry point chaining all stages |
| `gefs_tracks.py` | GEFS pgrb2a vortex tracking (GEFS branch of read_files) |

## Conformance with the FHLO paper

| point | paper status | this implementation |
|---|---|---|
| Markov chain on translational velocity, P(u_t, v_t \| u_{t-1}, v_{t-1}) | explicit (sec. 3a) | yes — conditional Gaussian each step |
| mixture of k Gaussians, k = 1 | explicit; k counts mixture components | single Gaussian (no EM/mixture) per lead time |
| per-lead-time estimation | implied by "proceeding to the next time step" with the 75% rule; validated numerically (pooled fits bias the synthetic mean ~800 km at 96 h on recurving Irma) | per-step fit from that step's member rows only |
| 75% survival rule | explicit: no step where <75% of members survive | `survival_horizon()` caps both training and sampling |
| first-point velocity | not specified in the paper | backward extrapolation u0 = 2*u1 - u2 (a copy u0=u1 makes the step-1 pair perfectly correlated and degenerates A -> I, Sigma_cond -> 0) |
| initial position/velocity | not prescribed beyond using the analyzed storm | bootstrapped from parent members (preserves parent t=0 spread) |
| 1000 members, 6h steps | explicit | yes (`N_TRACKS`, `DT_HOURS`) |

## Ensemble integration (verified)

The NC output was fed directly through the downstream loader
`prep/prepare_ensemble_storm._load_synthetic`:

- `irma_2017/2017090500`: 1000 tracks x 34 steps (198 h horizon),
  init 2017-09-05 00Z, lon wrapped to 0..360 correctly
- `beryl_2024/2024062900`: 1000 tracks x 25 steps (144 h horizon)

Both pass: member picking, time grid, 0..360 lon convention — i.e. the
module plugs into `ensemble/run_prep_div1000.py` as-is by setting
`PREP_SYNTH_NC` to a case NC path.

## Notes

- TIGGE XML longitude sign differs by generation (2017-22: `units='deg W'`
  with unsigned values; 2023+: `units='W'` with signed values);
  `read_files.parse_tigge_xml` normalizes both.
- The GEFS branch (`track_source = GEFS`) is wired but not yet exercised in
  batch; smoke-test one storm before bulk runs.
