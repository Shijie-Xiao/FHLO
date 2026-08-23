# FHLO Synthetic Track Module (`tracks/`)

Implementation of the ensemble synthetic-track sampler from
Lin, Emanuel & Vigh (2020), "Forecasts of Hurricanes Using Large-Ensemble Outputs", Wea. Forecasting, 35, 1713-1731 (section 3a). Parent ensemble
cyclone tracks (ECMWF TIGGE) are modeled as a Markov chain in translational
velocity; a per-lead-time conditional Gaussian is fitted and sampled to
generate 1000 statistically indistinguishable synthetic tracks.

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

## Output layout

```
tracks/processed/{storm_lowercase}_{year}/{YYYYMMDDHH}/
    raw.pkl                           # parent ensemble member tracks (this cycle)
    pairs_6h.pkl                      # 6h velocity pairs + 75% survival horizon
    markov_params_6h.pkl              # per-lead-time k=1 conditional Gaussians
    synthetic_tracks_1000members.nc   # sampled tracks (consumed downstream)
    tracks.png                        # written when --plot
```

NC variables: `lon/lat/u/v (track, time)` plus `parent_track (track)` when the
raw.pkl carries parent attribution; attrs include `init_time`, `dt_hours`,
`assignment` (`member_paired` | `pooled`), `parent_members` — the exact
interface consumed by `prep/prepare_ensemble_storm._load_synthetic` and
`run.py --ensemble`.

## Member-paired inheritance

When raw.pkl tracks carry `parent_member` (GEFS) or numeric `member_id`
(ECMWF TIGGE, synthesized to `e00..e50`), `sample_tracks.sample_case`
bootstraps synthetic tracks in equal per-parent blocks (with replacement
inside each block) and records `parent_track[i] = k`. Downstream
`run.py --ensemble --assign ecmwf|gefs` then serves track i its environment
from parent member k (ECMWF parents map onto the 31 GEFS members via
`k % 31`), keeping track and environment self-consistent (FHLO paper
member-paired spirit).

## Modules

| file | role |
|---|---|
| `config.py` | constants, config.txt parsing, storm discovery, dir naming |
| `read_files.py` | (storm, cycle) -> raw.pkl; ECMWF TIGGE XML |
| `build_pairs.py` | raw -> velocity pairs; 75% survival horizon |
| `train_markov.py` | per-lead-time k=1 Gaussian fit (authoritative Reproduce implementation) |
| `sample_tracks.py` | conditional-Gaussian chain, 1000 members, horizon-capped, member-paired bootstrap + `parent_track` NC write |
| `plot_tracks.py` | single plotting interface `plot_case(case_dir, plot=True/False)` |
| `batch_generate.py` | the one CLI entry point chaining all stages |

## Conformance with the FHLO paper

| point | paper status | this implementation |
|---|---|---|
| Markov chain on translational velocity, P(u_t, v_t \| u_{t-1}, v_{t-1}) | explicit (sec. 3a) | yes — conditional Gaussian each step |
| mixture of k Gaussians, k = 1 | explicit; k counts mixture components | single Gaussian (no EM/mixture) per lead time |
| per-lead-time estimation | implied by "proceeding to the next time step" with the 75% rule; validated numerically (pooled fits bias the synthetic mean ~800 km at 96 h on recurving Irma) | per-step fit from that step's member rows only |
| 75% survival rule | explicit: no step where <75% of members survive | `survival_horizon()` caps both training and sampling |
| first-point velocity | not specified in the paper | backward extrapolation u0 = 2*u1 - u2 (a copy u0=u1 makes the step-1 pair perfectly correlated and degenerates A -> I, Sigma_cond -> 0) |
| initial position/velocity | not prescribed beyond using the analyzed storm | member-paired bootstrap from parent members (preserves parent t=0 spread and parent attribution) |
| 1000 members, 6h steps | explicit | yes (`N_TRACKS`, `DT_HOURS`) |

## Ensemble integration (verified)

The NC output was fed directly through the downstream loader
`prep/prepare_ensemble_storm._load_synthetic`:

- `irma_2017/2017090500`: 1000 tracks x 34 steps (198 h horizon),
  init 2017-09-05 00Z, lon wrapped to 0..360 correctly
- `beryl_2024/2024062900`: 1000 tracks x 25 steps (144 h horizon),
  `parent_track` blocks over 51 ECMWF parents, `assignment=member_paired`

Both pass: member picking, time grid, 0..360 lon convention — i.e. the
module plugs into `run.py --ensemble` as-is via `--synth-nc`.

## Notes

- TIGGE XML longitude sign differs by generation (2017-22: `units='deg W'`
  with unsigned values; 2023+: `units='W'` with signed values);
  `read_files.parse_tigge_xml` normalizes both.
