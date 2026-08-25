"""FHLO surface wind field model (CLE15 + asymmetry + probabilities).

Faithful replication of the wind-field component of Lin, Emanuel & Vigh
(2020, Wea. Forecasting, 35, 1713-1731), section 3d/3e/4d:

  wind.cle15          Chavas et al. (2015) profile, official python port
                      (DOI 10.4231/CZ4P-D448, CC0)
  wind.wind_field     shape parameter k + translation/shear asymmetry
  wind.init_radii     (r0, k) initialization from IBTrACS 34/50/64-kt
                      quadrant radii (paper section 3e)
  wind.wind_prob      gridpoint exceedance probabilities (paper section 4d,
                      DeMaria et al. 2009 method)
  wind.plot_wind_prob probability maps

Typical run (Flossie 2025, fc start 2025-06-30 12Z):

  python -m wind.wind_prob --ens data/ensemble/flossie_gefs_fhlo48 \
      --ibtracs data/ibtracs/_cache/ibtracs.EP.list.v04r01.csv \
      --sid 2025180N13261 --thresholds 34,50,64 --window-h 120
"""
