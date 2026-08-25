"""FHLO surface wind field: CLE15 radial profile + shape parameter k +
translation/shear asymmetry (Lin, Emanuel & Vigh 2020, section 3d).

Chain (all per ensemble member, per hourly step):
  1. V_axisym(t)  from FAST        (ensemble_fast.nc fast_v_kts)
  2. r0, k        initialized from the ATCF/IBTrACS 34/50/64-kt quadrant
                  radii at fc_start (wind/init_radii.py); held constant
                  through the forecast unless an official radii forecast
                  exists (paper section 3e; none here -> constant).
  3. CLE15(V_axisym, r0, lat) -> V(r)   (wind/cle15.py)
  4. shape: V(r,k) = V(r)**k for r > rm (paper's bias fix, 3<=r/rm<=6)
  5. asymmetry: u_net = V + G*ut + 0.1*S*|V|^0.15  (vector form below)
  6. Vmax_surface = max |u_net| over the domain
"""

import numpy as np

from .cle15 import er11e04_nondim_r0input

# FHLO paper constants
R0_DEFAULT_M = 700.0e3      # paper section 3e default when no radii analysis
K_DEFAULT = 1.0
MS_TO_KT = 1.0 / 0.514444
OMEGA = 7.292e-5


def coriolis(lat_deg):
    return 2 * OMEGA * np.sin(np.radians(np.abs(lat_deg)))


def profile_cache():
    """(V_axisym, r0, lat) -> (rr_m, vv_ms, rmax_m) memo."""
    return {}


class ProfileLookup:
    """Precomputed (V_axisym, lat) table of CLE15 profiles at fixed r0.

    r0 and k are held constant through the FHLO forecast (paper section
    3e), so V(r) depends only on (V_axisym, lat) -- both smooth. A 2-D
    table + bilinear interpolation replaces ~0.4 s/profile solves with
    ~microsecond lookups for the 1000-member x 120-h probability run.
    """

    def __init__(self, r0_m, lat_grid, v_grid, cdvary=0, c_d=1.5e-3,
                 w_cool=1.0e-3, ckcd=1.0, r_common=None, verbose=True):
        import time as _time
        self.r0_m = float(r0_m)
        self.lat_grid = np.asarray(lat_grid, float)
        self.v_grid = np.asarray(v_grid, float)
        r_common = (np.arange(0.0, self.r0_m * 1.001, 1000.0)
                    if r_common is None else np.asarray(r_common, float))
        self.r_common = r_common
        nla, nv = len(self.lat_grid), len(self.v_grid)
        self.table = np.full((nla, nv, len(r_common)), np.nan)
        self.rmax_tab = np.full((nla, nv), np.nan)
        t0 = _time.time()
        cache = {}
        n_fail = 0
        for i, la in enumerate(self.lat_grid):
            for j, v in enumerate(self.v_grid):
                rr, vv, rmax = get_profile(float(v), self.r0_m, float(la),
                                           cdvary=cdvary, c_d=c_d,
                                           w_cool=w_cool, ckcd=ckcd,
                                           cache=cache)
                if rr is None:
                    n_fail += 1
                    continue
                self.table[i, j] = np.interp(r_common, rr, vv, right=0.0)
                self.rmax_tab[i, j] = rmax
            if verbose:
                print(f'[lookup] lat {la:.0f} ({i+1}/{nla}) '
                      f'{_time.time()-t0:.0f}s elapsed, {n_fail} failures',
                      flush=True)
        if verbose:
            print(f'[lookup] done: {nla}x{nv} profiles in '
                  f'{_time.time()-t0:.0f}s', flush=True)

    def __call__(self, v_axisym_ms, lat_deg):
        """Return (r_common, V(r) interpolated, rmax) for given (V, lat)."""
        v = float(v_axisym_ms)
        la = float(lat_deg)
        if (not np.isfinite(v) or not np.isfinite(la) or v <= 1.0
                or la < self.lat_grid[0] or la > self.lat_grid[-1]
                or v < self.v_grid[0] or v > self.v_grid[-1]):
            return None, None, None
        il = np.searchsorted(self.lat_grid, la)
        il0, il1 = max(il - 1, 0), min(il, len(self.lat_grid) - 1)
        iv = np.searchsorted(self.v_grid, v)
        iv0, iv1 = max(iv - 1, 0), min(iv, len(self.v_grid) - 1)
        wl = 0.0 if il1 == il0 else (la - self.lat_grid[il0]) \
            / (self.lat_grid[il1] - self.lat_grid[il0])
        wv = 0.0 if iv1 == iv0 else (v - self.v_grid[iv0]) \
            / (self.v_grid[iv1] - self.v_grid[iv0])
        t00 = self.table[il0, iv0]
        t01 = self.table[il0, iv1]
        t10 = self.table[il1, iv0]
        t11 = self.table[il1, iv1]
        if np.isnan(t00[-1]) and np.isnan(t01[-1]) \
                and np.isnan(t10[-1]) and np.isnan(t11[-1]):
            return None, None, None
        prof = ((1 - wl) * (1 - wv) * np.nan_to_num(t00)
                + (1 - wl) * wv * np.nan_to_num(t01)
                + wl * (1 - wv) * np.nan_to_num(t10)
                + wl * wv * np.nan_to_num(t11))
        rmax = ((1 - wl) * (1 - wv) * self.rmax_tab[il0, iv0]
                + (1 - wl) * wv * self.rmax_tab[il0, iv1]
                + wl * (1 - wv) * self.rmax_tab[il1, iv0]
                + wl * wv * self.rmax_tab[il1, iv1])
        return self.r_common, prof, float(rmax)


def get_profile(v_axisym_ms, r0_m, lat_deg, cdvary=0, c_d=1.5e-3,
                w_cool=1.0e-3, ckcd=1.0, cache=None):
    """CLE15 profile for one (V, r0, lat). 2Cd/w_cool = 1 s-1 per FHLO.

    cdvary=0, Ck/Cd=1 are the FHLO simplifications (paper: "we set
    2Cd/Wcool = 1 s-1 for simplicity"; Ck=Cd in the ER11 derivation).
    Cache key rounds lat to 1 deg (Coriolis varies smoothly; sub-degree
    lat changes shift the profile negligibly vs. the model's own
    approximations) so a 1000-member ensemble reuses profiles heavily.
    """
    key = (round(float(v_axisym_ms), 1), round(float(r0_m), 0),
           round(float(lat_deg), 0))
    if cache is not None and key in cache:
        return cache[key]
    if not np.isfinite(v_axisym_ms) or v_axisym_ms <= 1.0 or r0_m <= 0:
        out = (None, None, None)
        if cache is not None:
            cache[key] = out
        return out
    fcor = coriolis(lat_deg)
    try:
        rr, vv, rmerge, vmerge, rmax = er11e04_nondim_r0input(
            v_axisym_ms, r0_m, fcor, cdvary, c_d, w_cool, 0, ckcd, 0, 1)
    except Exception:
        out = (None, None, None)
        if cache is not None:
            cache[key] = out
        return out
    out = (rr, vv, rmax)
    if cache is not None:
        cache[key] = out
    return out


def apply_shape_k(rr_m, vv_ms, k, rmax_m):
    """V(r,k) = V(r)**k for r > rm (FHLO paper eq., section 3d)."""
    vv = np.array(vv_ms, dtype=float)
    out = vv.copy()
    m = rr_m > rmax_m
    out[m] = np.power(np.clip(vv[m], 0, None), k)
    return out


def _asym_increment(v_az, ut_ms, vt_ms, u_shr, v_shr, clat):
    """Translation + shear wind increment (Lin tc_wind.py / our
    physics/run_fast_reference.py): U_inc = G*ut + 0.1*u_shr*V/15, with the
    total increment capped at 50% of |V| (mag_fac)."""
    g = min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((abs(clat) - 35.0) / 10.0)))
    u_inc = g * ut_ms + 0.1 * u_shr * v_az / 15.0
    v_inc = g * vt_ms + 0.1 * v_shr * v_az / 15.0
    mag = np.hypot(u_inc, v_inc)
    v_abs = np.abs(v_az)
    fac = np.minimum(1.0, np.where(mag > 1e-12, 0.5 * v_abs / mag, 0.0))
    return u_inc * fac, v_inc * fac


def wind_uv_at_points(lon_pts, lat_pts, clon, clat, rr_m, vv_ms,
                      ut_ms, vt_ms, u_shr, v_shr):
    """Asymmetric surface wind (u, v) at arbitrary points around the center.

    u_net = V_azimuthal + (G*ut + 0.1*S_vec*V/15), increment capped at 50% of
    V (paper section 3d; identical to axi_to_max_wind in this repo).
    """
    # local flat-plane approximation in km around the storm center
    dx = (np.asarray(lon_pts, float) - clon) * 111.32 * np.cos(np.radians(clat))
    dy = (np.asarray(lat_pts, float) - clat) * 110.57
    r_km = np.hypot(dx, dy)
    r_m = np.minimum(r_km * 1000.0, rr_m[-1])
    v_az = np.interp(r_m, rr_m, vv_ms, right=0.0)

    # azimuthal unit vector (counterclockwise, Northern Hemisphere)
    az = np.arctan2(dy, dx)
    sgn = 1.0 if clat >= 0 else -1.0
    u_v = sgn * v_az * (-np.sin(az))
    v_v = sgn * v_az * (np.cos(az))

    du, dv = _asym_increment(v_az, ut_ms, vt_ms, u_shr, v_shr, clat)
    return u_v + du, v_v + dv


def quad_wind_speeds(rr_m, vv_ms, ut_ms, vt_ms, u_shr, v_shr, clat,
                     n_az=720):
    """Net wind speed on a (theta, r) polar grid -- for quadrant radii.

    theta: position azimuth in degrees CCW from east (0=E, 90=N, 180=W,
    270=S). Returns (theta_deg (n_az,), speed (n_az, len(rr))).
    """
    th = np.radians(np.linspace(0, 360, n_az, endpoint=False))
    v_az = vv_ms[None, :]
    sgn = 1.0 if clat >= 0 else -1.0
    u_v = sgn * v_az * (-np.sin(th))[:, None]
    v_v = sgn * v_az * (np.cos(th))[:, None]
    du, dv = _asym_increment(v_az, ut_ms, vt_ms, u_shr, v_shr, clat)
    return np.degrees(th), np.hypot(u_v + du, v_v + dv)
