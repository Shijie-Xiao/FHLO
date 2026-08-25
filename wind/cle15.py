"""CLE15 complete TC radial wind profile -- faithful Python 3 port.

Source: CLE15_2020-06-23.py, the official release of the wind profile model
of Chavas, Lin & Emanuel (2015, JAS) and Chavas & Lin (2016, JAS), DOI
10.4231/CZ4P-D448 (python version by Chia-Ying Lee), licensed CC0.

This is the same physics FHLO uses for its 2-D surface wind field
(Lin, Emanuel & Vigh 2020, Wea. Forecasting, section 3d):
  - inner convecting region: Emanuel & Rotunno (2011) solution
  - outer subsiding region : Emanuel (2004) solution, integrated inwards
  - merged at the tangent point of M(r) via shapely line intersection.

Port notes (py2 -> py3 only, NO physics changes):
  - `print x` -> `print(x)`; `np.float('NaN')` -> `np.float64('nan')`;
    `zip` wrapped in `list(...)` for shapely LineString.
  - Iteration-limit guards added where the original could loop forever
    on pathological inputs (guards raise no physics change; they cap).
Everything else, including integration step sizes, convergence thresholds,
the variable-Cd Donelan fit and the geometric bisection, is unchanged.
"""

import copy

import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString

__all__ = ['e04_outerwind_r0input_nondim_mm0', 'er11_radprof_raw',
           'er11_radprof', 'er11e04_nondim_r0input',
           'er11e04_nondim_rmaxinput', 'er11e04_nondim_rfitinput']


def e04_outerwind_r0input_nondim_mm0(r0, fcor, cdvary, c_d, w_cool, nr):
    """Emanuel (2004) outer solution: integrate M/M0 inwards from r0.

    Returns (rrfracr0, MMfracM0): nondimensional radius and angular momentum.
    """
    fcor = abs(fcor)
    m0 = .5 * fcor * r0**2                                   # M at outer radius

    drfracr0 = .001
    if (r0 > 2500 * 1000) or (r0 < 200 * 1000):
        # extra precision for very large storm to avoid funny bumps near r0,
        # or for tiny storm that requires E04 extend to very small radii
        drfracr0 = drfracr0 / 10

    if nr > 1 / drfracr0:
        nr = int(1 / drfracr0)          # grid radii must be > 0

    rfracr0_max = 1                      # start at r0, move radially inwards
    rfracr0_min = rfracr0_max - (nr - 1) * drfracr0
    rrfracr0 = np.arange(rfracr0_min, rfracr0_max + drfracr0, drfracr0)
    mmfracm0 = np.full(rrfracr0.size, np.float64('nan'))
    mmfracm0[-1] = 1

    # First step inwards from r0: d(M/M0)/d(r/r0) = 0 by definition
    rfracr0_temp = rrfracr0[-2]
    mfracm0_temp = mmfracm0[-1]
    mmfracm0[-2] = mfracm0_temp

    # Variable C_d: piecewise linear Donelan (2004) fit (Cd_Donelan04.m)
    cd_lowv = 6.2e-4
    v_thresh1 = 6          # m/s constant -> linearly increasing
    v_thresh2 = 35.4       # m/s linearly increasing -> constant
    cd_highv = 2.35e-3
    linear_slope = (cd_highv - cd_lowv) / (v_thresh2 - v_thresh1)

    # Integrate inwards from r0 to obtain profile of M/M0 vs. r/r0
    for ii in range(0, int(nr) - 2, 1):
        if cdvary == 1:
            # V at this r/r0 (for variable C_d only)
            v_temp = (m0 / r0) * ((mfracm0_temp / rfracr0_temp) - rfracr0_temp)
            if v_temp <= v_thresh1:
                c_d = cd_lowv
            elif v_temp > v_thresh2:
                c_d = cd_highv
            else:
                c_d = cd_lowv + linear_slope * (v_temp - v_thresh1)

        gam = c_d * fcor * r0 / w_cool     # non-dimensional model parameter

        d_mfracm0_drfracr0 = gam * ((mfracm0_temp - rfracr0_temp**2)**2) \
            / (1 - rfracr0_temp**2)
        mfracm0_temp = mfracm0_temp - d_mfracm0_drfracr0 * drfracr0
        rfracr0_temp = rfracr0_temp - drfracr0

        mmfracm0[mmfracm0.shape[0] - 1 - ii - 2] = mfracm0_temp

    return rrfracr0, mmfracm0


def er11_radprof_raw(vmax, r_in, rmax_or_r0, fcor, ckcd, rr_er11):
    """Emanuel & Rotunno (2011) inner solution (raw, un-adjusted)."""
    fcor = abs(fcor)
    if rmax_or_r0 != 'rmax':
        raise ValueError('rmax_or_r0 must be set to "rmax"')
    rmax = r_in
    v_er11 = (1. / rr_er11) * (vmax * rmax + .5 * fcor * rmax**2) \
        * ((2 * (rr_er11 / rmax)**2)
           / (2 - ckcd + ckcd * (rr_er11 / rmax)**2))**(1 / (2 - ckcd)) \
        - .5 * fcor * rr_er11
    v_er11[rr_er11 == 0] = 0           # make V=0 at r=0

    i_rmax = np.argwhere(v_er11 == np.max(v_er11))[0, 0]
    f = interp1d(v_er11[i_rmax + 1:], rr_er11[i_rmax + 1:],
                 fill_value='extrapolate')
    r0_profile = f(0.)
    r_out = r0_profile            # use value from profile itself
    return v_er11, r_out


def er11_radprof(vmax, r_in, rmax_or_r0, fcor, ckcd, rr_er11):
    """ER11 profile adjusted to converge to the input (rmax, Vmax)."""
    dr = rr_er11[1] - rr_er11[0]
    v_er11, r_out = er11_radprof_raw(vmax, r_in, rmax_or_r0, fcor, ckcd,
                                     rr_er11)
    if rmax_or_r0 == 'rmax':
        drin_temp = r_in - rr_er11[np.argwhere(v_er11 == np.max(v_er11))[0, 0]]
    elif rmax_or_r0 == 'r0':
        f = interp1d(v_er11[2:], rr_er11[2:])
        drin_temp = r_in - f(0)
    dvmax_temp = vmax - np.max(v_er11)

    r_in_save = copy.copy(r_in)
    vmax_save = copy.copy(vmax)

    n_iter = 0
    # NOTE: FIRST ARGUMENT MUST BE ">" NOT ">=" or else rmax values at exactly
    # dr/2 intervals (e.g. 10.5 for dr=1 km) will not converge
    while (np.abs(drin_temp) > dr / 2) or (np.abs(dvmax_temp / vmax_save) >= 10**-2):
        n_iter += 1
        if n_iter > 20:
            v_er11 = np.full(rr_er11.size, np.float64('nan'))
            r_out = np.float64('nan')
            break

        # adjust estimate of r_in according to error
        r_in = r_in + drin_temp

        # Vmax second
        vmax = vmax + dvmax_temp
        while np.abs(dvmax_temp / vmax) >= 10**-2:
            vmax = vmax + dvmax_temp
            v_er11, r_out = er11_radprof_raw(vmax, r_in, rmax_or_r0, fcor,
                                             ckcd, rr_er11)
            vmax_prof = np.max(v_er11)
            dvmax_temp = vmax_save - vmax_prof

        v_er11, r_out = er11_radprof_raw(vmax, r_in, rmax_or_r0, fcor, ckcd,
                                         rr_er11)
        vmax_prof = np.max(v_er11)
        dvmax_temp = vmax_save - vmax_prof
        if rmax_or_r0 == 'rmax':
            drin_temp = r_in_save - rr_er11[
                np.argwhere(v_er11 == vmax_prof)[0, 0]]
        elif rmax_or_r0 == 'r0':
            f = interp1d(v_er11[2:], rr_er11[2:])
            drin_temp = r_in_save - f(0)

    return v_er11, r_out


def _line_intersection(l1, l2):
    """Return (x, y) of the first intersection point, or None."""
    inter = l1.intersection(l2)
    if inter.is_empty:
        return None
    wkt0 = inter.wkt.split(' ')[0]
    if wkt0 in ('POINT',):
        return inter.coords[0]
    if wkt0 in ('MULTIPOINT', 'GEOMETRYCOLLECTION'):
        geoms = getattr(inter, 'geoms', [inter])
        for g in geoms:
            if g.geom_type == 'Point':
                return g.coords[0]
    if wkt0 in ('LINESTRING', 'MULTILINESTRING'):
        # degenerate: curves touch along a segment; take its first endpoint
        geoms = getattr(inter, 'geoms', [inter])
        return list(geoms[0].coords)[0]
    return None


def er11e04_nondim_r0input(vmax, r0, fcor, cdvary, c_d, w_cool, ckcdvary,
                           ckcd, eye_adj, alpha_eye, max_outer_iter=200):
    """CLE15 with r0 specified (FHLO's usage: r0 from wind-radii analysis).

    Converge rmax/r0 geometrically until the ER11 M/M0 curve has a tangent
    point with the E04 M/M0 curve; merge at the intersection.
    Returns (rr, VV, rmerge, Vmerge, rmax) with rr/VV in m / m s-1.
    """
    fcor = abs(fcor)
    if ckcdvary == 1:
        ckcd_coefquad = 5.5041e-04
        ckcd_coeflin = -0.0259
        ckcd_coefcnst = 0.7627
        ckcd = ckcd_coefquad * vmax**2 + ckcd_coeflin * vmax + ckcd_coefcnst
    ckcd = min(1.9, ckcd)   # capped (see original comment)

    # Step 1: E04 M/M0 vs. r/r0
    nr = 100000
    rrfracr0_e04, mmfracm0_e04 = e04_outerwind_r0input_nondim_mm0(
        r0, fcor, cdvary, c_d, w_cool, nr)
    m0_e04 = .5 * fcor * r0**2

    # Step 2: geometric bisection on rmax/r0 until the two M(r) curves touch
    count = 0
    soln_converged = 0
    while soln_converged == 0:
        count += 1
        if count > max_outer_iter:
            raise RuntimeError('ER11E04_r0input did not converge '
                               f'(Vmax={vmax:.1f}, r0={r0/1000:.0f} km)')
        rmaxr0_min, rmaxr0_max = .001, .75
        rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 2.
        rmaxr0 = rmaxr0_new
        drmaxr0 = rmaxr0_max - rmaxr0
        drmaxr0_thresh = .000001
        rfracrm_min, rfracrm_max = 0., 50.
        rmerger0 = mmergem0 = None
        iter_n = 0
        while np.abs(drmaxr0) >= drmaxr0_thresh:
            iter_n += 1
            if iter_n > 200:
                break
            rmax = rmaxr0_new * r0
            drfracrm = .01
            if rmax > 100. * 1000:
                drfracrm = drfracrm / 10.
            rrfracrm_er11 = np.arange(rfracrm_min, rfracrm_max + drfracrm,
                                      drfracrm)
            rr_er11 = rrfracrm_er11 * rmax
            vv_er11, _ = er11_radprof(vmax, rmax, 'rmax', fcor, ckcd, rr_er11)

            if not np.isnan(np.max(vv_er11)):
                rrfracr0_er11 = rr_er11 / r0
                mmfracm0_er11 = (rr_er11 * vv_er11 + .5 * fcor * rr_er11**2) \
                    / m0_e04
                l1 = LineString(list(zip(rrfracr0_e04, mmfracm0_e04)))
                l2 = LineString(list(zip(rrfracr0_er11, mmfracm0_er11)))
                hit = _line_intersection(l1, l2)
                if hit is None:       # no intersections -- rmaxr0 too small
                    drmaxr0 = np.abs(drmaxr0) / 2
                else:
                    x0, y0 = hit
                    drmaxr0 = -np.abs(drmaxr0) / 2
                    rmerger0 = np.mean(x0)
                    mmergem0 = np.mean(y0)
            else:
                # ER11 did not converge (low CkCd, high Ro) -> reduce rmax
                drmaxr0 = -abs(drmaxr0) / 2
            rmaxr0 = rmaxr0_new
            rmaxr0_new = rmaxr0_new + drmaxr0

        if (not np.isnan(np.max(vv_er11))) and (rmerger0 is not None):
            soln_converged = 1
        else:
            soln_converged = 0
            ckcd = ckcd + .1   # 'Adjusting CkCd to find convergence'

    m0 = .5 * fcor * r0**2
    mm = .5 * fcor * rmax**2 + rmax * vmax
    mmm0 = mm / m0

    # merge the two branches and interpolate onto a r/rmax grid
    ii_er11 = np.argwhere((rrfracr0_er11 < rmerger0)
                          & (mmfracm0_er11 < mmergem0))[:, 0]
    ii_e04 = np.argwhere((rrfracr0_e04 >= rmerger0)
                         & (mmfracm0_e04 >= mmergem0))[:, 0]
    mmfracm0_temp = np.hstack((mmfracm0_er11[ii_er11], mmfracm0_e04[ii_e04]))
    rrfracr0_temp = np.hstack((rrfracr0_er11[ii_er11], rrfracr0_e04[ii_e04]))

    # drfracrm = .01 keeps resolution relative to rmax -> no smoothing near rmax
    drfracrm = .01
    rfracrm_min = 0
    rfracrm_max = r0 / rmax
    rrfracrm = np.arange(rfracrm_min, rfracrm_max, drfracrm)
    f = interp1d(rrfracr0_temp * (r0 / rmax), mmfracm0_temp * (m0 / mm))
    mmfracmm = f(rrfracrm)

    rrfracr0 = rrfracrm * rmax / r0
    mmfracm0 = mmfracmm * mm / m0

    vv = (mm / rmax) * (mmfracmm / rrfracrm) - .5 * fcor * rmax * rrfracrm
    rr = rrfracrm * rmax
    vv[rr == 0] = 0

    rmerge = rmerger0 * r0
    vmerge = (m0 / r0) * ((mmergem0 / rmerger0) - rmerger0)
    _ = mmm0, mmfracm0, rrfracr0, alpha_eye, eye_adj   # kept for parity
    return rr, vv, rmerge, vmerge, rmax
