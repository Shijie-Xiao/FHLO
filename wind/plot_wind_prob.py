"""Plot FHLO wind exceedance probability maps (34/50/64-kt panels).

Style follows the FHLO paper and this repo's tracks/plot_tracks.py
conventions: cartopy coastlines + land/ocean background, probability fill
(transparent where 0) on top, gray member spaghetti + bold blue
ensemble-mean track + black IBTrACS best track.
"""

import numpy as np


def plot_prob(ds, aux, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    thr = ds['threshold_kt'].values
    n = len(thr)
    fig, axes = plt.subplots(
        1, n, figsize=(5.6 * n, 5.0),
        subplot_kw={'projection': proj}, constrained_layout=True)
    if n == 1:
        axes = [axes]

    lon_lo, lon_hi = float(ds['lon'].min()), float(ds['lon'].max())
    lat_lo, lat_hi = float(ds['lat'].min()), float(ds['lat'].max())
    inputs = aux.get('inputs')

    # ensemble tracks: gray spaghetti + blue bold mean (plot_tracks style)
    tracks = []
    mlat = mlon = None
    if inputs is not None:
        lats, lons = inputs['lats'], inputs['lons']
        ki = aux.get('ki')
        if ki is not None and len(ki):
            for m in range(min(len(lats), 60)):
                tracks.append((lons[m, ki], lats[m, ki]))
            mlon = np.nanmean(lons[:, ki], axis=0)
            mlat = np.nanmean(lats[:, ki], axis=0)

    # IBTrACS best track over the window (black, plot_tracks convention)
    bt = aux.get('best_track')

    for i, ax in enumerate(axes):
        p = ds['wind_exceedance_prob'].isel(threshold_kt=i)
        # background map first
        ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor='0.98',
                       zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='0.92',
                       zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.7,
                       color='0.25', zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale('110m'), lw=0.4,
                       color='0.5', zorder=1)
        # probability fill; mask <5% as no-signal (NHC convention) so the
        # single-member noise doesn't paint a huge faint-yellow blob over
        # the map -- transparent there, colorbar floor at 5%
        parr = np.ma.masked_less(p.values * 100, 5.0)
        cmap = plt.get_cmap('YlOrRd').copy()
        cmap.set_bad(alpha=0.0)
        pcm = ax.pcolormesh(ds['lon'], ds['lat'], parr,
                            cmap=cmap, vmin=5.0, vmax=100,
                            shading='auto', zorder=2)
        cs = ax.contour(ds['lon'], ds['lat'], p.values * 100,
                        levels=[10, 50], colors='k', linewidths=0.7,
                        zorder=4)
        ax.clabel(cs, fmt='%d%%', fontsize=6)
        for lo, la in tracks:
            ax.plot(lo, la, color='0.55', lw=0.3, alpha=0.6, zorder=3)
        if mlon is not None:
            ax.plot(mlon, mlat, color='#1f77b4', lw=1.8, zorder=5,
                    label='Ensemble mean')
        if bt is not None:
            ax.plot(bt[0], bt[1], 'k-', lw=2.2, zorder=6,
                    label='Best track (IBTrACS)')
            ax.plot(bt[0][0], bt[1][0], 'k*', ms=11, zorder=7)
        if mlon is not None:
            ax.plot(mlon[0], mlat[0], 'o', color='#d62728', ms=5, zorder=7,
                    label='Init')
        ax.set_extent([lon_lo, lon_hi, lat_lo, lat_hi], crs=proj)
        gl = ax.gridlines(draw_labels=True, lw=0.4, color='0.6',
                          linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(f'{int(thr[i])}-kt wind probability (%)', fontsize=11)
        cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label('probability (%)')

    fig.suptitle(f"{ds.attrs['storm']}  FHLO wind probability  "
                 f"init {ds.attrs['fc_start']}  "
                 f"{ds.attrs['window_h']:.0f} h  N={ds.attrs['n_members']}  "
                 f"r0={ds.attrs['r0_km']:.0f} km  k={ds.attrs['shape_k']:.2f}",
                 fontsize=12)
    fig.savefig(out_png, dpi=150)
    if out_png.suffix == '.png':
        fig.savefig(str(out_png).replace('.png', '.svg'))
    plt.close(fig)
