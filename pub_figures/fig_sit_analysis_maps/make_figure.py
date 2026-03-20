"""
Figure: Sea-ice thickness RMSE and Mean against model analysis
Produces TWO separate files:
  SIT_RMSE_AnalysisMap.png  (3 rows × 2 columns)
  SIT_Mean_AnalysisMap.png  (3 rows × 2 columns)
Projection: Lambert Conformal Conic — same as MET AROME Arctic
Models: Category B sea-ice prediction systems, manuscript Table 1 order.
Style: matches fig_t2m_analysis_maps (fontsize, projection, layout).
"""

import matplotlib
matplotlib.use("Agg")

import pathlib, shutil
import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ── paths ────────────────────────────────────────────────────────────────────
HERE  = pathlib.Path(__file__).parent
ROOT  = HERE.parents[1]
PAPER = ROOT.parents[1] / "papers" / "SvalMIZ24-MIP" / "figures"
MPATH = pathlib.Path("/lustre/storeB/project/nwp/SALIENSEAS/SvalMIZ2024/models/")

# ── projection (same as T2M maps — AROME Arctic Lambert Conformal Conic) ────
GLOBE     = ccrs.Globe(semimajor_axis=6371000, semiminor_axis=6371000)
PLOT_PROJ = ccrs.LambertConformal(
    central_longitude=-25.0,
    central_latitude=77.5,
    standard_parallels=(77.5, 77.5),
    globe=GLOBE,
)
DATA_PROJ = ccrs.PlateCarree()

# Same extent as T2M figures (AROME Arctic native x/y range, padded slightly)
EXTENT_LCC = [250_000, 2_150_000, -920_000, 1_490_000]  # [xmin,xmax,ymin,ymax]

# ── model table (manuscript Table 1 order, Category B — sea-ice models) ─────
# coord_style: 'lonlat'  → fc.lon / fc.lat
#              'lonlat2' → fc.longitude / fc.latitude
MODELS = [
    ("MET-TOPAZ5",         "sithick",       "lonlat2"),
    ("NERSC-NextSIM",      "sithick",       "lonlat2"),
    ("NRL-GOFS",           "hi",            "lonlat"),
    ("MET-ROMS",           "ice_thickness", "lonlat"),
    ("ECCC-RIOPS",         "iicevol",       "lonlat2"),
    ("ECCC-RIOPScoupled",  "iicevol",       "lonlat2"),
]
MODEL_LABELS = [
    "MET-TOPAZ", "NERSC-neXtSIM",
    "NRL-GOFS",  "MET-ROMS",
    "ECCC-RIOPS", "ECCC-RIOPS\n(coupled)",
]

# ── colour config ─────────────────────────────────────────────────────────────
# RMSE: white at 0 → deep red (same family as T2M RMSE, capped at 0.5 m)
CMAP_RMSE = cmocean.cm.amp
VMIN_RMSE, VMAX_RMSE = 0.0, 0.5   # metres

# Mean SIT: sequential blue-green (tempo), 0–3 m
CMAP_MEAN = cmocean.cm.tempo
VMIN_MEAN, VMAX_MEAN = 0.0, 3.0   # metres

# ── data loading ──────────────────────────────────────────────────────────────
def load_field(model_dir, varname, coord_style, metric, max_pts=400):
    """Return (lon, lat, data) clipped and stride-thinned, or None."""
    lnmin, lnmax = -30., 100.
    ltmin, ltmax =  58.,  93.

    ifile = MPATH / model_dir / f"48h_{metric}.{model_dir}_202404.nc"
    if not ifile.exists():
        return None

    fc  = xr.open_dataset(ifile)
    raw = np.squeeze(fc[varname].values).astype(float)
    # mask fill values / unphysical negatives
    raw = np.where((np.abs(raw) > 1e10) | (raw < 0), np.nan, raw)

    if coord_style == "lonlat":
        lon, lat = fc["lon"].values, fc["lat"].values
    elif coord_style == "lonlat2":
        lon, lat = fc["longitude"].values, fc["latitude"].values
    else:
        lon, lat = fc["lon"].values, fc["lat"].values
    fc.close()

    # clip 1-D regular grids to region
    if lon.ndim == 1 and lat.ndim == 1:
        lon_w = lon.copy(); lon_w[lon_w > 180] -= 360
        li = np.where((lon_w >= lnmin) & (lon_w <= lnmax))[0]
        lj = np.where((lat   >= ltmin) & (lat   <= ltmax))[0]
        if li.size > 0 and lj.size > 0:
            lon = lon_w[li]; lat = lat[lj]
            raw = raw[np.ix_(lj, li)]

    # stride-thin
    sy = max(1, raw.shape[0] // max_pts)
    sx = max(1, raw.shape[1] // max_pts)
    raw = raw[::sy, ::sx]
    lon = lon[::sx] if lon.ndim == 1 else lon[::sy, ::sx]
    lat = lat[::sy] if lat.ndim == 1 else lat[::sy, ::sx]

    return lon, lat, raw


# ── style (identical to T2M figures) ─────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      12,
    "axes.titlesize": 17,
    "savefig.dpi":    300,
})

NROWS, NCOLS = 3, 2


def make_panel_figure(metric, cmap, vmin, vmax, cbar_label, outname):
    """Create and save a 3×2 panel figure for one metric."""
    dx = EXTENT_LCC[1] - EXTENT_LCC[0]
    dy = EXTENT_LCC[3] - EXTENT_LCC[2]
    aspect  = dx / dy
    panel_w = 3.8
    panel_h = panel_w / aspect
    fig_w   = NCOLS * panel_w + 1.0
    fig_h   = NROWS * panel_h + 0.6

    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": PLOT_PROJ},
        constrained_layout=False,
    )
    plt.subplots_adjust(
        left=0.04, right=0.98, top=0.96, bottom=0.07,
        wspace=0.04, hspace=0.08,
    )

    last_mesh = None

    for idx, ((model_dir, varname, cstyle), label) in enumerate(
        zip(MODELS, MODEL_LABELS)
    ):
        row, col = divmod(idx, NCOLS)
        ax = axes[row, col]

        ax.set_facecolor("#e0e0e0")
        ax.set_extent(EXTENT_LCC, crs=PLOT_PROJ)

        # lightweight lat/lon reference grid
        for xl in [-10, 0, 10, 20, 30]:
            ax.plot([xl, xl], [60, 92], transform=DATA_PROJ,
                    color="0.6", lw=0.3, ls="--", alpha=0.6, zorder=1)
        for yl in [70, 75, 80, 85]:
            lv = np.linspace(-30, 100, 200)
            ax.plot(lv, np.full_like(lv, yl), transform=DATA_PROJ,
                    color="0.6", lw=0.3, ls="--", alpha=0.6, zorder=1)

        print(f"  {model_dir:25s}", end="  ")
        res = load_field(model_dir, varname, cstyle, metric)

        if res is not None:
            lo, la, da = res
            print(f"shape={da.shape}  [{np.nanmin(da):.3f}, {np.nanmax(da):.3f}]")
            mesh = ax.pcolormesh(lo, la, da,
                                 transform=DATA_PROJ, cmap=cmap,
                                 vmin=vmin, vmax=vmax,
                                 shading="auto", zorder=2, rasterized=True)
            last_mesh = mesh
        else:
            print("no data")
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.5")

        # coastlines on top — values shown on land AND ocean
        ax.coastlines(resolution="50m", linewidth=0.65, color="0.1", zorder=5)

        ax.set_title(label, fontsize=17, pad=5, linespacing=1.1)

        # panel letter (a)(b)…
        ax.text(0.02, 0.97, f"({chr(97+idx)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=18, fontweight="black",
                color="k", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    # shared horizontal colourbar at the bottom
    if last_mesh is not None:
        p_left  = axes[-1, 0].get_position()
        p_right = axes[-1, -1].get_position()
        cax = fig.add_axes([p_left.x0, 0.03,
                            p_right.x1 - p_left.x0, 0.015])
        cb = fig.colorbar(last_mesh, cax=cax,
                          orientation="horizontal", extend="max")
        cb.set_label(cbar_label, fontsize=15)
        cb.ax.tick_params(labelsize=14)

    outfile = HERE / outname
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved  → {outfile}")
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy(outfile, PAPER / outname)
    print(f"Copied → {PAPER / outname}")


# ── generate both figures ─────────────────────────────────────────────────────
print("\n=== SIT RMSE ===")
make_panel_figure("rmse", CMAP_RMSE, VMIN_RMSE, VMAX_RMSE,
                  "Sea-ice thickness RMSE (m)", "SIT_RMSE_AnalysisMap.png")

print("\n=== SIT Mean ===")
make_panel_figure("mean", CMAP_MEAN, VMIN_MEAN, VMAX_MEAN,
                  "Mean sea-ice thickness (m)", "SIT_Mean_AnalysisMap.png")
