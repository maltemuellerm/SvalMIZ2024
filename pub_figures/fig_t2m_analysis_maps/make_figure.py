"""
Figure: 2-m air temperature RMSE and BIAS against model analysis
Produces TWO separate files:
  T2M_RMSE_AnalysisMap.png  (4 rows × 2 columns)
  T2M_BIAS_AnalysisMap.png  (4 rows × 2 columns)
Projection: Lambert Conformal Conic — same as MET AROME Arctic
            (domain appears as a proper rectangle with no distortion)
Values plotted on land AND ocean; only coastlines drawn for reference.
Models in manuscript Table 1 order (Category A, row-major 4×2).
"""

import matplotlib
matplotlib.use("Agg")

import pathlib, shutil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

# ── paths ────────────────────────────────────────────────────────────────────
HERE  = pathlib.Path(__file__).parent
ROOT  = HERE.parents[1]
PAPER = ROOT.parent / "papers" / "SvalMIZ24-MIP" / "figures"
MPATH = pathlib.Path("/lustre/storeB/project/nwp/SALIENSEAS/SvalMIZ2024/models/")

# ── projection ───────────────────────────────────────────────────────────────
# Lambert Conformal Conic identical to MET AROME Arctic (grid_mapping_name =
# lambert_conformal_conic, standard_parallel=77.5, central_meridian=-25).
# With this projection the AROME Arctic domain is exactly rectangular.
GLOBE     = ccrs.Globe(semimajor_axis=6371000, semiminor_axis=6371000)
PLOT_PROJ = ccrs.LambertConformal(
    central_longitude=-25.0,
    central_latitude=77.5,
    standard_parallels=(77.5, 77.5),
    globe=GLOBE,
)
DATA_PROJ = ccrs.PlateCarree()

# Display extent in Lambert native metres — taken from the AROME Arctic file's
# actual x/y range (x: 278620..2123621, y: -897985..1472014), padded slightly.
EXTENT_LCC = [250_000, 2_150_000, -920_000, 1_490_000]  # [xmin,xmax,ymin,ymax]

# ── model table (manuscript Table 1 order, Category A) ──────────────────────
MODELS = [
    ("DWD-ICON",           "2t",                 "lonlat"),
    ("ECMWF-IFS",          "T2M",                "lonlat"),
    ("MF-ARPEGE",          "T_2M",               "arpege"),
    ("ECCC-HRDPSN",        "Tair",               "eccc"),
    ("ECCC-HRDPSNcoupled", "Tair",               "eccc"),
    ("MET_AROMEArctic",    "air_temperature_2m", "lonlat2"),
    ("MF-AROME",           "T_2M",               "arome"),
    ("ECMWF-AIFS",         "2t",                 "lonlat"),
]
MODEL_LABELS = [
    "DWD-ICON", "ECMWF-IFS", "MF-ARPEGE", "ECCC-HRDPSN",
    "ECCC-HRDPSN\n(coupled)", "MET-AROME", "MF-AROME", "ECMWF-AIFS",
]

# ── colour config ─────────────────────────────────────────────────────────────
CMAP_RMSE = cmocean.cm.amp        # white at 0 → deep red
VMIN_RMSE, VMAX_RMSE = 0.0, 4.0

CMAP_BIAS = cmocean.cm.balance    # blue – white – red
VMIN_BIAS, VMAX_BIAS = -3.0, 3.0
NORM_BIAS = TwoSlopeNorm(vcenter=0, vmin=VMIN_BIAS, vmax=VMAX_BIAS)

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
    raw = np.where(np.abs(raw) > 1e10, np.nan, raw)

    if coord_style == "lonlat":
        lon, lat = fc["lon"].values, fc["lat"].values
    elif coord_style == "lonlat2":
        lon, lat = fc["longitude"].values, fc["latitude"].values
    elif coord_style in ("arpege", "arome"):
        tag  = "ARPEGE" if coord_style == "arpege" else "AROME"
        cfil = next(MPATH.joinpath(model_dir).glob(f"{tag}_SVALBARD_*.nc"), None)
        if cfil is None: fc.close(); return None
        cf   = xr.open_dataset(cfil)
        lon, lat = cf["Longitude"].values, cf["Latitude"].values
        cf.close()
    elif coord_style == "eccc":
        cfil = MPATH / model_dir / "2024040100.nc.000002"
        if not cfil.exists(): fc.close(); return None
        cf  = xr.open_dataset(cfil)
        lat = cf["nav_lat"].values
        lon = cf["nav_lon"].values
        arr = fc[varname]
        raw = (arr.isel(time_counter=0, z=1)
               if ("time_counter" in arr.dims and "z" in arr.dims)
               else arr).values.squeeze().astype(float)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        cf.close()
    else:
        lon, lat = fc["lon"].values, fc["lat"].values
    fc.close()

    # clip 1-D regular grids
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


# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      9,
    "axes.titlesize": 11,
    "savefig.dpi":    300,
})

NROWS, NCOLS = 4, 2


def make_panel_figure(metric, cmap, vmin, vmax, norm, cbar_label, outname):
    """Create and save a 4×2 panel figure for one metric."""
    dx = EXTENT_LCC[1] - EXTENT_LCC[0]   # 1 900 000 m
    dy = EXTENT_LCC[3] - EXTENT_LCC[2]   # 2 410 000 m
    aspect  = dx / dy                      # ≈ 0.79
    panel_w = 3.8
    panel_h = panel_w / aspect             # ≈ 4.8 inches
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

        # light grey background (visible where a model has no data coverage)
        ax.set_facecolor("#e0e0e0")
        ax.set_extent(EXTENT_LCC, crs=PLOT_PROJ)

        # lightweight lat/lon reference grid (no shapely overhead)
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
            print(f"shape={da.shape}  [{np.nanmin(da):.1f}, {np.nanmax(da):.1f}]")
            pkw = dict(transform=DATA_PROJ, cmap=cmap,
                       shading="auto", zorder=2, rasterized=True)
            if norm is not None:
                pkw["norm"] = norm
            else:
                pkw["vmin"] = vmin
                pkw["vmax"] = vmax
            mesh = ax.pcolormesh(lo, la, da, **pkw)
            last_mesh = mesh
        else:
            print("no data")
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.5")

        # coastlines on top — no land masking, data shown everywhere
        ax.coastlines(resolution="50m", linewidth=0.65, color="0.1", zorder=5)

        ax.set_title(label, fontsize=11, pad=3, linespacing=1.2)

        # panel letter
        ax.text(0.02, 0.97, f"({chr(97+idx)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                color="k", bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

    # shared horizontal colourbar at the bottom
    if last_mesh is not None:
        p_left = axes[-1, 0].get_position()
        p_right = axes[-1, -1].get_position()
        cax = fig.add_axes([
            p_left.x0,
            0.03,
            p_right.x1 - p_left.x0,
            0.015,
        ])
        cb = fig.colorbar(last_mesh, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        if norm is not None:
            cb.set_ticks(np.linspace(vmin, vmax, 7))

    title = ("2-m air temperature RMSE — April 2024"
             if metric == "rmse" else
             "2-m air temperature Bias — April 2024")
    fig.suptitle(title, fontsize=11, y=0.985)

    outfile = HERE / outname
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved  → {outfile}")
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy(outfile, PAPER / outname)
    print(f"Copied → {PAPER / outname}")


# ── generate both figures ─────────────────────────────────────────────────────
print("\n=== RMSE ===")
make_panel_figure("rmse", CMAP_RMSE, VMIN_RMSE, VMAX_RMSE, None,
                  "RMSE (°C)", "T2M_RMSE_AnalysisMap.png")

print("\n=== BIAS ===")
make_panel_figure("bias", CMAP_BIAS, VMIN_BIAS, VMAX_BIAS, NORM_BIAS,
                  "Bias (°C)", "T2M_BIAS_AnalysisMap.png")
