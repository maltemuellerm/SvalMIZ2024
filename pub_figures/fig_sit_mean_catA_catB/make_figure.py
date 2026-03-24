"""
Figure: Mean sea-ice thickness by model category + CryoSat-2/SMOS reference
Produces three files:
  SIT_Mean_CatA_AnalysisMap.png   (4 rows x 2 columns)
  SIT_Mean_CatB_AnalysisMap.png   (3 rows x 2 columns)
  SIT_Mean_CryoSMOS_AWI.png       (single panel)

Projection, fonts, and styling follow the latest manuscript figure guidelines.
"""

import matplotlib
matplotlib.use("Agg")

import pathlib
import shutil

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

# Paths
HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
PAPER = ROOT.parents[1] / "papers" / "SvalMIZ24-MIP" / "figures"
MPATH = pathlib.Path("/lustre/storeB/project/nwp/SALIENSEAS/SvalMIZ2024/models/")
OPATH = pathlib.Path("/lustre/storeB/project/nwp/SALIENSEAS/SvalMIZ2024/observations/remotesensing/")

# Lambert projection (same as AROME Arctic)
GLOBE = ccrs.Globe(semimajor_axis=6371000, semiminor_axis=6371000)
PLOT_PROJ = ccrs.LambertConformal(
    central_longitude=-25.0,
    central_latitude=77.5,
    standard_parallels=(77.5, 77.5),
    globe=GLOBE,
)
DATA_PROJ = ccrs.PlateCarree()
EXTENT_LCC = [250_000, 2_150_000, -920_000, 1_490_000]

# Common style (same as latest T2M/SIT maps)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 17,
    "savefig.dpi": 300,
})

# Category A (notebook-based ordering)
# tuple: (model_dir, variable_name_or_None, coord_style, panel_label)
CAT_A = [
    ("ECMWF-IFS", "var98", "lonlat", "ECMWF-IFS"),
    ("DWD-ICON", "icetk", "lonlat", "DWD-ICON"),
    ("MF-ARPEGE", "H_ICE", "arpege", "MF-ARPEGE"),
    ("MF-AROME", "H_ICE", "arome", "MF-AROME"),
    ("MET_AROMEArctic", "SFX_ICE_THK", "lonlat2", "MET-AROME"),
    # SIT proxy mapping requested by project workflow:
    # ECCC-HRDPSN  -> ECCC-RIOPS, ECCC-CAPS -> ECCC-RIOPScoupled.
    ("ECCC-RIOPS", "iicevol", "lonlat2", "ECCC-HRDPSN"),
    ("ECCC-RIOPScoupled", "iicevol", "lonlat2", "ECCC-CAPS"),
    # Last panel: CryoSat-2/SMOS reference observation
    ("__CRYOSMOS__", None, None, "CryoSat-2/SMOS"),
]

# Category B (notebook-based ordering)
CAT_B = [
    ("ECCC-RIOPS", "iicevol", "lonlat2", "ECCC-RIOPS"),
    ("ECCC-RIOPScoupled", "iicevol", "lonlat2", "ECCC-CAPS"),
    ("NRL-GOFS", "hi", "lonlat", "NRL-GOFS"),
    ("NERSC-NextSIM", "sithick", "lonlat2", "NERSC-neXtSIM"),
    ("MET-TOPAZ5", "sithick", "lonlat2", "MET-TOPAZ"),
    ("MET-ROMS", "ice_thickness", "lonlat", "MET-ROMS"),
]


# Use same color scale as previous SIT-mean figure
CMAP_MEAN = cmocean.cm.tempo
VMIN_MEAN, VMAX_MEAN = 0.0, 3.0


def _open_coords_from_aux(model_dir, coord_style):
    """Return lon, lat arrays for AROME/ARPEGE style files."""
    if coord_style == "arome":
        cfile = MPATH / model_dir / "AROME_SVALBARD_202404210000_202404221100.nc"
    else:
        cfile = MPATH / model_dir / "ARPEGE_SVALBARD_202404210000_202404221100.nc"
    if not cfile.exists():
        return None, None
    ds = xr.open_dataset(cfile)
    lon = ds["Longitude"].values
    lat = ds["Latitude"].values
    ds.close()
    return lon, lat


def load_mean_field(model_dir, varname, coord_style, max_pts=420):
    """Load 48h mean SIT field; return (lon, lat, data) or None."""
    if varname is None:
        return None

    ifile = MPATH / model_dir / f"48h_mean.{model_dir}_202404.nc"
    if not ifile.exists():
        return None

    fc = xr.open_dataset(ifile)
    if varname not in fc:
        fc.close()
        return None

    data = np.squeeze(fc[varname].values).astype(float)
    data = np.where((np.abs(data) > 1e10) | (data < 0), np.nan, data)

    if coord_style == "lonlat":
        lon, lat = fc["lon"].values, fc["lat"].values
    elif coord_style == "lonlat2":
        lon, lat = fc["longitude"].values, fc["latitude"].values
    elif coord_style in ("arome", "arpege"):
        lon, lat = _open_coords_from_aux(model_dir, coord_style)
        if lon is None:
            fc.close()
            return None
    elif coord_style == "eccc":
        cfile = MPATH / model_dir / "2024040100.nc.000002"
        if cfile.exists():
            cfc = xr.open_dataset(cfile)
            lon = cfc["nav_lon"].values
            lat = cfc["nav_lat"].values
            cfc.close()
        else:
            fc.close()
            return None
    else:
        lon, lat = fc["lon"].values, fc["lat"].values

    fc.close()

    # region clipping for regular grids
    if lon.ndim == 1 and lat.ndim == 1:
        lon_w = lon.copy()
        lon_w[lon_w > 180] -= 360
        li = np.where((lon_w >= -30.0) & (lon_w <= 100.0))[0]
        lj = np.where((lat >= 58.0) & (lat <= 93.0))[0]
        if li.size > 0 and lj.size > 0:
            lon = lon_w[li]
            lat = lat[lj]
            data = data[np.ix_(lj, li)]

    # thin for speed
    sy = max(1, data.shape[0] // max_pts)
    sx = max(1, data.shape[1] // max_pts)
    data = data[::sy, ::sx]
    lon = lon[::sx] if lon.ndim == 1 else lon[::sy, ::sx]
    lat = lat[::sy] if lat.ndim == 1 else lat[::sy, ::sx]

    return lon, lat, data


def draw_common_map_decor(ax):
    ax.set_facecolor("#e0e0e0")
    ax.set_extent(EXTENT_LCC, crs=PLOT_PROJ)

    for xl in [-10, 0, 10, 20, 30]:
        ax.plot([xl, xl], [60, 92], transform=DATA_PROJ,
                color="0.6", lw=0.3, ls="--", alpha=0.6, zorder=1)
    for yl in [70, 75, 80, 85]:
        lv = np.linspace(-30, 100, 200)
        ax.plot(lv, np.full_like(lv, yl), transform=DATA_PROJ,
                color="0.6", lw=0.3, ls="--", alpha=0.6, zorder=1)


def make_category_figure(category, nrows, ncols, outname):
    dx = EXTENT_LCC[1] - EXTENT_LCC[0]
    dy = EXTENT_LCC[3] - EXTENT_LCC[2]
    aspect = dx / dy
    panel_w = 3.8
    panel_h = panel_w / aspect
    fig_w = ncols * panel_w + 1.0
    fig_h = nrows * panel_h + 0.6

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": PLOT_PROJ},
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.07,
                        wspace=0.04, hspace=0.08)

    axes = np.atleast_2d(axes)
    last_mesh = None

    for idx, (model_dir, varname, cstyle, label) in enumerate(category):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        draw_common_map_decor(ax)

        print(f"  {model_dir:25s}", end="  ")

        # Special case: CryoSat-2/SMOS reference panel
        if model_dir == "__CRYOSMOS__":
            thickfile = OPATH / "CryoSMOS_icethickness_April2024.nc"
            if thickfile.exists():
                ds = xr.open_dataset(thickfile)
                data = np.squeeze(ds["analysis_sea_ice_thickness"].values).astype(float)
                data = np.where((np.abs(data) > 1e10) | (data < 0), np.nan, data)
                lon = ds["lon"].values
                lat = ds["lat"].values
                ds.close()
                print(f"shape={data.shape}  [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
                last_mesh = ax.pcolormesh(
                    lon, lat, data,
                    transform=DATA_PROJ,
                    cmap=CMAP_MEAN,
                    vmin=VMIN_MEAN,
                    vmax=VMAX_MEAN,
                    shading="auto",
                    zorder=2,
                    rasterized=True,
                )
            else:
                print("CryoSMOS file missing")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="0.5")
        else:
            loaded = load_mean_field(model_dir, varname, cstyle)
            if loaded is None:
                print("no data")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="0.5")
            else:
                lon, lat, data = loaded
                print(f"shape={data.shape}  [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
                last_mesh = ax.pcolormesh(
                    lon, lat, data,
                    transform=DATA_PROJ,
                    cmap=CMAP_MEAN,
                    vmin=VMIN_MEAN,
                    vmax=VMAX_MEAN,
                    shading="auto",
                    zorder=2,
                    rasterized=True,
                )

        ax.coastlines(resolution="50m", linewidth=0.65, color="0.1", zorder=5)
        ax.set_title(label, fontsize=17, pad=5, linespacing=1.1)
        ax.text(0.02, 0.97, f"({chr(97 + idx)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=18, fontweight="black",
                color="k", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    # hide unused axes (if any)
    for k in range(len(category), nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    if last_mesh is not None:
        p_left = axes[-1, 0].get_position()
        p_right = axes[-1, -1].get_position()
        cax = fig.add_axes([p_left.x0, 0.03, p_right.x1 - p_left.x0, 0.015])
        cb = fig.colorbar(last_mesh, cax=cax, orientation="horizontal", extend="max")
        cb.set_label("Mean sea-ice thickness (m)", fontsize=15)
        cb.ax.tick_params(labelsize=14)

    outfile = HERE / outname
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy(outfile, PAPER / outname)
    print(f"Saved and copied: {outname}")


def make_cryosmos_figure(outname="SIT_Mean_CryoSMOS_AWI.png"):
    thickfile = OPATH / "CryoSMOS_icethickness_April2024.nc"
    if not thickfile.exists():
        print(f"CryoSMOS file missing: {thickfile}")
        return

    dx = EXTENT_LCC[1] - EXTENT_LCC[0]
    dy = EXTENT_LCC[3] - EXTENT_LCC[2]
    aspect = dx / dy
    panel_w = 5.0
    panel_h = panel_w / aspect

    fig, ax = plt.subplots(
        1, 1,
        figsize=(panel_w + 0.6, panel_h + 0.9),
        subplot_kw={"projection": PLOT_PROJ},
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.14)

    draw_common_map_decor(ax)

    ds = xr.open_dataset(thickfile)
    data = np.squeeze(ds["analysis_sea_ice_thickness"].values).astype(float)
    data = np.where((np.abs(data) > 1e10) | (data < 0), np.nan, data)
    lon = ds["lon"].values
    lat = ds["lat"].values

    mesh = ax.pcolormesh(
        lon, lat, data,
        transform=DATA_PROJ,
        cmap=CMAP_MEAN,
        vmin=VMIN_MEAN,
        vmax=VMAX_MEAN,
        shading="auto",
        zorder=2,
        rasterized=True,
    )
    ds.close()

    ax.coastlines(resolution="50m", linewidth=0.75, color="0.1", zorder=5)
    ax.set_title("CryoSat-2/SMOS (AWI)", fontsize=17, pad=5)

    cax = fig.add_axes([0.12, 0.06, 0.76, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Mean sea-ice thickness (m)", fontsize=15)
    cb.ax.tick_params(labelsize=14)

    outfile = HERE / outname
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy(outfile, PAPER / outname)
    print(f"Saved and copied: {outname}")


def _wrap_lon(lon):
    """Wrap longitudes to [-180, 180]."""
    out = np.array(lon, dtype=float, copy=True)
    out[out > 180] -= 360
    return out


def _interp_to_target_grid(lon_src, lat_src, data_src, lon_tgt, lat_tgt):
    """Interpolate one model field onto target lon/lat grid."""
    # Build source point cloud
    if lon_src.ndim == 1 and lat_src.ndim == 1:
        lon2, lat2 = np.meshgrid(_wrap_lon(lon_src), lat_src)
    else:
        lon2, lat2 = _wrap_lon(lon_src), lat_src

    vals = np.asarray(data_src, dtype=float)
    mask = np.isfinite(vals)
    if np.count_nonzero(mask) < 10:
        return np.full_like(lon_tgt, np.nan, dtype=float)

    points = np.column_stack((lon2[mask], lat2[mask]))
    values = vals[mask]

    # Conservative interpolation: linear only.
    # We deliberately avoid nearest-neighbour fill to prevent artificial
    # non-zero values in no-ice ocean areas.
    target = (lon_tgt, lat_tgt)
    out = griddata(points, values, target, method="linear")
    return out


def compute_spread_on_cryosmos_grid(category):
    """Compute model spread (std dev) of mean SIT on the CryoSMOS grid."""
    thickfile = OPATH / "CryoSMOS_icethickness_April2024.nc"
    if not thickfile.exists():
        return None

    ds_ref = xr.open_dataset(thickfile)
    lon_tgt = _wrap_lon(ds_ref["lon"].values)
    lat_tgt = ds_ref["lat"].values
    ds_ref.close()

    fields = []
    used = []
    for model_dir, varname, cstyle, label in category:
        loaded = load_mean_field(model_dir, varname, cstyle)
        if loaded is None:
            continue
        lon_src, lat_src, data_src = loaded
        regridded = _interp_to_target_grid(lon_src, lat_src, data_src, lon_tgt, lat_tgt)
        fields.append(regridded)
        used.append(label)

    if len(fields) < 2:
        return None

    stack = np.stack(fields, axis=0)

    # Require at least two contributing models at each grid point.
    n_valid = np.sum(np.isfinite(stack), axis=0)
    spread = np.nanstd(stack, axis=0)
    spread = np.where(n_valid >= 2, spread, np.nan)

    # If all available model values are essentially zero, force spread to zero.
    # This removes tiny interpolation/rounding artefacts in open-ocean no-ice areas.
    eps = 1e-4
    abs_stack = np.where(np.isfinite(stack), np.abs(stack), -1.0)
    max_abs = np.max(abs_stack, axis=0)
    all_zero = (n_valid > 0) & (max_abs >= 0.0) & (max_abs <= eps)
    spread = np.where(all_zero, 0.0, spread)

    return lon_tgt, lat_tgt, spread, used


def make_spread_figure(category, title, outname):
    """Create a single-panel spread map on CryoSMOS grid for a category."""
    result = compute_spread_on_cryosmos_grid(category)
    if result is None:
        print(f"Spread unavailable: {outname}")
        return

    lon, lat, spread, used = result
    print(f"Spread models used ({len(used)}): {', '.join(used)}")

    p95 = float(np.nanpercentile(spread, 95)) if np.isfinite(spread).any() else 0.5
    vmax = max(0.2, min(2.5, p95))

    dx = EXTENT_LCC[1] - EXTENT_LCC[0]
    dy = EXTENT_LCC[3] - EXTENT_LCC[2]
    aspect = dx / dy
    panel_w = 5.0
    panel_h = panel_w / aspect

    fig, ax = plt.subplots(
        1, 1,
        figsize=(panel_w + 0.6, panel_h + 0.9),
        subplot_kw={"projection": PLOT_PROJ},
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.14)

    draw_common_map_decor(ax)

    mesh = ax.pcolormesh(
        lon, lat, spread,
        transform=DATA_PROJ,
        cmap=cmocean.cm.amp,
        vmin=0.0,
        vmax=vmax,
        shading="auto",
        zorder=2,
        rasterized=True,
    )

    ax.coastlines(resolution="50m", linewidth=0.75, color="0.1", zorder=5)
    ax.set_title(title, fontsize=17, pad=5)

    cax = fig.add_axes([0.12, 0.06, 0.76, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Spread of mean sea-ice thickness (m)", fontsize=15)
    cb.ax.tick_params(labelsize=14)

    outfile = HERE / outname
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy(outfile, PAPER / outname)
    print(f"Saved and copied: {outname}")


if __name__ == "__main__":
    print("\n=== Category A mean SIT ===")
    make_category_figure(CAT_A, nrows=4, ncols=2, outname="SIT_Mean_CatA_AnalysisMap.png")

    print("\n=== Category B mean SIT ===")
    make_category_figure(CAT_B, nrows=3, ncols=2, outname="SIT_Mean_CatB_AnalysisMap.png")

    print("\n=== CryoSat-2/SMOS (AWI) mean SIT ===")
    make_cryosmos_figure("SIT_Mean_CryoSMOS_AWI.png")
