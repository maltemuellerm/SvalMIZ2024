"""
Figure: Model spread (std dev) of T2M and sea-ice thickness at analysis time.
Produces one file:
  Spread_T2M_SIT_AnalysisMap.png   (1 row × 2 columns)

Data source: AnalysisOfFields files regridded to AROME Arctic grid.
  T2M spread: 6 models — DWD-ICON, ECMWF-IFS, MF-ARPEGE, ECCC-HRDPSN,
              MET-AROMEArctic, MF-AROME
  SIT spread: 5 models — DWD-ICON, ECMWF-IFS, MF-ARPEGE, MET-AROMEArctic,
              MF-AROME (ECCC-HRDPSN has no SIT in these files)

Projection, fonts, and styling follow the manuscript figure guidelines.
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

# ── paths ─────────────────────────────────────────────────────────────────────
HERE  = pathlib.Path(__file__).parent
ROOT  = HERE.parents[1]
PAPER = ROOT.parents[1] / "papers" / "SvalMIZ24-MIP" / "figures"
APATH = pathlib.Path(
    "/lustre/storeB/project/nwp/SALIENSEAS/SvalMIZ2024/models/AnalysisOfFields/"
)

# ── projection (identical to MET AROME Arctic) ────────────────────────────────
GLOBE     = ccrs.Globe(semimajor_axis=6371000, semiminor_axis=6371000)
PLOT_PROJ = ccrs.LambertConformal(
    central_longitude=-25.0,
    central_latitude=77.5,
    standard_parallels=(77.5, 77.5),
    globe=GLOBE,
)
# Native x/y extent of the AROME Arctic domain (metres in Lambert projection)
EXTENT_LCC = [250_000, 2_150_000, -920_000, 1_490_000]

# ── common style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "savefig.dpi": 300,
})

# ── model tables ──────────────────────────────────────────────────────────────
# (filename_key, t2m_varname, sit_varname_or_None, eccc_level)
# eccc_level=True means Tair[:,1,:,:] + 273.15
MODELS = [
    ("DWD-ICON",       "2t",                 "icetk",      False),
    ("ECMWF-IFS",      "T2M",                "var98",      False),
    ("MF-ARPEGE",      "T_2M",               "H_ICE",      False),
    ("ECCC-HRDPSN",    "Tair",               None,         True),
    ("MET-AROMEArctic","air_temperature_2m",  "SFX_ICE_THK",False),
    ("MF-AROME",       "T_2M",               "H_ICE",      False),
]

AFILE_PATTERN = "{model}_04042025-30042025_AAgrid_00L.nc"

# ── data loading ──────────────────────────────────────────────────────────────

def _load_field(ds, varname, eccc_level=False):
    """Extract a 2D time-mean field from dataset, handling special cases."""
    if varname not in ds:
        return None
    data = ds[varname].values.astype(float)
    if eccc_level:
        # ECCC-HRDPSN: (time, level, y, x), level 1 is 2m, units Celsius
        data = data[:, 1, :, :]  # → (time, y, x)
        data += 273.15
    data = np.squeeze(data)
    # Remove fill values
    data = np.where(np.abs(data) > 1e10, np.nan, data)
    # For T2M: sanity check — should be in Kelvin (~200–300 K)
    return data  # shape: (time, y, x) or (y, x)


def compute_spread(variable):
    """
    Load all models for 'T2M' or 'SIT', stack arrays, return (x, y, spread_2d).
    The x/y coords are native Lambert metres from the AA grid.
    """
    stack = []
    for model, t2m_var, sit_var, eccc_level in MODELS:
        fpath = APATH / AFILE_PATTERN.format(model=model)
        if not fpath.exists():
            print(f"  Skipping {model}: file not found")
            continue

        varname = t2m_var if variable == "T2M" else sit_var
        if varname is None:
            print(f"  Skipping {model}: no {variable} variable")
            continue

        ds = xr.open_dataset(fpath)
        data = _load_field(ds, varname, eccc_level=(eccc_level and variable == "T2M"))
        ds.close()

        if data is None:
            print(f"  Skipping {model}: variable {varname} not in file")
            continue

        if data.ndim == 3:
            # (time, y, x) → time-mean
            with np.errstate(all="ignore"):
                data = np.nanmean(data, axis=0)

        # Replace negatives (unphysical SIT or bad T2M)
        if variable == "SIT":
            data = np.where(data < 0, 0.0, data)

        print(f"  {model}: {varname}, shape {data.shape}, "
              f"mean={np.nanmean(data):.3f}")
        stack.append(data)

    print(f"  → {len(stack)} models used for {variable} spread")

    if len(stack) < 2:
        return None, None, None

    # Get grid coords from last successfully loaded file
    ds = xr.open_dataset(APATH / AFILE_PATTERN.format(model=MODELS[0][0]))
    x  = ds["x"].values   # 1D, metres in Lambert
    y  = ds["y"].values   # 1D, metres in Lambert
    ds.close()

    arr   = np.stack(stack, axis=0)           # (n_models, y, x)
    spread = np.std(arr, axis=0, ddof=0)      # (y, x)

    # Zero-mask where all models are effectively zero (SIT only)
    if variable == "SIT":
        max_val = np.nanmax(np.where(np.isfinite(arr), arr, 0), axis=0)
        spread = np.where(max_val <= 1e-4, 0.0, spread)

    return x, y, spread


# ── plotting ──────────────────────────────────────────────────────────────────

def make_figure(outname="Spread_T2M_SIT_AnalysisMap.png"):
    print("\n=== T2M spread ===")
    x, y, spread_t2m = compute_spread("T2M")

    print("\n=== SIT spread ===")
    _, _, spread_sit = compute_spread("SIT")

    if spread_t2m is None or spread_sit is None:
        print("Insufficient data — aborting.")
        return

    dx = EXTENT_LCC[1] - EXTENT_LCC[0]
    dy = EXTENT_LCC[3] - EXTENT_LCC[2]
    aspect = dx / dy

    panel_w = 5.0
    panel_h = panel_w / aspect

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * panel_w + 1.0, panel_h + 1.2),
        subplot_kw={"projection": PLOT_PROJ},
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.14, wspace=0.08)

    panels = [
        (axes[0], spread_t2m, cmocean.cm.amp, 0.0, 3.5,
         "(a) T2M spread at analysis", "Spread (\u00b0C)"),
        (axes[1], spread_sit, cmocean.cm.amp, 0.0, 0.5,
         "(b) Sea-ice thickness spread at analysis", "Spread (m)"),
    ]

    meshes = []
    for ax, data, cmap, vmin, vmax, title, _ in panels:
        ax.set_facecolor("#e0e0e0")
        ax.set_extent(EXTENT_LCC, crs=PLOT_PROJ)

        mesh = ax.pcolormesh(
            x, y, data,
            transform=PLOT_PROJ,   # data is already in projection coords
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            zorder=2,
            rasterized=True,
        )
        ax.coastlines(resolution="50m", linewidth=0.75, color="0.1", zorder=5)
        ax.set_title(title, fontsize=17, linespacing=1.1, pad=5)
        meshes.append(mesh)

    # Individual colorbars per panel
    for ax, mesh, (_, _, _, _, _, _, cblabel) in zip(
        axes, meshes, panels
    ):
        # position colorbar below each panel
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, 0.06, pos.width, 0.03])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
        cb.set_label(cblabel, fontsize=15)
        cb.ax.tick_params(labelsize=14)

    outfile = HERE / outname
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy(outfile, PAPER / outname)
    print(f"\nSaved and copied: {outname}")


if __name__ == "__main__":
    make_figure()
