"""
Figure: 2-m air temperature vs model (leadtime 36-48 hours) — SIC positive
Manuscript section: 4.1 – Temperature along buoy trajectories
Leads: 36-48 hours | Condition: SIC_diff > 0.1
"""

import matplotlib
matplotlib.use("Agg")

import sys
import subprocess
from math import ceil

# Install missing dependencies
REQUIRED_IMPORTS = {
    "pyyaml": "yaml",
    "zipp": "zipp",
}
for pkg, import_name in REQUIRED_IMPORTS.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pathlib
import shutil
import re

# ── paths ────────────────────────────────────────────────────────────────────
HERE  = pathlib.Path(__file__).parent
ROOT  = HERE.parents[1]
PAPER = ROOT.parents[1] / "papers" / "SvalMIZ24-MIP" / "figures"

# Input data files (using final datasets)
ifiletemp     = "/home/maltem/work/python/SvalMIZ2024data/colocatedFiles/final_23072025/dataset_temp.nc"
ifiledrift    = "/home/maltem/work/python/SvalMIZ2024data/colocatedFiles/final_23072025/dataset_traj.nc"

# ── figure style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.format":   "png",
})

# ── data loading ────────────────────────────────────────────────────────────
print("Loading temperature and radiation-flag data...")
OMBtemp = xr.open_dataset(ifiletemp)
OMBdrift = xr.open_dataset(ifiledrift)

# Radiation threshold for flagging
rad_threshold = 100

# Date range and leadtime
start_date = pd.Timestamp('2024-04-04T00:00:00')
end_date   = pd.Timestamp('2024-04-19T00:00:00')
ldint = int(sys.argv[1]) if len(sys.argv) > 1 else 3

time_mask = (OMBtemp.time_ds >= start_date) & (OMBtemp.time_ds <= end_date)

# Plot order consistent with previous manuscript figures, with requested exclusions.
PLOT_MODEL_ORDER = [
    "DWD-ICON",
    "ECMWF-IFS",
    "MF-ARPEGE",
    "ECCC-HRDPSN",
    "ECCC-HRDPSNcoupled",
    "MET-AROMEArctic",
    "MF-AROME",
]

TEMP_MODEL_INDEX = {str(m): i for i, m in enumerate(OMBtemp.model.values)}
plot_models = [m for m in PLOT_MODEL_ORDER if m in TEMP_MODEL_INDEX]

if not plot_models:
    raise RuntimeError("No requested plot models were found in dataset_temp.nc")

DRIFT_MODEL_FOR_TEMP_MODEL = {
    "ECCC-HRDPSN": "ECCC-RIOPS",
    "ECCC-HRDPSNcoupled": "ECCC-RIOPScoupled",
}
DRIFT_MODEL_INDEX = {str(m): i for i, m in enumerate(OMBdrift.model.values)}


def parse_lt_label(label):
    match = re.match(r"\](\d+)_(\d+)\]", str(label))
    if not match:
        raise ValueError(f"Unrecognized lt_int label: {label}")
    return int(match.group(1)), int(match.group(2))


def map_temp_lt_to_drift_lt(temp_label, drift_labels):
    temp_lo, temp_hi = parse_lt_label(temp_label)
    candidates = []
    for i, dlabel in enumerate(drift_labels):
        d_lo, d_hi = parse_lt_label(dlabel)
        if d_lo <= temp_lo and temp_hi <= d_hi:
            candidates.append((d_hi - d_lo, i))
    if not candidates:
        raise RuntimeError(
            f"Could not map temperature leadtime {temp_label} to trajectory lt_int labels"
        )
    candidates.sort()
    return candidates[0][1]


temp_lt_label = str(OMBtemp.lt_int[ldint].values)
drift_lt_labels = [str(x) for x in OMBdrift.lt_int.values]
drift_lt_idx = map_temp_lt_to_drift_lt(temp_lt_label, drift_lt_labels)

# ── plotting ────────────────────────────────────────────────────────────────
print(f"Creating figure for leadtime: {OMBtemp.lt_int[ldint].values}")

n_models = len(plot_models)
ncols = 2
nrows = ceil(n_models / ncols)
fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4.2 * nrows), sharex=True, sharey=True)
axs_flat = np.atleast_1d(axs).ravel()
plt.subplots_adjust(wspace=0.15, hspace=0.25)

# Storage arrays
all_temp_1m = []
all_temp_models = {}
all_SIC_diff = {}

for ipanel, model_name in enumerate(plot_models):
    ax = axs_flat[ipanel]
    temp_model_idx = TEMP_MODEL_INDEX[model_name]

    all_temp_models[model_name] = []
    all_SIC_diff[model_name] = []

    # Loop through all buoys
    for buoy in range(len(OMBtemp.tr_nr)):

        # Radiation flag (interpolated)
        flag3 = OMBtemp.ssdr[5, 0, buoy, :].values
        arr_series = pd.Series(flag3)
        flag3 = arr_series.interpolate(method='linear').to_numpy()

        # Quality flags: temperature + radiation
        flag_condition = (
            (OMBtemp.temp_flag_1m[buoy, :] == 0) &
            (OMBtemp.temp_flag_cons[buoy, :] == 0) &
            (flag3 < rad_threshold) &
            time_mask
        )

        # Extract observations and model forecast
        temp_obs = OMBtemp.temp_1m_calibrated[buoy, flag_condition.values] - 273.15
        temp_model = OMBtemp.T2M[temp_model_idx, ldint, buoy, flag_condition.values] - 273.15

        # SIC difference (model − observed)
        if model_name in DRIFT_MODEL_FOR_TEMP_MODEL:
            drift_model_name = DRIFT_MODEL_FOR_TEMP_MODEL[model_name]
            if drift_model_name not in DRIFT_MODEL_INDEX:
                raise RuntimeError(f"Missing trajectory model {drift_model_name} in dataset_traj.nc")
            drift_model_idx = DRIFT_MODEL_INDEX[drift_model_name]
            SIC_model = OMBdrift.SIC[drift_model_idx, buoy, drift_lt_idx, flag_condition.values]
            SIC_observed = OMBdrift.AMSR2_SIC[buoy, flag_condition.values]
        else:
            SIC_model = OMBtemp.SIC[temp_model_idx, ldint, buoy, flag_condition.values]
            SIC_observed = OMBtemp.AMSR2_SIC[buoy, flag_condition.values]
        SIC_diff = SIC_model - SIC_observed

        all_temp_models[model_name].append(temp_model.values)
        all_SIC_diff[model_name].append(SIC_diff.values)

        if ipanel == 0:  # Store observations only once
            all_temp_1m.append(temp_obs.values)

    # Concatenate all buoy data
    temp_obs_flat = np.concatenate(all_temp_1m) if all_temp_1m else np.array([])
    temp_model_flat = np.concatenate(all_temp_models[model_name]) if all_temp_models[model_name] else np.array([])
    SIC_diff_flat = np.concatenate(all_SIC_diff[model_name]) if all_SIC_diff[model_name] else np.array([])

    # Remove NaN/Inf values
    valid_mask = np.isfinite(temp_obs_flat) & np.isfinite(temp_model_flat) & np.isfinite(SIC_diff_flat)
    temp_obs_flat = temp_obs_flat[valid_mask]
    temp_model_flat = temp_model_flat[valid_mask]
    SIC_diff_flat = SIC_diff_flat[valid_mask]

    # Check if data exists
    if len(temp_obs_flat) > 0 and len(temp_model_flat) > 0:

        # Calculate bias and standard deviation
        bias = np.mean(temp_model_flat - temp_obs_flat)
        stde = np.std(temp_model_flat - temp_obs_flat)

        # Set equal axis limits
        temp_min = min(temp_obs_flat.min(), temp_model_flat.min()) - 1
        temp_max = max(temp_obs_flat.max(), temp_model_flat.max()) + 1
        ax.set_xlim(temp_min, temp_max)
        ax.set_ylim(temp_min, temp_max)

        # Scatter plot with color shading based on SIC difference (sicpositive filter)
        mask_positive_sic = SIC_diff_flat > 0.1
        if np.any(mask_positive_sic):
            ax.scatter(
                temp_obs_flat[mask_positive_sic],
                temp_model_flat[mask_positive_sic],
                c=SIC_diff_flat[mask_positive_sic],
                cmap='coolwarm',
                s=40,
                vmin=-1,
                vmax=1,
                alpha=0.7,
                edgecolors='none'
            )

        # 1:1 reference line
        ax.plot([temp_min, temp_max], [temp_min, temp_max], 'k--', linewidth=0.8, alpha=0.6)

        # Model title
        ax.set_title(model_name, fontsize=11, fontweight='bold')

        # Statistics text box
        textstr = f'Bias: {bias:+.2f}°C\nStdDev: {stde:.2f}°C\nn = {np.sum(mask_positive_sic)}'
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left', bbox=props)

        # Grid
        ax.grid(True, alpha=0.3, linewidth=0.5)

    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No valid data", fontsize=10, ha='center', va='center',
                transform=ax.transAxes, style='italic', color='gray')

# Hide any unused panels if model count is not a multiple of ncols
for ax in axs_flat[n_models:]:
    ax.axis("off")

# Global labels
fig.supxlabel("Observed Temperature (°C)", fontsize=12, fontweight='bold')
fig.supylabel("Model Temperature (°C)", fontsize=12, fontweight='bold')
fig.suptitle(f"Leadtime {OMBtemp.lt_int[ldint].values} | SIC difference > 0.1", fontsize=13, y=0.98)

# Colorbar for SIC difference
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-1, vmax=1)), cax=cbar_ax)
cbar.set_label("SIC Difference (model − obs)", fontsize=10, fontweight='bold')

# ── save ────────────────────────────────────────────────────────────────────
lead_label_safe = temp_lt_label.strip("]").replace("]", "").replace("_", "to")
outfile = HERE / f"Fig_TempVsModel_Lead{lead_label_safe}_SICpositive.png"
print(f"Saving figure to: {outfile}")
fig.savefig(outfile, dpi=300, bbox_inches='tight')

# Copy to paper figures directory
if PAPER.exists():
    dest = PAPER / outfile.name
    shutil.copy(outfile, dest)
    print(f"Copied to: {dest}")
else:
    print(f"Warning: Paper figures directory not found at {PAPER}")

print("Done!")
