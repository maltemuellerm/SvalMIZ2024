"""
Figure: Temperature bias (model − obs) vs SIC difference
Manuscript section: 4.1 – Understanding SIC impact on temperature errors
Leads: variable (default 36-48 hours) | Shows scatter (colored by T_obs) and hexbin density
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
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
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

# Plot order consistent with previous manuscript figures
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
ncols = 2  # Two columns per model: scatter + hexbin
nrows = n_models
fig, axs = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharex=False, sharey=False)
if nrows == 1:
    axs = axs.reshape(1, -1)
plt.subplots_adjust(wspace=0.3, hspace=0.35)

# Collect all data for global statistics
all_temp_obs = []
all_temp_model = []
all_SIC_diff = []

for ipanel, model_name in enumerate(plot_models):
    temp_model_idx = TEMP_MODEL_INDEX[model_name]
    
    temp_obs_list = []
    temp_model_list = []
    SIC_diff_list = []

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
        
        temp_obs_list.append(temp_obs.values)
        temp_model_list.append(temp_model.values)
        SIC_diff_list.append(SIC_diff.values)
    
    # Concatenate all buoy data
    temp_obs_flat = np.concatenate(temp_obs_list) if temp_obs_list else np.array([])
    temp_model_flat = np.concatenate(temp_model_list) if temp_model_list else np.array([])
    SIC_diff_flat = np.concatenate(SIC_diff_list) if SIC_diff_list else np.array([])
    
    # Remove NaN/Inf values
    valid_mask = np.isfinite(temp_obs_flat) & np.isfinite(temp_model_flat) & np.isfinite(SIC_diff_flat)
    temp_obs_flat = temp_obs_flat[valid_mask]
    temp_model_flat = temp_model_flat[valid_mask]
    SIC_diff_flat = SIC_diff_flat[valid_mask]
    
    all_temp_obs.extend(temp_obs_flat)
    all_temp_model.extend(temp_model_flat)
    all_SIC_diff.extend(SIC_diff_flat)
    
    if len(temp_obs_flat) == 0:
        continue
    
    # Calculate bias
    bias = temp_model_flat - temp_obs_flat
    
    # ── LEFT PANEL: Scatter plot (bias vs SIC_diff, colored by T_obs) ──
    ax_scatter = axs[ipanel, 0]
    sc = ax_scatter.scatter(
        SIC_diff_flat,
        bias,
        c=temp_obs_flat,
        cmap='RdYlBu_r',
        s=30,
        alpha=0.6,
        edgecolors='none',
        vmin=-15,
        vmax=5
    )
    ax_scatter.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_scatter.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_scatter.set_xlabel('SIC difference (model − obs)', fontsize=10)
    ax_scatter.set_ylabel('Temperature bias (model − obs) [°C]', fontsize=10)
    ax_scatter.set_title(f'{model_name} — Scatter', fontsize=10, fontweight='bold')
    ax_scatter.grid(True, alpha=0.2)
    
    # Calculate correlation
    corr = np.corrcoef(SIC_diff_flat, bias)[0, 1]
    ax_scatter.text(0.05, 0.95, f'r = {corr:.2f}\nn = {len(bias)}',
                    transform=ax_scatter.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    # ── RIGHT PANEL: Hexbin (2D density of bias vs SIC_diff) ──
    ax_hexbin = axs[ipanel, 1]
    hb = ax_hexbin.hexbin(
        SIC_diff_flat,
        bias,
        gridsize=15,
        cmap='YlOrRd',
        mincnt=1,
        extent=[-1, 1, -10, 5],
        edgecolors='face',
        linewidths=0.2
    )
    ax_hexbin.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_hexbin.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_hexbin.set_xlabel('SIC difference (model − obs)', fontsize=10)
    ax_hexbin.set_ylabel('Temperature bias (model − obs) [°C]', fontsize=10)
    ax_hexbin.set_title(f'{model_name} — Density', fontsize=10, fontweight='bold')
    
    # Add colorbar to hexbin
    cbar_hb = plt.colorbar(hb, ax=ax_hexbin)
    cbar_hb.set_label('Count', fontsize=8)

# ── Global colorbar for scatter plot ──
cbar_ax = fig.add_axes([0.08, 0.02, 0.3, 0.015])
cbar_sc = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cbar_sc.set_label('Observed Temperature [°C]', fontsize=9)

# ── Title ──
lead_label = temp_lt_label.strip("]").replace("]", "").replace("_", "–")
fig.suptitle(f'Temperature Bias vs SIC Difference (Leadtime {lead_label})', fontsize=13, y=0.995, fontweight='bold')

# ── save ────────────────────────────────────────────────────────────────────
lead_label_safe = temp_lt_label.strip("]").replace("]", "").replace("_", "to")
outfile = HERE / f"Fig_BiasVsSICdiff_Lead{lead_label_safe}.png"
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
