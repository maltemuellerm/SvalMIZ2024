# Publication Figures — SvalMIZ24-MIP

## Overview

This directory contains publication-quality figures for the manuscript:

> **"Intercomparison of coupled forecast models against in situ data in the Marginal Ice Zone"**
> (`SvalMIZ24-MIP/MainManuscript_V01.tex`)

Each figure lives in its own subdirectory and is produced by a self-contained Python script (`make_figure.py`). The final PNG is copied to (or symlinked from) `SvalMIZ24-MIP/figures/` for inclusion in the LaTeX manuscript.

---

## Directory layout

```
pub_figures/
├── FIGURES_RECIPE.md            ← this file
├── fig01_variogram_temp/
│   ├── make_figure.py
│   └── Figure_Variogram_Temp.png
├── fig02_bias_rmse_temp/
│   ├── make_figure.py
│   └── BiasRMSE_Temp2024.png
├── fig03_sic_bias_temp/
│   ├── make_figure.py
│   └── ...
├── fig04_sic_verification/
│   ├── make_figure.py
│   └── ...
├── fig05_icedrift/
│   ├── make_figure.py
│   └── ...
├── fig06_waves_spectra/
│   ├── make_figure.py
│   └── ...
└── fig07_swh_verification/
    ├── make_figure.py
    └── ...
```

---

## Figure inventory

| Dir | LaTeX label | Filename | Manuscript section | Status |
|-----|-------------|----------|--------------------|--------|
| `fig01_variogram_temp` | `fig:variogram_april` | `Figure_Variogram_Temp.png` | 4.1 Surface and air temperature | done |
| `fig02_bias_rmse_temp` | `fig:bias_rmse_cao_wai` | `BiasRMSE_Temp2024.png` | 4.1 Surface and air temperature | done |
| `fig_t2m_analysis_maps` | `fig:t2m_analysis_maps` | `T2M_RMSE_BIAS_AnalysisMap.png` | 4.1 – spatial RMSE/Bias maps | done |
| `fig_sit_analysis_maps` | `fig:sit_rmse_map`, `fig:sit_mean_map` | `SIT_RMSE_AnalysisMap.png`, `SIT_Mean_AnalysisMap.png` | 4.2 Sea-ice characteristics | done |
| `fig_sit_mean_catA_catB` | `fig:sit_mean_catA`, `fig:sit_mean_catB`, `fig:sit_mean_awi` | `SIT_Mean_CatA_AnalysisMap.png`, `SIT_Mean_CatB_AnalysisMap.png`, `SIT_Mean_CryoSMOS_AWI.png` | 4.2 Sea-ice characteristics | done |
| `fig03_sic_bias_temp`  | — | `BiasTempConditionalSIC.png` | 4.1 – SIC-conditional bias | todo |
| `fig04_sic_verification` | — | `SIC_Verification.png` | 4.2 Sea-ice characteristics | todo |
| `fig05_icedrift`       | — | `IceDrift_Verification.png` | 4.2 Sea-ice drift | todo |
| `fig06_waves_spectra`  | — | `WaveSpectra.png` | 4.3 Waves – energy spectra | todo |
| `fig07_swh_verification` | — | `SWH_Verification.png` | 4.3 Waves – SWH | todo |

---

## Recipe for adding a new figure

### 1. Create a subdirectory

```bash
mkdir pub_figures/figXX_short_name
```

### 2. Write `make_figure.py`

Each script should follow this structure:

```python
"""
Figure XX – Short description
Manuscript section: X.X
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# ── paths ──────────────────────────────────────────────────────────────────
HERE   = pathlib.Path(__file__).parent
ROOT   = HERE.parents[1]       # SvalMIZ2024/
PAPER  = ROOT.parent / "papers" / "SvalMIZ24-MIP" / "figures"

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
# ... load your data here ...

# ── plotting ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

# ... your plot code here ...

ax.set_xlabel("...")
ax.set_ylabel("...")
ax.set_title("...")

# ── save ────────────────────────────────────────────────────────────────────
outfile = HERE / "FigureXX_ShortName.png"
fig.savefig(outfile)
print(f"Saved: {outfile}")

# copy to paper figures dir
import shutil
shutil.copy(outfile, PAPER / outfile.name)
print(f"Copied to: {PAPER / outfile.name}")
```

### 3. Run the script

```bash
cd /home/maltem/work/python/SvalMIZ2024
python pub_figures/figXX_short_name/make_figure.py
```

### 4. Add the figure to the LaTeX manuscript

In `MainManuscript_V01.tex`:

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\textwidth]{figures/FigureXX_ShortName.png}
    \caption{...}
    \label{fig:short_name}
\end{figure}
```

---

## Style conventions

- **Figure size**: use `figsize=(7, 4)` for single-column, `(14, 5)` for full-width, `(7, 8)` for tall panels.
- **Resolution**: always save at 300 dpi (`savefig.dpi = 300`).
- **Color maps**: prefer perceptually uniform maps (`viridis`, `RdBu_r`, `coolwarm`). Avoid `jet`.
- **Line widths**: 1.5 pt for data lines, 0.8 pt for reference/grid lines.
- **Legend**: place inside axes when possible; use `framealpha=0.8`.
- **Model colors**: keep a shared color dict across figures (define once, import everywhere).

---

## Existing analysis notebooks (source material)

| Topic | Notebook |
|-------|----------|
| Temperature variogram | `11_MIPanalysisPaper/Temp_Variogram.ipynb` |
| Temperature bias/RMSE | `11_MIPanalysisPaper/BiasTempVerification.ipynb` |
| SIC-conditional bias | `11_MIPanalysisPaper/BiasTempConditionalSIC.ipynb` |
| Sea-ice drift analysis | `11_MIPanalysisPaper/Analyse_drift_trajectory.ipynb` |
| Model spread | `09_AnalysisPaper/T2M_spread.ipynb` |
| MODIS / SIC projected fields | `04_ProjectedFields_Evaluation/` |
| Wave analysis | `08_WaveAnalysis/` |
| Trajectory temperature | `06_TemperatureAlongTrajectory_Evaluation/` |
