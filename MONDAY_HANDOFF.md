# Monday Handoff Notes (2026-03-20)

## Current Repos
- Python repo: `/home/maltem/work/python/SvalMIZ2024`
  - HEAD: `5fea97c`
  - Working tree: clean
- Manuscript repo: `/home/maltem/work/papers/SvalMIZ24-MIP`
  - HEAD: `557f92f`
  - Working tree: clean

## What Was Completed
- Added new spread figure script:
  - `pub_figures/fig_spread_maps/make_figure.py`
- Generated combined 2-panel spread figure:
  - `pub_figures/fig_spread_maps/Spread_T2M_SIT_AnalysisMap.png`
  - Copied to manuscript figures folder as:
    - `figures/Spread_T2M_SIT_AnalysisMap.png`
- Manuscript updated to include the combined spread figure:
  - Reference text added in sea-ice section around `MainManuscript_V01.tex:164`
  - Figure block and label added:
    - `\label{fig:spread_t2m_sit}` (around `MainManuscript_V01.tex:193`)
- Standalone sea-ice spread figure blocks were removed from the manuscript text (as requested earlier).
- All LaTeX `\includegraphics[width=...]` values were scaled by factor `0.5` earlier in the session.

## Figure Details (Combined Spread)
- File: `Spread_T2M_SIT_AnalysisMap.png`
- Layout: 1x2 subplots
  - (a) T2M spread at analysis time
  - (b) Sea-ice thickness spread at analysis time
- Projection/styling follows existing manuscript figure standard:
  - Lambert Conformal (`central_lon=-25`, `central_lat=77.5`, `std_parallel=77.5`)
  - Extent: `[250000, 2150000, -920000, 1490000]`
  - Coastlines only, light gray background
  - Large titles and readable colorbar fonts consistent with earlier figures
- Data source:
  - `/lustre/storeB/project/nwp/SALIENSEAS/SvalMIZ2024/models/AnalysisOfFields/*_00L.nc`
- Models used:
  - T2M spread: 6 models
  - SIT spread: 5 models (ECCC-HRDPSN excluded for SIT due to missing SIT variable)

## LaTeX Compile Status
- `MainManuscript_V01.pdf` is produced successfully.
- There is a pre-existing LaTeX warning/error source still present:
  - `equation` environment inside a figure caption (variogram figure section)
  - Reported messages include:
    - `Missing $ inserted`
    - `You can't use \eqno in math mode`
  - PDF still outputs despite this.

## Suggested First Step on Monday
1. Decide whether to keep `Fig. \ref{fig:spread_t2m_sit}` in the manuscript (it is currently included).
2. If yes: keep current state and continue with next analysis figure.
3. If no: remove the figure block and sentence reference in `MainManuscript_V01.tex`.
4. Optionally fix the caption equation issue for a fully clean LaTeX build.
