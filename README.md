# Black Sea profile prediction (FGN), v4

Probabilistic prediction of the whole water column: O2 + 8 other variables (T, S, sigma0, chl, NO3, PO4, DIC, pH),
all predicted by the same FGN model and weighted equally in the loss. Three experiments, selected with `experiment=`:

| experiment | what the model is given | rows of the scores |
|---|---|---|
| `cutoff` | the profile from the surface down to D_input | D_input (m) |
| `sparse` | the profile every `spacing` m | spacing (m) |
| `neighbours` | 1-4 nearby casts of the same time step, nothing of the profile itself | distance to the nearest cast (km) |

| file | what |
|---|---|
| `config.py` | every setting (`--set key=value` on any script) |
| `data_download.py`, `preprocess.py` | Copernicus download, raw NetCDF -> profile arrays |
| `dataset.py`, `model.py`, `metrics.py` | data + drift removal, FGN models, every score |
| `train.py` | training and `--eval-only`, every experiment, 1 GPU or DDP. `Task` is the one place where the experiments differ |
| `figures.py` | every figure from the metrics files |
| `run.slurm`, `submit.sh` | cluster jobs |
| `test.py` | checks that nothing is broken |

## Use

```bash
python data_download.py                        # login node, after `copernicusmarine login`
python preprocess.py --workers 8               # CPU job
bash submit.sh v4_cutoff $B experiment=cutoff input_drift_mode=lagged "input_drift_vars=('thetao','so','sigma0','dissic','ph')"
bash submit.sh v4_sparse $B experiment=sparse input_drift_mode=shallowest "input_drift_vars=('thetao','so','sigma0','dissic','ph')"
EVAL=only bash submit.sh v3_fgn_lag            # runs of older code need a v4 <run>/val_metrics.npz for the figures
python figures.py all "v4=$R/v4_cutoff" "v3=$R/v3_fgn_lag" --out figs/cutoff --processed-dir $D
python figures.py all "v4=$R/v4_sparse" "v2=$R/v2_s_fgn_eq" --out figs/sparse
python train.py --eval-only --split test --set ckpt_dir=$R/v4_cutoff processed_dir=$D   # once, at the very end
```
The two `submit.sh` lines reproduce the best v3 settings (cutoff: lagged drift; sparse: shallowest).
`$B` in `~/tfec/env.sh` still says `input_drift_mode=basin`, which no longer exists: the later
`input_drift_mode=` overrides it, or remove it from `B`. W&B runs offline: `wandb sync <run>/wandb/offline-run-*`.
`--eval-only` scores the validation years unless `--split test` is given.

Metrics files written before the seasonal, rank and coherence scores existed still give every older figure;
re-evaluate a run (`EVAL=only bash submit.sh NAME`) to get the new ones.

## Members and coherence

A member is one draw of a noise vector. That one draw gives all 9 variables at every level, so a member is one
water column: its O2, density, nutrients ... belong together. This holds in the three experiments.

- **Between variables and levels** it is measured at every evaluation (`coherence.png`, the log line, `summary.csv`):
  the members are scored on the difference between two consecutive levels, and between two variables, and again
  after being shuffled independently at every level and variable (same error bars, no coherence).
  Coherence = 1 - CRPS as predicted / CRPS shuffled: 0 = worth no more than independent draws, 1 = perfect,
  below 0 = the members hang together in the wrong way.
- **Physically**: `figures.py ensemble` writes `ensemble_coherence.csv` (density inversions, and how far a member's
  sigma0 is from the equation of state applied to its own T and S), and draws three members in the same colour in
  every variable panel.
- **Between neighbouring predicted columns** (`neighbours`): a member uses the same noise in neighbouring columns,
  always. In training and in every evaluation a sample is a pair, a profile and a profile next to it, given the
  same casts and predicted with the same noise; the loss also scores the difference between the two (member m
  minus member m), so the model learns how a member must change from one column to the next. Each evaluation
  reports it as "coherence between columns" (same scale as above; the reference is the two columns drawn
  separately). `figures.py field` maps a whole time step with ONE draw per member for the whole map, and prints
  the correlation between two profiles, next to each other and `neighbour_max_km` apart, for the real error and for
  the members. If the members are more alike far apart than the real error is, give the noise a length scale with
  `--noise-km` (noise that varies smoothly over the map) and pick the distance that matches.

## Neighbours (2-D) experiment

Given 1..`neighbour_casts` (4) casts of the same time step within `neighbour_max_km` (50) of a position, predict
the whole profile there (members of all 9 variables). Every profile of the data is used as a target, its casts drawn
at random among the profiles around it; validation always draws the same casts. Each cast is encoded with its
offset (km east, km north); the target only gives its context (position, month, bottom depth, MLD, SSH).
```bash
bash submit.sh v4_nb $B experiment=neighbours input_drift_mode=lagged "input_drift_vars=('thetao','so','sigma0','dissic','ph')"
# exactly 1 (or 4) casts instead of 1-4 (GPU job):
python train.py --eval-only --split val --max-profiles 50000 --tag 1cast --set ckpt_dir=$R/v4_nb processed_dir=$D neighbour_eval_casts=1
python figures.py all "model=$R/v4_nb" --out figs/nb                 # model vs IDW in every figure, + the field map
python figures.py compare "1 cast=$R/v4_nb/val_metrics_1cast.npz" "4 casts=$R/v4_nb/val_metrics_4cast.npz" --out figs/nb_casts
python figures.py ensemble $R/v4_nb --row 1 --out figs/nb             # members + the given cast
python figures.py field $R/v4_nb --out figs/nb --station-km 60 --var o2 --depth 50 [--step 6] [--noise-km 80]
```
Rows of every score are the distance to the nearest given cast (`neighbour_val_km`, bin upper edges; the default
10-km bins suit a ~5 km grid: with `spatial_stride=4` data (~11 km between profiles) use e.g.
`"neighbour_val_km=(15,25,35,50)"`). Each evaluation also scores the baseline: the inverse-distance-weighted mean of
the same casts (`<metrics file>_idw.npz`), which `figures.py` adds automatically. A sample has up to 5x the tokens
of a cutoff one: lower `batch_size` if the GPU runs out of memory. Drift: `none` or `lagged` (`shallowest` has no
meaning here). `neighbour_eval_casts` only changes evaluation; training always draws 1..`neighbour_casts`.
`batch_size` counts profiles, so a batch holds `batch_size / 2` pairs. The scores (and the IDW baseline) are those
of the first profile of every pair, the same profiles as without pairs. `neighbour_pair_weight` (1) weights the
loss on the difference between the two columns; 0 trains without it (the pairs still share their noise).

### Where the given profiles are: regular grid or Argo-like

Real floats are not on a grid: there are almost none in the coastal area (CMEMS-BLK-QUID-007-010, Fig. 1).
`figures.py layouts` reconstructs whole maps (snapshots: the stations do not move) from stations on a regular grid,
one grid point in N each way, and from the same number of stations drawn at random, a coastal column (bottom
shallower than `--coastal-depth`, 200 m) being `--coastal-weight` (0.05) times as likely as a deep-sea one.
```bash
python figures.py layouts $R/v4_nb --out figs/nb_layouts --every 5 10 15 [--steps 0 3 6 9] [--var o2 --depth 50]
```
It writes `layouts.csv`, `layouts_error.png` (share of the sea reconstructed, and the error in the whole sea, the deep
sea and the coastal area) and `layouts_map_1inN.png` (stations, reconstructed map, error). A column further than
`neighbour_max_km` from every station is not reconstructed, so the errors are compared on the columns that both
layouts reconstruct and the share of the sea is given next to them. With one station in 10 grid points (~56 km on
the 5.6 km grid) a 50 km range leaves holes, above all for the Argo-like stations: to reconstruct from sparse
stations, train with a larger range, e.g. `neighbour_max_km=100 "neighbour_val_km=(10,25,50,75,100)"`.
The model is trained with casts drawn anywhere around the profile (coastal or not), so that it serves both layouts.

Drift modes: `none`, `shallowest` (minus the shallowest observed value), `lagged` (minus the basin-mean profile of the
same month over the archive years [Y - drift_lag_years - drift_window_years + 1, Y - drift_lag_years]).

## Figures (`figures.py all`)

| question | figures |
|---|---|
| How good, and where in the column? | `crps_`, `bias_`, `coverage_`, `spread_skill_` + `depth` / `density`; `heatmap_crps_*`, `o2_per_row_*`, `crps_by_row`, `relative_crps_*`, `oxycline_by_row`, `summary.csv` |
| When? | `time_*` (every month of the evaluation years), `season_*` (calendar month; `_layers` = per water-mass layer), `season_depth_*` (calendar month x depth) |
| Where in the sea? | `maps/maps_*` (per variable, per water-mass layer, oxycline), `maps_<variable>_seasons` (CRPS and bias per season) |
| Are the error bars right? | `coverage_*`, `spread_skill_*`, `rank_histogram` (flat = calibrated, U = members too close together, dome = too far apart, slope = bias) |
| Do the members hang together? | `coherence`, `ensemble_all_variables` + `ensemble_coherence.csv`, `field_*` (`neighbours`) |
| Does it matter where the given profiles are? | `figures.py layouts` (`neighbours`): `layouts_error`, `layouts_map_1inN`, `layouts.csv` |
| Better than another run? | every figure takes 1-3 runs; `heatmap_change_*`, `maps_crps_change` |
| The data itself | `data/`, `drift/` (with `--processed-dir`) |

## Checks

```bash
python test.py        # ~7 min on CPU: every experiment and drift mode, resume, eval, DDP, old checkpoints, figures
sbatch --gres=gpu:1 -c 4 --mem=48G -t 04:00:00 -o logs/check.out --wrap \
  ". ~/tfec/env.sh; cd ~/tfec/v4; python test.py --check-run $R/v3_fgn_lag $R/v3_s_fgn_lag $R/v2_fgn_eq --processed-dir $D"
```
`--check-run` re-scores runs trained by the old code and compares with the scores the old code saved
(same 50,000 profiles); it also leaves the v4 `val_metrics.npz` in each run for `figures.py`.

## Changes from v3 (results unchanged)

- Removed: the quantile model, the `surface` / `basin` drift modes, `train_sparse.py`, `run_sparse.slurm`,
  `model_*.py`, `scores.py`, `analysis*.py`, `wandb_logging*.py`, `presentation_figures.py`, `drift_diagnostic.py`
  and the `train_v*/plots_v*/night.sh` scripts. Their figures are now `figures.py compare|maps|ensemble|data|drift|all`.
- One metrics file per evaluation: `<split>_metrics.npz` (`val_metrics_best.npz` during training) instead of
  `*_depth_metrics.npz` / `*_sparse_metrics.npz` + `*_maps.npz`. Older runs: `EVAL=only bash submit.sh NAME`.
- Every run trained by v2/v3 (and earlier FGN runs) still loads, resumes and evaluates; a sparse run is recognised
  from its weights. With the same seed v4 trains to bit-identical weights and the same scores as v3 (checked on CPU).
