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
- **Between profiles** (`neighbours`): scoring draws a separate noise for every profile, so two neighbouring
  predictions are unrelated draws. `figures.py field` gives every member ONE draw for the whole map instead, which
  makes a member a coherent field without retraining (the error bars of each profile are unchanged). It prints the
  correlation between two profiles, next to each other and `neighbour_max_km` apart, for the real error and for the
  members. If the members are more alike far apart than the real error is, give the noise a length scale with
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
