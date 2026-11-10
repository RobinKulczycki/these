# Black Sea profile prediction (FGN), v4

Probabilistic prediction of the whole water column (O2 + 8 other variables, all weighted equally in the loss).
Three experiments, selected with `experiment=`: `cutoff` (observed from the surface down to D_input), `sparse`
(observed every `spacing` m) and `neighbours` (a whole profile from 1-4 nearby casts of the same time step).

| file | what |
|---|---|
| `config.py` | every setting (`--set key=value` on any script) |
| `data_download.py`, `preprocess.py` | Copernicus download, raw NetCDF -> profile arrays |
| `dataset.py`, `model.py`, `metrics.py` | data + drift removal, FGN models, scores (depth, density, maps, months, oxycline) |
| `train.py` | training and `--eval-only`, every experiment, 1 GPU or DDP |
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

## Neighbours (2-D) experiment

Given 1..`neighbour_casts` (4) casts of the same time step within `neighbour_max_km` (50) of a position, predict
the whole profile there (members of all 9 variables). Every profile of the data is used as a target, its casts drawn
at random among the profiles around it; validation always draws the same casts. Each cast is encoded with its
offset (km east, km north); the target only gives its context (position, month, bottom depth, MLD, SSH).
```bash
bash submit.sh v4_nb $B experiment=neighbours input_drift_mode=lagged "input_drift_vars=('thetao','so','sigma0','dissic','ph')"
# exactly 1 (or 4) casts instead of 1-4 (GPU job):
python train.py --eval-only --split val --max-profiles 50000 --tag 1cast --set ckpt_dir=$R/v4_nb processed_dir=$D neighbour_eval_casts=1
python figures.py all "model=$R/v4_nb" --out figs/nb                 # model vs IDW in every figure
python figures.py compare "1 cast=$R/v4_nb/val_metrics_1cast.npz" "4 casts=$R/v4_nb/val_metrics_4cast.npz" --out figs/nb_casts
python figures.py ensemble $R/v4_nb --row 1 --out figs/nb             # members + the given cast
```
Rows of every score are the distance to the nearest given cast (`neighbour_val_km`, bin upper edges; the default
10-km bins suit a ~5 km grid: with `spatial_stride=4` data (~11 km between profiles) use e.g.
`"neighbour_val_km=(15,25,35,50)"`). Each evaluation also scores the baseline: the inverse-distance-weighted mean of
the same casts (`*_idw.npz`), which `figures.py` adds automatically. A sample has up to 5x the tokens of a cutoff
one: lower `batch_size` if the GPU runs out of memory. Drift: `none` or `lagged` (`shallowest` has no meaning here).

Drift modes: `none`, `shallowest` (minus the shallowest observed value), `lagged` (minus the basin-mean profile of the
same month over the archive years [Y - drift_lag_years - drift_window_years + 1, Y - drift_lag_years]).

## Checks

```bash
python test.py        # ~5 min on CPU: every experiment and drift mode, resume, eval, DDP, old checkpoints, figures
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
