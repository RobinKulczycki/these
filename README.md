# Black Sea profile prediction (FGN), v4

Probabilistic prediction of the whole water column: O2 + 8 other variables (T, S, sigma0, chl, NO3, PO4, DIC, pH),
all predicted by the same FGN model and weighted equally in the loss. Three experiments, selected with `experiment=`:

| experiment | what the model is given | rows of the scores |
|---|---|---|
| `cutoff` | the profile from the surface down to D_input | D_input (m) |
| `sparse` | the profile every `spacing` m | spacing (m) |
| `neighbours` | 1-4 other columns of the same time step within `neighbour_max_km` (100 km), nothing of the column itself | distance to the nearest given column (km) |

| file | what |
|---|---|
| `config.py` | every setting and its default; `key=value` on a command changes one for that command |
| `data_download.py`, `preprocess.py` | Copernicus download, raw NetCDF -> profile arrays (with the pH correction) |
| `check_ph.py` | figure and numbers on the pH jumps of a processed folder |
| `dataset.py`, `model.py`, `metrics.py` | data + drift removal, FGN models, every score |
| `train.py` | training and `--eval-only`. `Task` is the one place where the experiments differ |
| `figures.py` | every figure from the metrics files |
| `submit.sh`, `analyse.sh`, `run.slurm` | cluster jobs: train + evaluate one run; every figure of any runs |
| `cluster.sh` | what depends on the cluster: the Python environment and the folder of the runs (`$R`) |
| `test.py` | checks that nothing is broken (~8 min on CPU, synthetic data) |

## From nothing to every figure

```bash
python data_download.py                        # login node, after `copernicusmarine login`
python preprocess.py --workers 8               # CPU job
python test.py                                 # CPU job
bash submit.sh v5_cutoff experiment=cutoff
bash submit.sh v5_sparse experiment=sparse
bash submit.sh v5_nb experiment=neighbours
bash analyse.sh v5_cutoff v5_sparse v5_nb      # once the runs are trained
SPLIT=test bash analyse.sh v5_cutoff v5_sparse v5_nb   # the test years: once, at the very end
```
Every setting of these runs is a default of `config.py` (data folder, model size, 20 epochs, EMA, drift removal,
range of the neighbours ...): a command only names what differs. `submit.sh NAME` trains `$R/NAME`, then evaluates
its best checkpoint on 50,000 validation profiles (`$R/NAME/val_metrics.npz`). A name that already exists is
continued, not restarted: use a new name for a new training. `analyse.sh RUN ...` writes `figs/RUN/`
(`figs/RUN_test/`); a `neighbours` run also gets `casts/`, `layouts/` and `field/` (see the head of the script).
`--eval-only` scores the validation years unless `--split test` is given. W&B runs offline:
`wandb sync <run>/wandb/offline-run-*`. The scripts read `cluster.sh` themselves; for a command typed in a terminal
that contains `$R`, load it first: `. cluster.sh`.

Several `neighbours` set-ups: one run per set-up, then up to three of them compared on the same rows
(`neighbour_val_km` is the same for every range; a row beyond a run's `neighbour_max_km` stays empty):
```bash
bash submit.sh v5_nb_50 experiment=neighbours neighbour_max_km=50
bash submit.sh v5_nb_200 experiment=neighbours neighbour_max_km=200
python figures.py compare "50 km=$R/v5_nb_50" "100 km=$R/v5_nb" "200 km=$R/v5_nb_200" --out figs/nb_ranges   # login node
python figures.py maps "50 km=$R/v5_nb_50" "100 km=$R/v5_nb" --out figs/nb_ranges
```

**pH correction.** The pH of the product has a layer, a few levels thick somewhere between ~100 and ~300 m, where
it is ~7.73 whatever the DIC, with smooth values (~7.1) above and below; the layer follows the density surfaces over
the whole sea, every year. `preprocess.py` (`fix_ph`) finds it by the jump up of pH from one level to the next and
replaces it by a straight line in depth between the levels around it. `fix_ph=False` keeps the product as it is.
A folder processed before the correction existed is corrected without the raw files, and checked, with:
```bash
python preprocess.py --from-processed /globalsc/ulg/sysmod/rkulczyc/data/processed_daily_w2_s4_v4   # -> processed_dir
python check_ph.py > check_ph.txt              # check_ph.png: the profiles, a map, the share of profiles with a jump
```

Drift removal (`input_drift_mode`): `shallowest` (minus the shallowest observed value; the default of `sparse`),
`lagged` (minus the basin-mean profile of the same month over the archive years [Y - drift_lag_years -
drift_window_years + 1, Y - drift_lag_years]; the default of `cutoff` and `neighbours`), `none`.

## What the figures answer (`figs/<run>/`)

| question | figures |
|---|---|
| How good, and where in the column? | `compare/`: `crps_`, `bias_`, `coverage_`, `spread_skill_` + `depth` / `density`; `heatmap_crps_*`, `relative_crps_*`, `summary.csv` |
| How does the error grow with the row (D_input, spacing, distance to the nearest given column)? | `compare/`: `crps_by_row`, `<variable>_per_row_*`, `heatmap_crps_*`, `oxycline_by_row` |
| When? | `compare/`: `time_*` (every month of the evaluation years), `season_*` (calendar month; `_layers` = per water-mass layer), `season_depth_*` (calendar month x depth) |
| Where in the sea? | `maps/`: `maps_*` (per variable, per water-mass layer, oxycline), `maps_<variable>_seasons` |
| Are the error bars right? | `compare/`: `coverage_*`, `spread_skill_*`, `rank_histogram` (flat = calibrated, U = members too close together, dome = too far apart, slope = bias) |
| Do the members hang together? | `compare/coherence`, `ensemble_all_variables` + `ensemble_coherence.csv`, `field_*` (`neighbours`) |
| `neighbours`: 1, 2 or 4 given columns? | `casts/` |
| `neighbours`: stations on a grid or placed like Argo floats, and how many? | `layouts/`: `layouts_error`, `layouts_map_1inN`, `layouts.csv` |
| `neighbours`: maps, how far apart the stations are, coherence between columns | `field/`: `field_<variable>_<depth>m_stations<km>km[_noise<km>km]` |
| The data itself | `data/`, `drift/` |

Every `compare` / `maps` figure takes 1-3 runs of the same experiment:
`python figures.py all "a=$R/run_a" "b=$R/run_b" --out figs/a_vs_b` (adds `heatmap_change_*`, `maps_crps_change`).

## Members and coherence

A member is one draw of a noise vector. That one draw gives all 9 variables at every level, so a member is one
water column: its O2, density, nutrients ... belong together. This holds in the three experiments.

- **Between levels and between variables** it is measured at every evaluation (`coherence.png`, the log line,
  `summary.csv`): the members are scored on the difference between two consecutive levels, and between two
  variables, and again after being shuffled independently at every level and variable (same error bars, no
  coherence). Coherence = 1 - CRPS as predicted / CRPS shuffled: 0 = worth no more than independent draws,
  1 = perfect, below 0 = the members hang together in the wrong way.
- **Physically**: `figures.py ensemble` writes `ensemble_coherence.csv` (density inversions, and how far a member's
  sigma0 is from the equation of state applied to its own T and S), and draws three members in the same colour in
  every variable panel.
- **Between columns** (`neighbours`): columns predicted together share their noise. `figures.py field` (and
  `layouts`) predicts a whole map at once, and a member gives close columns the same noise, which makes the member a
  map that passes smoothly from one column to the next. The noise of a member varies slowly over `--noise-km`
  (default `neighbour_max_km`), so columns far apart get different draws; `--noise-km 0` uses one draw for the whole
  sea. `field` prints the correlation between two columns, next to each other and `neighbour_max_km` apart, for the
  real error and for the members: choose the `--noise-km` that makes them agree. The scores of single columns draw
  one noise per column: for one column this changes nothing, and it makes the scores less noisy.

## Neighbours experiment

Given 1..`neighbour_casts` (4) columns of the same time step within `neighbour_max_km` of a position, predict the
whole profile there. Every profile of the data is used as a target, its given columns drawn at random among the
profiles around it (validation always draws the same). A given column is as likely to be close as far: every
distance up to `neighbour_max_km` is equally likely (drawn among all the profiles in range alike, most would be far
away, because there are more of them there, and the model would rarely learn from a close one). Each given column is
encoded with its offset (km east, km north); the target only gives its context (position, month, bottom depth, MLD,
SSH). `neighbour_max_km` is how far
the model may look: the scores are given per distance to the nearest given column (`neighbour_val_km`, bin upper
edges, the last one >= `neighbour_max_km`). Each evaluation also scores a baseline, the inverse-distance-weighted
mean of the same columns (`<metrics file>_idw.npz`), which `figures.py` adds to every figure.
`neighbour_eval_casts=N` evaluates with exactly N given columns (`--tag` names the metrics file); training always
draws 1..`neighbour_casts`. A sample has up to 5x the tokens of a cutoff one: lower `batch_size` if the GPU runs out
of memory. Drift: `none` or `lagged`.

```bash
python figures.py field $R/v5_nb --out figs/v5_nb --station-km 60 --var o2 --depth 50 [--step 6] [--noise-km 80]
python figures.py layouts $R/v5_nb --out figs/v5_nb/layouts --every 5 10 15 [--steps 0 3 6 9] [--coastal-weight 0.05]
```
`layouts`: real floats are not on a grid, and there are almost none in the coastal area (CMEMS-BLK-QUID-007-010,
Fig. 1). Whole maps (snapshots: the stations do not move) are reconstructed from stations on a regular grid, one
grid point in N each way, and from the same number of stations drawn at random, a coastal column (bottom shallower
than `--coastal-depth`, 200 m) being `--coastal-weight` times as likely as a deep-sea one. A column further than
`neighbour_max_km` from every station is not reconstructed, so the errors are compared on the columns that both
layouts reconstruct, next to the share of the sea each one reconstructs. The model is trained with given columns
drawn anywhere (coastal or not), so that it serves both layouts.
