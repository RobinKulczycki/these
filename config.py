"""All settings. Every field can be overridden on the command line: --set key=value (parsed with ast.literal_eval)."""
from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

PRODUCT_BGC = "BLKSEA_MULTIYEAR_BGC_007_005"  # BAMHBI biogeochemistry, 59 levels
PRODUCT_PHY = "BLKSEA_MULTIYEAR_PHY_007_004"  # NEMO physics, 121 levels
FREQUENCY_CODES = {"monthly": "P1M-m", "daily": "P1D-m"}


@dataclass(frozen=True)
class DatasetGroup:
    product_id: str
    pattern: str  # regex on the dataset id: group(1) = stream ("my" or "myint"), {freq} = P1M-m / P1D-m
    variables: Tuple[str, ...]
    is_3d: bool = True

    def regex(self, temporal_resolution: str) -> str:
        return self.pattern.replace("{freq}", FREQUENCY_CODES[temporal_resolution])


DATASET_GROUPS: Dict[str, DatasetGroup] = {
    "bgc_bio": DatasetGroup(PRODUCT_BGC, r"^cmems_mod_blk_bgc-bio_(my|myint)_2\.5km_{freq}$", ("o2",)),
    "bgc_nut": DatasetGroup(PRODUCT_BGC, r"^cmems_mod_blk_bgc-nut_(my|myint)_2\.5km_{freq}$", ("no3", "po4")),
    "bgc_car": DatasetGroup(PRODUCT_BGC, r"^cmems_mod_blk_bgc-car_(my|myint)_2\.5km_{freq}$", ("dissic", "talk", "ph")),
    "bgc_plankton": DatasetGroup(PRODUCT_BGC, r"^cmems_mod_blk_bgc-plankton_(my|myint)_2\.5km_{freq}$", ("chl",)),
    "phy_tem": DatasetGroup(PRODUCT_PHY, r"^cmems_mod_blk_phy-temp?_(my|myint)_2\.5km_{freq}$", ("thetao",)),
    "phy_sal": DatasetGroup(PRODUCT_PHY, r"^cmems_mod_blk_phy-sal_(my|myint)_2\.5km_{freq}$", ("so",)),
    "phy_mld": DatasetGroup(PRODUCT_PHY, r"^cmems_mod_blk_phy-mld_(my|myint)_2\.5km_{freq}$", ("mlotst",), is_3d=False),
    "phy_ssh": DatasetGroup(PRODUCT_PHY, r"^cmems_mod_blk_phy-ssh_(my|myint)_2\.5km_{freq}$", ("zos",), is_3d=False),
}
REFERENCE_GROUP = "bgc_bio"  # its lat/lon/depth grid is the common grid
PHYSICAL_FEATURES = ("thetao", "so", "sigma0")  # may be extended one level towards the seafloor
DERIVED_FEATURES = {"sigma0": ("thetao", "so")}


@dataclass
class Config:
    # ---- paths
    raw_dir: str = "/globalsc/ulg/sysmod/rkulczyc/data/raw"
    raw_dir_daily: str = "/globalsc/ulg/sysmod/rkulczyc/data/raw_daily"
    processed_dir: str = "/globalsc/ulg/sysmod/rkulczyc/data/processed_v3"
    ckpt_dir: str = "/globalsc/ulg/sysmod/rkulczyc/data/runs/default"

    # ---- download / preprocessing
    lon_min: float = 27.25
    lon_max: float = 42.0
    lat_min: float = 40.5
    lat_max: float = 47.0
    max_depth_m: Optional[float] = 500.0
    include_interim: bool = True
    temporal_resolution: str = "monthly"  # or "daily" (raw_dir_daily, days_per_week days of every week)
    days_per_week: int = 1
    week_day_offset: int = 0
    spatial_stride: int = 2
    min_valid_levels: int = 3
    max_interior_gap_levels: int = 2
    max_bottom_extend_levels: int = 1
    profile_features: Tuple[str, ...] = ("o2", "thetao", "so", "sigma0", "chl", "no3", "po4", "dissic", "ph")
    context_vars_2d: Tuple[str, ...] = ("mlotst", "zos")
    ssh_anomaly: bool = True  # SSH minus its basin mean of each time step
    target_var: str = "o2"
    log_transform_vars: Tuple[str, ...] = ("chl",)
    log_eps: float = 1e-3

    # ---- chronological splits
    train_years: Sequence[int] = tuple(range(1993, 2019))
    val_years: Sequence[int] = (2019, 2020, 2021)
    test_years: Sequence[int] = (2022, 2023, 2024)

    # ---- experiment: "cutoff" = observed from the surface down to D_input; "sparse" = observed every `spacing` m;
    #      "neighbours" = nothing observed, 1..neighbour_casts casts of the same time step within neighbour_max_km
    experiment: str = "cutoff"
    min_input_levels: int = 1
    max_input_depth_m: Optional[float] = 150.0
    val_cutoff_depths_m: Tuple[float, ...] = (5, 10, 20, 30, 50, 75, 100, 125, 150)
    sparse_train_spacings_m: Tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
    sparse_val_spacings_m: Tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
    sparse_first_depth_m: Optional[float] = None  # None = one spacing below the surface
    sparse_max_obs_depth_m: Optional[float] = None
    neighbour_casts: int = 4
    neighbour_max_km: float = 50.0
    neighbour_eval_casts: Optional[int] = None  # evaluation: exactly this many casts (None: as in training)
    # A sample is a pair of neighbouring profiles predicted with the same noise; the loss also scores their
    # difference (x this weight), which is what keeps a member coherent from one column to the next. 0 = off.
    neighbour_pair_weight: float = 1.0
    neighbour_val_km: Tuple[float, ...] = (10, 20, 30, 40, 50)  # rows: distance of the nearest cast, bin upper edges

    # ---- model (FGN: every LayerNorm modulated by a noise vector shared by the whole profile)
    d_model: int = 192
    n_heads: int = 8
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.1
    n_depth_freqs: int = 16
    depth_min_scale_m: float = 2.0
    depth_max_scale_m: float = 4000.0
    enforce_nonnegative: bool = True
    nonnegative_vars: Tuple[str, ...] = ("o2", "no3", "po4")  # bounded at 0 unless drift-shifted
    fgn_noise_dim: int = 32
    fgn_train_members: int = 2
    fgn_eval_members: int = 19  # the member range is a 90 % interval when calibrated
    fgn_member_chunk: int = 0  # members per forward pass in evaluation, 0 = automatic
    fgn_crps_alpha: float = 1.0  # 1 = fair CRPS
    fgn_epistemic_dropout: bool = False  # independent dropout masks per member at evaluation

    # ---- targets and loss
    aux_target_vars: Tuple[str, ...] = ("thetao", "so", "sigma0", "chl", "no3", "po4", "dissic", "ph")
    variable_weighting: str = "equal"  # "equal": mean over variables; "o2": O2 + aux_loss_weight x mean aux
    variable_weights: Dict[str, float] = field(default_factory=dict)  # with "equal", e.g. {'o2': 2.0}
    aux_loss_weight: float = 0.5

    # ---- drift removal of input_drift_vars (inputs and targets; O2 never)
    #   "shallowest": minus the profile's value at its shallowest observed level
    #   "lagged":     minus the basin-mean profile of the same calendar month over archive years
    #                 [year - drift_lag_years - drift_window_years + 1, year - drift_lag_years]
    input_drift_mode: str = "none"
    input_drift_vars: Tuple[str, ...] = ("thetao", "so", "sigma0", "dissic", "talk", "ph")
    drift_lag_years: int = 1
    drift_window_years: int = 3

    # ---- evaluation
    coverage_tol_mmol_m3: float = 0.5  # lets an interval resting on 0 cover an exact 0
    oxycline_o2_threshold: float = 20.0
    density_range: Tuple[float, float] = (9.0, 17.5)  # scores also binned by the true sigma0 of each point
    density_bin_width: float = 0.05
    density_layers: Tuple[float, ...] = (14.5, 15.4, 16.2)  # upper | oxycline | suboxic | anoxic
    map_resolution_deg: float = 0.2
    example_profiles: int = 4
    plot_max_depth_m: float = 250.0

    # ---- optimisation
    batch_size: int = 512
    eval_batch_size: int = 1024
    epochs: int = 8
    steps_per_epoch: Optional[int] = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup_steps: int = 2000
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    ema_decay: float = 0.0  # > 0: validate / save an exponential moving average of the weights
    amp_dtype: str = "auto"  # bf16 on Ampere+, fp16 on Volta/Turing, fp32 otherwise
    num_workers: int = 8
    in_memory: bool = True
    val_max_profiles: Optional[int] = 50_000
    log_every: int = 50
    seed: int = 42

    # ---- Weights & Biases
    wandb_project: Optional[str] = None  # None: blacksea-o2-profiles (cutoff) / -sparse / -neighbours
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"  # the WANDB_MODE environment variable takes precedence

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def raw_root(cfg: Config) -> str:
    DATASET_GROUPS[REFERENCE_GROUP].regex(cfg.temporal_resolution)  # validates the value
    return cfg.raw_dir_daily if cfg.temporal_resolution == "daily" else cfg.raw_dir


def kept_weekdays(cfg: Config, days_per_week: Optional[int] = None) -> List[int]:
    """Days kept in daily mode, as (days since 1970-01-01) % 7, evenly spread over the week."""
    n = int(cfg.days_per_week if days_per_week is None else days_per_week)
    if not 1 <= n <= 7:
        raise ValueError(f"days_per_week must be between 1 and 7, not {n}")
    return sorted({(int(i * 7 / n + 0.5) + int(cfg.week_day_offset)) % 7 for i in range(n)})


def apply_overrides(cfg: Config, overrides: Optional[Sequence[str]]) -> Config:
    names = {f.name for f in dataclasses.fields(cfg)}
    for item in overrides or []:
        key, _, raw = (s.strip() for s in item.partition("="))
        if not _:
            raise ValueError(f"Override '{item}' must look like key=value")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        if key == "model_type":  # older command lines; only the FGN model exists now
            if value != "fgn":
                raise ValueError("only model_type=fgn exists (the quantile model was removed)")
            continue
        if key not in names:
            raise KeyError(f"Unknown config key '{key}'")
        setattr(cfg, key, value)
    return cfg


# Fields added after some runs were trained, with the value that reproduces those runs.
LEGACY_DEFAULTS = {"aux_target_vars": (), "variable_weighting": "o2", "nonnegative_vars": ("o2",)}
# Fields that change the trained function: a resumed run keeps the checkpoint's value.
FROZEN_ON_RESUME = ("variable_weighting", "nonnegative_vars", "variable_weights", "experiment")


def config_from_checkpoint(ckpt: dict) -> Config:
    """Config of a saved run, readable for checkpoints of every earlier version of this code."""
    d = ckpt["cfg"]
    if str(d.get("model_type", "fgn")).lower() != "fgn":
        raise SystemExit("this checkpoint is a quantile model, which this code no longer supports")
    if d.get("input_drift_mode", "none") in ("surface", "basin"):
        raise SystemExit(f"this checkpoint uses input_drift_mode={d['input_drift_mode']!r}, no longer supported")
    names = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in d.items() if k in names})
    for key, value in LEGACY_DEFAULTS.items():
        if key not in d:
            setattr(cfg, key, value)
    if "experiment" not in d:  # runs of train_sparse.py have the sparse query encodings
        cfg.experiment = "sparse" if "no_obs.weight" in ckpt["model"] else "cutoff"
    return cfg
