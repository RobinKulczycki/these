"""Year splits, training-only normalisation, drift references and the PyTorch dataset of whole profiles."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config

log = logging.getLogger("dataset")
DRIFT_MODES = ("none", "shallowest", "lagged")


def load_grid(cfg: Config) -> dict:
    path = Path(cfg.processed_dir) / "grid.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run preprocess.py first")
    return json.loads(path.read_text())


def resolve_splits(cfg: Config, available_years: Sequence[int]) -> Dict[str, List[int]]:
    years = set(int(y) for y in available_years)
    splits = {name: sorted(set(map(int, getattr(cfg, f"{name}_years"))) & years) for name in ("train", "val", "test")}
    if not splits["train"] or not splits["val"]:
        raise ValueError(f"Empty train or validation split: {splits}")
    if max(splits["train"]) >= min(splits["val"]) or (splits["test"] and max(splits["val"]) >= min(splits["test"])):
        raise ValueError(f"Splits must be chronological and disjoint: {splits}")
    return splits


def cutoff_levels(depth: Sequence[float], cutoff_depths_m: Sequence[float], k_min: int = 1) -> Tuple[List[int], List[float]]:
    """D_input values (m) -> k = number of observed levels, with D_input = depth[k-1]."""
    depth = np.asarray(depth, dtype=np.float64)
    ks = sorted({int(np.searchsorted(depth, d, side="right")) for d in cutoff_depths_m})
    ks = [k for k in ks if max(1, k_min) <= k <= len(depth) - 1]
    return ks, [float(depth[k - 1]) for k in ks]


def max_input_levels(depth: Sequence[float], max_input_depth_m: Optional[float]) -> int:
    if max_input_depth_m is None:
        return len(depth) - 1
    return int(min(len(depth) - 1, max(1, np.searchsorted(np.asarray(depth), max_input_depth_m, side="right"))))


# --------------------------------------------------------------------------------------- drift
def drift_indices(cfg: Config, grid: dict) -> List[int]:
    """Feature indices of input_drift_vars present in the data (never the target)."""
    feats = list(grid["features"])
    names = [v for v in (cfg.input_drift_vars or ()) if v in feats]
    if grid["target"] in names:
        raise ValueError(f"input_drift_vars must not contain the target '{grid['target']}'")
    return [feats.index(v) for v in names]


def subtract_shallowest(x: torch.Tensor, observed: torch.Tensor, valid_len: torch.Tensor,
                        drift_idx: torch.Tensor) -> torch.Tensor:
    """Drift features minus their value at each profile's shallowest observed level, at every valid level."""
    bsz, nz, _ = x.shape
    lev = torch.arange(nz, device=x.device)
    valid = lev[None, :] < valid_len[:, None]
    obs = observed & valid
    first = torch.where(obs, lev[None, :], torch.full_like(lev[None, :], nz)).amin(dim=1)
    has_obs = (first < nz).to(x.dtype)
    ref = x[torch.arange(bsz, device=x.device), first.clamp(max=nz - 1)].index_select(-1, drift_idx)
    out = x.clone()
    shifted = x.index_select(-1, drift_idx) - (ref * has_obs[:, None]).unsqueeze(1)
    out[..., drift_idx] = shifted * valid.unsqueeze(-1).to(x.dtype)
    return out


def profile_months(d: Path) -> np.ndarray:
    """Months since 1970-01 of every profile of one year folder."""
    meta = np.load(d / "meta.npz")
    unit = str(meta["time_unit"]) if "time_unit" in meta.files else "M"
    t = meta["time"]
    return t.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) if unit == "D" else t.astype(np.int64)


def _basin_monthly_path(cfg: Config, grid: dict) -> Path:
    key = json.dumps({"years": sorted(map(int, grid["years"])), "features": grid["features"],
                      "depth": grid["depth"]}, sort_keys=True)
    return Path(cfg.processed_dir) / f"basin_monthly_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"


def load_or_compute_basin_monthly(cfg: Config, grid: dict) -> Dict[str, np.ndarray]:
    """Basin-mean profile of every feature per processed year and calendar month (physical units), cached:
    mean (n_years, 12, Z, F) with NaN where no profile reaches a level, count (n_years, 12, Z), years."""
    path = _basin_monthly_path(cfg, grid)
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    log.info("Computing the basin-mean profile of every year and month (cached in %s)", path.name)
    years = sorted(map(int, grid["years"]))
    nz, nf = len(grid["depth"]), len(grid["features"])
    total, count = np.zeros((len(years), 12, nz, nf)), np.zeros((len(years), 12, nz))
    lev = np.arange(nz)
    for yi, year in enumerate(years):
        d = Path(cfg.processed_dir) / f"year_{year}"
        prof = np.load(d / "profiles.npy", mmap_mode="r")
        vlen = np.load(d / "valid_len.npy").astype(np.int64)
        month = profile_months(d) % 12
        for a in range(0, len(vlen), 50_000):
            p = np.asarray(prof[a:a + 50_000], dtype=np.float64)
            valid = (lev[None, :] < vlen[a:a + 50_000, None]).astype(np.float64)
            mo = month[a:a + 50_000]
            for m in np.unique(mo):
                total[yi, m] += p[mo == m].sum(axis=0)
                count[yi, m] += valid[mo == m].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"mean": np.where(count[..., None] > 0, total / count[..., None], np.nan), "count": count,
               "years": np.asarray(years)}
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out


def lagged_years(year: int, available: Sequence[int], lag: int, window: int) -> List[int]:
    """Archive years of the reference of `year`; the first years of the record use the first `window` years."""
    avail = sorted(map(int, available))
    return [y for y in avail if year - lag - window + 1 <= y <= year - lag] or avail[:max(1, window)]


def lagged_reference(bm: Dict[str, np.ndarray], year: int, lag: int, window: int) -> np.ndarray:
    """Per calendar month, the basin-mean profile averaged over the lagged years (physical units, (12, Z, F))."""
    years = [int(y) for y in bm["years"]]
    m = bm["mean"][[years.index(y) for y in lagged_years(year, years, lag, window)]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # mean of an empty slice is the intended NaN
        per_month = np.nanmean(m, axis=0)
        all_months = np.nanmean(m.reshape(-1, *m.shape[2:]), axis=0)
    return np.where(np.isfinite(per_month), per_month, all_months[None])


# --------------------------------------------------------------------------------------- statistics
def _stats_path(cfg: Config, grid: dict, train_years: Sequence[int]) -> Path:
    key = json.dumps({"years": sorted(map(int, train_years)), "features": grid["features"],
                      "context": grid["context"]}, sort_keys=True)
    return Path(cfg.processed_dir) / f"stats_{hashlib.md5(key.encode()).hexdigest()[:10]}.json"


def load_or_compute_stats(cfg: Config, grid: dict, train_years: Sequence[int]) -> Dict[str, np.ndarray]:
    """Per-feature mean / std over all valid levels of the training years (+ NaN-aware context stats), cached."""
    path = _stats_path(cfg, grid, train_years)
    if path.exists():
        return {k: np.asarray(v, dtype=np.float64) for k, v in json.loads(path.read_text()).items()}
    nf, nc, nz = len(grid["features"]), len(grid["context"]), len(grid["depth"])
    s, ss, n = np.zeros(nf), np.zeros(nf), 0
    cs, css, cn = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    lev = np.arange(nz)
    for year in train_years:
        d = Path(cfg.processed_dir) / f"year_{year}"
        prof = np.load(d / "profiles.npy", mmap_mode="r")
        vlen = np.load(d / "valid_len.npy").astype(np.int64)
        for a in range(0, len(vlen), 50_000):
            v = np.asarray(prof[a:a + 50_000], dtype=np.float64)[lev[None, :] < vlen[a:a + 50_000, None]]
            s, ss, n = s + v.sum(0), ss + np.square(v).sum(0), n + v.shape[0]
        ctx = np.load(d / "context.npy").astype(np.float64)
        fin = np.isfinite(ctx)
        cs, css, cn = cs + np.where(fin, ctx, 0.0).sum(0), css + np.where(fin, ctx ** 2, 0.0).sum(0), cn + fin.sum(0)
    mean, ctx_mean = s / max(n, 1), cs / np.maximum(cn, 1)
    stats = {"mean": mean, "std": np.maximum(np.sqrt(np.maximum(ss / max(n, 1) - mean ** 2, 0.0)), 1e-6),
             "ctx_mean": ctx_mean,
             "ctx_std": np.maximum(np.sqrt(np.maximum(css / np.maximum(cn, 1) - ctx_mean ** 2, 0.0)), 1e-6)}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({k: v.tolist() for k, v in stats.items()}))
    tmp.replace(path)
    return stats


# --------------------------------------------------------------------------------------- dataset
class ProfileDataset(Dataset):
    """Standardised profiles: x (Z, F) zero below valid_len, valid_len, ctx (C,); "lagged" drift removal applied.
    extras=True (evaluation) adds x_raw (x without drift removal), lat, lon and month (months since 1970)."""

    def __init__(self, cfg: Config, grid: dict, stats: Dict[str, np.ndarray], years: Sequence[int],
                 max_profiles: Optional[int] = None, seed: int = 0, in_memory: bool = True, extras: bool = False):
        root = Path(cfg.processed_dir)
        self.dirs = [root / f"year_{int(y)}" for y in years]
        for d in self.dirs:
            if not (d / "_DONE").exists():
                raise FileNotFoundError(f"{d} is missing or incomplete - run preprocess.py")
        counts = [np.load(d / "valid_len.npy", mmap_mode="r").shape[0] for d in self.dirs]
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.mean, self.std = (np.asarray(stats[k], dtype=np.float32) for k in ("mean", "std"))
        self.ctx_mean, self.ctx_std = (np.asarray(stats[k], dtype=np.float32) for k in ("ctx_mean", "ctx_std"))
        if cfg.input_drift_mode not in DRIFT_MODES:
            raise ValueError(f"input_drift_mode must be one of {DRIFT_MODES}, not {cfg.input_drift_mode!r}")
        self.lagged = cfg.input_drift_mode == "lagged" and bool(drift_indices(cfg, grid))
        self.drift_idx = np.asarray(drift_indices(cfg, grid), dtype=np.int64)
        self._arrays: List[Optional[tuple]] = [self._load(i, mmap=False) if in_memory else None
                                               for i in range(len(self.dirs))]
        self.extras = extras
        if extras:
            metas = [np.load(d / "meta.npz") for d in self.dirs]
            self._latlon = [(m["lat"].astype(np.float32), m["lon"].astype(np.float32)) for m in metas]
            self._months = [profile_months(d).astype(np.int32) for d in self.dirs]
        if self.lagged:  # per folder: reference of every calendar month (standardised) and each profile's month
            bm = load_or_compute_basin_monthly(cfg, grid)
            di = self.drift_idx
            self._lag_ref, self._month_of = [], []
            for d in self.dirs:
                year = int(d.name.split("_")[-1])
                ref = (lagged_reference(bm, year, cfg.drift_lag_years, cfg.drift_window_years)[..., di]
                       - self.mean[di]) / self.std[di]
                self._lag_ref.append(np.nan_to_num(ref, nan=0.0).astype(np.float32))
                self._month_of.append((profile_months(d) % 12).astype(np.int8))
        self.index = None
        if max_profiles is not None and self.offsets[-1] > max_profiles:
            self.index = np.sort(np.random.default_rng(seed).choice(int(self.offsets[-1]), size=int(max_profiles),
                                                                    replace=False))
        log.info("Dataset years %s: %d profiles%s", list(years), int(self.offsets[-1]),
                 f" (fixed random subset of {len(self.index)})" if self.index is not None else "")

    def _load(self, i: int, mmap: bool) -> tuple:
        mode = "r" if mmap else None
        d = self.dirs[i]
        return tuple(np.load(d / f, mmap_mode=mode) for f in ("profiles.npy", "valid_len.npy", "context.npy"))

    def __len__(self) -> int:
        return len(self.index) if self.index is not None else int(self.offsets[-1])

    def _locate(self, i: int) -> Tuple[int, int, int]:
        """Dataset index -> (global index, year folder, index in the folder)."""
        g = int(self.index[i]) if self.index is not None else int(i)
        f = int(np.searchsorted(self.offsets, g, side="right") - 1)
        return g, f, g - int(self.offsets[f])

    def _profile(self, f: int, j: int):
        """Profile j of folder f -> x (standardised, drift removed, 0 below valid_len), x_raw or None, valid_len, ctx."""
        if self._arrays[f] is None:  # lazily memory-mapped inside each worker
            self._arrays[f] = self._load(f, mmap=True)
        prof, vlen, ctx = self._arrays[f]
        n_valid = int(vlen[j])
        x = (np.asarray(prof[j], dtype=np.float32) - self.mean) / self.std
        x_raw = x.copy() if self.extras else None
        if self.lagged:
            x[:, self.drift_idx] -= self._lag_ref[f][self._month_of[f][j]]
        x[n_valid:] = 0.0
        if x_raw is not None:
            x_raw[n_valid:] = 0.0
        c = np.nan_to_num((np.asarray(ctx[j], dtype=np.float32) - self.ctx_mean) / self.ctx_std, nan=0.0)
        return x, x_raw, n_valid, c

    def __getitem__(self, i: int) -> dict:
        _, f, j = self._locate(i)
        x, x_raw, n_valid, c = self._profile(f, j)
        out = {"x": torch.from_numpy(x), "valid_len": torch.tensor(n_valid, dtype=torch.long), "ctx": torch.from_numpy(c)}
        if self.extras:
            lat, lon = self._latlon[f]
            out.update(x_raw=torch.from_numpy(x_raw), lat=torch.tensor(float(lat[j])), lon=torch.tensor(float(lon[j])),
                       month=torch.tensor(int(self._months[f][j]), dtype=torch.long))
        return out


KM_PER_DEG = 111.2


def nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the nearest value of a sorted axis."""
    j = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    return np.where(np.abs(values - axis[j - 1]) <= np.abs(values - axis[j]), j - 1, j)


class NeighbourDataset(ProfileDataset):
    """Experiment "neighbours": every profile with 1..neighbour_casts other profiles of the same time step within
    neighbour_max_km (the given casts): src_x (K, Z, F), src_len (K,) [0 = no cast], src_ctx (K, C), src_dxy (K, 2)
    = km east / north of the predicted profile. train=False draws the same casts at every evaluation."""

    def __init__(self, cfg: Config, grid: dict, stats: Dict[str, np.ndarray], years: Sequence[int], train: bool = True,
                 max_profiles: Optional[int] = None, seed: int = 0, in_memory: bool = True, extras: bool = False):
        super().__init__(cfg, grid, stats, years, max_profiles, seed, in_memory, extras)
        self.train, self.seed = train, int(seed)
        self.k_max, self.k_eval, self.max_km = int(cfg.neighbour_casts), cfg.neighbour_eval_casts, float(cfg.neighbour_max_km)
        lat_ax, lon_ax = np.asarray(grid["lat"], dtype=np.float64), np.asarray(grid["lon"], dtype=np.float64)
        step = lambda a: float(np.median(np.diff(a))) if a.size > 1 else 1.0  # noqa: E731
        self.ry = int(np.ceil(self.max_km / (KM_PER_DEG * step(lat_ax))))
        self.rx = int(np.ceil(self.max_km / (KM_PER_DEG * step(lon_ax) * np.cos(np.deg2rad(lat_ax.max())))))
        self._ll, self._tyx, self._cell = [], [], []
        for d in self.dirs:  # per year: (time step, row, column) of every profile and the profile of every cell
            meta = np.load(d / "meta.npz")
            lat, lon = meta["lat"].astype(np.float64), meta["lon"].astype(np.float64)
            t = np.unique(meta["time"], return_inverse=True)[1].reshape(-1)
            iy, ix = nearest_index(lat_ax, lat), nearest_index(lon_ax, lon)
            cell = np.full((int(t.max()) + 1 if t.size else 0, len(lat_ax), len(lon_ax)), -1, dtype=np.int32)
            cell[t, iy, ix] = np.arange(len(t), dtype=np.int32)
            self._ll.append((lat, lon))
            self._tyx.append(np.stack([t, iy, ix], axis=1).astype(np.int32))
            self._cell.append(cell)

    def __getitem__(self, i: int) -> dict:
        out = super().__getitem__(i)
        g, f, j = self._locate(i)
        t, y, x = (int(v) for v in self._tyx[f][j])
        cand = self._cell[f][t, max(y - self.ry, 0):y + self.ry + 1, max(x - self.rx, 0):x + self.rx + 1].reshape(-1)
        cand = cand[(cand >= 0) & (cand != j)]
        lat, lon = self._ll[f]
        dy = (lat[cand] - lat[j]) * KM_PER_DEG
        dx = (lon[cand] - lon[j]) * KM_PER_DEG * np.cos(np.deg2rad(lat[j]))
        near = np.hypot(dx, dy) <= self.max_km
        cand, dx, dy = cand[near], dx[near], dy[near]
        rng = np.random.default_rng(int(torch.randint(2 ** 62, ())) if self.train else (self.seed, g))
        k = min(int(self.k_eval or rng.integers(1, self.k_max + 1)), len(cand))
        pick = np.sort(rng.choice(len(cand), size=k, replace=False))
        nz, nf = out["x"].shape
        src_x = np.zeros((self.k_max, nz, nf), dtype=np.float32)
        src_len = np.zeros(self.k_max, dtype=np.int64)
        src_ctx = np.zeros((self.k_max, out["ctx"].shape[0]), dtype=np.float32)
        src_dxy = np.zeros((self.k_max, 2), dtype=np.float32)
        for s, p in enumerate(pick):
            src_x[s], _, src_len[s], src_ctx[s] = self._profile(f, int(cand[p]))
            src_dxy[s] = dx[p], dy[p]
        out.update(src_x=torch.from_numpy(src_x), src_len=torch.from_numpy(src_len), src_ctx=torch.from_numpy(src_ctx),
                   src_dxy=torch.from_numpy(src_dxy))
        return out


def make_dataset(cfg: Config, grid: dict, stats, years, train: bool, **kw) -> ProfileDataset:
    if cfg.experiment == "neighbours":
        return NeighbourDataset(cfg, grid, stats, years, train=train, **kw)
    return ProfileDataset(cfg, grid, stats, years, **kw)
