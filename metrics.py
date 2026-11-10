"""
Scores of the FGN ensemble, the same for every predicted variable: fair CRPS, error of the member median (MAE, RMSE,
bias = median - truth), coverage of the member range, spread-skill. Accumulated per row (D_input, spacing or distance
to the nearest cast) and model level, per bin of the TRUE sigma0 of the predicted point, per lat/lon cell and density
layer, per month of the record, per calendar month and level, and per season and lat/lon cell. Plus the rank
histogram (calibration), the coherence of the members (between levels and between variables) and the oxycline
(O2 < threshold) as a depth and as a density, from whole member profiles.
Evaluator.finish() returns everything as one dict of numpy arrays (saved as one .npz file, read by figures.py).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist

from model import aux_feature_indices

UNITS = {"o2": "mmol m$^{-3}$", "thetao": "°C", "so": "psu", "sigma0": "kg m$^{-3}$", "chl": "log$_{10}$ mg m$^{-3}$",
         "no3": "mmol m$^{-3}$", "po4": "mmol m$^{-3}$", "dissic": "mol m$^{-3}$", "talk": "mol m$^{-3}$", "ph": "-"}
LONG_NAMES = {"o2": "O$_2$", "thetao": "temperature", "so": "salinity", "sigma0": "$\\sigma_0$",
              "chl": "chlorophyll (log$_{10}$)", "no3": "nitrate", "po4": "phosphate", "dissic": "DIC",
              "talk": "alkalinity", "ph": "pH"}
SHORT_NAMES = {"o2": "O$_2$", "thetao": "T", "so": "S", "sigma0": "$\\sigma_0$", "chl": "chl", "no3": "NO$_3$",
               "po4": "PO$_4$", "dissic": "DIC", "talk": "TA", "ph": "pH"}
POINT_KEYS = ("pinball", "abs_err", "sq_err", "bias", "covered", "ens_var", "sq_err_mean")
MAP_KEYS = ("pinball", "bias", "abs_err", "covered", "ens_var", "sq_err_mean")
TIME_KEYS = ("pinball", "bias", "abs_err")
SEASON_KEYS = ("pinball", "bias")


def _all_reduce(tensors) -> None:
    if dist.is_available() and dist.is_initialized():
        for t in tensors:
            dist.all_reduce(t)


# --------------------------------------------------------------------------------------- ensemble scores
def _pair_sum(srt: torch.Tensor, dim: int) -> torch.Tensor:
    """sum over member pairs i < j of |x_i - x_j|, from the sorted members."""
    m = srt.shape[dim]
    shape = [1] * srt.dim()
    shape[dim] = m
    weight = (2.0 * torch.arange(m, device=srt.device, dtype=srt.dtype) - (m - 1)).reshape(shape)
    return (srt * weight).sum(dim=dim)


def crps_ensemble(samples: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, dim: int = 1) -> torch.Tensor:
    """(Almost-)fair CRPS, members along `dim`: alpha = 1 fair CRPS, alpha < 1 mixes in the empirical one."""
    m = samples.shape[dim]
    skill = (samples - target.unsqueeze(dim)).abs().mean(dim=dim)
    if m < 2:
        return skill
    eps = (1.0 - float(alpha)) / m
    return skill - (1.0 - eps) * _pair_sum(samples.sort(dim=dim).values, dim) / (m * (m - 1))


def mean_member_distance(samples: torch.Tensor, dim: int = 1) -> torch.Tensor:
    m = samples.shape[dim]
    if m < 2:
        return torch.zeros_like(samples.select(dim, 0))
    return 2.0 * _pair_sum(samples.sort(dim=dim).values, dim) / (m * (m - 1))


def ensemble_median(samples: torch.Tensor, dim: int = 1) -> torch.Tensor:
    m = samples.shape[dim]
    srt = samples.sort(dim=dim).values
    return srt.select(dim, m // 2) if m % 2 else 0.5 * (srt.select(dim, m // 2 - 1) + srt.select(dim, m // 2))


@torch.no_grad()
def point_metrics(samples: torch.Tensor, y: torch.Tensor, tol: float) -> Dict[str, torch.Tensor]:
    """samples (B, M, T), truth y (B, T), standardised -> (B, T) per-point quantities (summed by the accumulators)."""
    m = samples.shape[1]
    srt = samples.sort(dim=1).values
    median = srt[:, m // 2] if m % 2 else 0.5 * (srt[:, m // 2 - 1] + srt[:, m // 2])
    mean = samples.mean(dim=1)
    return {"pinball": 0.5 * crps_ensemble(samples, y, 1.0, dim=1), "abs_err": (y - median).abs(),
            "sq_err": (y - median) ** 2, "bias": median - y,
            "covered": (y >= srt[:, 0] - tol) & (y <= srt[:, -1] + tol),
            "ens_var": ((samples - mean.unsqueeze(1)) ** 2).sum(dim=1) / max(m - 1, 1), "sq_err_mean": (y - mean) ** 2}


def truth_rank(samples: torch.Tensor, y: torch.Tensor, tol: float, generator) -> torch.Tensor:
    """Rank of the truth among the M members, 0..M (B, T). A calibrated ensemble gives every rank equally often.
    Members within tol of the truth are ties, broken at random (an O2 of exactly 0 in the anoxic layer)."""
    below = (samples < (y - tol).unsqueeze(1)).sum(dim=1)
    ties = ((samples - y.unsqueeze(1)).abs() <= tol).sum(dim=1)
    return below + (torch.rand(y.shape, generator=generator, device=y.device) * (ties + 1)).long()


def coherence_sums(members: torch.Tensor, truth: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, generator):
    """What the coherence of the members is worth. members (B, M, T, V), truth (B, T, V), standardised.
    The members are scored on quantities that depend on their coherence: the difference between two consecutive
    levels, and the difference between two variables at a level. The same is done after shuffling the members
    independently at every level and variable, which keeps every error bar but destroys the coherence.
    -> summed CRPS [as predicted, shuffled] of the level differences (2, V) and of every pair of variables (2, P)."""
    shuffled = members.gather(1, torch.rand(members.shape, generator=generator, device=members.device).argsort(dim=1))
    step = (mask[:, 1:] & mask[:, :-1] & (idx[:, 1:] == idx[:, :-1] + 1))[..., None]  # two consecutive predicted levels
    a, b = torch.triu_indices(members.shape[-1], members.shape[-1], offset=1, device=members.device)
    levels, pairs = [], []
    for e in (members, shuffled):
        levels.append((crps_ensemble(e[:, :, 1:] - e[:, :, :-1], truth[:, 1:] - truth[:, :-1]) * step).sum(dim=(0, 1)))
        pairs.append((crps_ensemble(e[..., a] - e[..., b], truth[..., a] - truth[..., b]) * mask[..., None]).sum(dim=(0, 1)))
    return torch.stack(levels).double(), torch.stack(pairs).double()


def coherence(sums: np.ndarray) -> np.ndarray:
    """Sums of coherence_sums per row (rows, 2, ...), pooled over the rows -> 1 - CRPS as predicted / CRPS shuffled:
    0 = the members are worth no more than independent draws, 1 = perfect, < 0 = their coherence is wrong."""
    s = np.asarray(sums, dtype=np.float64).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - s[0] / s[1]


# --------------------------------------------------------------------------------------- whole profiles
def fill_columns(truth: torch.Tensor, members: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Whole columns per member: the truth everywhere except at the predicted levels. -> (B, M, Z)."""
    bsz, nz = truth.shape
    n_t = idx.shape[1]
    dest = torch.where(mask, idx, torch.full_like(idx, nz))
    pos = torch.full((bsz, nz + 1), -1, dtype=torch.long, device=truth.device)
    pos.scatter_(1, dest, torch.arange(n_t, device=truth.device).expand(bsz, n_t).contiguous())
    pos = pos[:, :nz]
    got = members.gather(2, pos.clamp(min=0).unsqueeze(1).expand(-1, members.shape[1], -1))
    return torch.where((pos >= 0).unsqueeze(1), got, truth.unsqueeze(1).to(members.dtype))


def crossing_depth(o2: torch.Tensor, valid: torch.Tensor, depth: torch.Tensor, thr: float):
    """Shallowest depth where O2 drops below thr (linear between levels) -> depth, found, first level below thr.
    Never below thr: the deepest valid level; below thr already at the first level: the first level."""
    nz = o2.shape[-1]
    valid = valid.expand_as(o2)
    lev = torch.arange(nz, device=o2.device)
    first = torch.where((o2 < thr) & valid, lev, torch.full_like(lev, nz)).amin(dim=-1)
    any_below = first < nz
    j = first.clamp(max=nz - 1)
    found = any_below & (j > 0)
    jm = (j - 1).clamp(min=0)
    a, b = o2.gather(-1, jm.unsqueeze(-1)).squeeze(-1), o2.gather(-1, j.unsqueeze(-1)).squeeze(-1)
    z = depth[jm] + ((a - thr) / (a - b).clamp(min=1e-9)).clamp(0.0, 1.0) * (depth[j] - depth[jm])
    bottom = depth[(valid.sum(dim=-1) - 1).clamp(min=0)]
    return torch.where(found, z, torch.where(any_below, depth[0].expand_as(z), bottom)), found, j


def interp_profile(values: torch.Tensor, depth: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """values (n, Z) at depth z (n,) or (n, M), linear between levels."""
    squeeze = z.dim() == 1
    zz = (z.unsqueeze(1) if squeeze else z).to(depth.dtype).contiguous()
    j = torch.searchsorted(depth.contiguous(), zz).clamp(1, depth.numel() - 1)
    w = ((zz - depth[j - 1]) / (depth[j] - depth[j - 1]).clamp(min=1e-9)).clamp(0.0, 1.0)
    v0, v1 = values.gather(1, j - 1), values.gather(1, j)
    out = v0 + w * (v1 - v0)
    return out.squeeze(1) if squeeze else out


class OxyclineAccumulator:
    """Oxycline depth (m) and its sigma0 (kg m-3) from whole member profiles, per row: CRPS, |median - truth|,
    coverage of the member range. Only profiles whose true oxycline exists and is not pinned by observations."""

    KEYS = ("crps", "abs_err", "covered", "count", "crps_sigma", "abs_err_sigma", "covered_sigma")

    def __init__(self, n_rows: int, device, threshold: float):
        self.threshold = float(threshold)
        self.sums = {k: torch.zeros(n_rows, dtype=torch.float64, device=device) for k in self.KEYS}

    @torch.no_grad()
    def update(self, row, truth, members, idx, mask, observed, valid_len, depth, sigma=None) -> Optional[dict]:
        """truth (B, Z), members (B, M, T) in mmol m-3, sigma (B, Z) true sigma0 or None. Returns the per-profile
        scores (ok (B,) and crps / abs_err / covered (n_ok,)) for the maps."""
        lev = torch.arange(truth.shape[1], device=truth.device)
        valid = lev[None, :] < valid_len[:, None]
        z_true, found, j = crossing_depth(truth, valid, depth, self.threshold)
        obs = observed & valid
        pinned = obs.gather(1, j[:, None]).squeeze(1) & obs.gather(1, (j - 1).clamp(min=0)[:, None]).squeeze(1)
        ok = found & ~pinned & mask.any(dim=1)
        if not bool(ok.any()):
            return None
        cols = fill_columns(truth[ok], members[ok].float(), idx[ok], mask[ok])
        z_mem, _, _ = crossing_depth(cols, valid[ok].unsqueeze(1), depth, self.threshold)
        zt = z_true[ok]

        def score(pred, true):
            covered = (true >= pred.min(dim=1).values) & (true <= pred.max(dim=1).values)
            return crps_ensemble(pred, true, 1.0, dim=1), (true - ensemble_median(pred, dim=1)).abs(), covered

        crps, abs_err, covered = score(z_mem, zt)
        values = {"crps": crps, "abs_err": abs_err, "covered": covered.double(), "count": torch.ones_like(zt)}
        if sigma is not None:
            sig = sigma[ok].float()
            s_crps, s_abs, s_cov = score(interp_profile(sig, depth, z_mem), interp_profile(sig, depth, zt))
            values.update(crps_sigma=s_crps, abs_err_sigma=s_abs, covered_sigma=s_cov.double())
        for key, val in values.items():
            self.sums[key][row] += val.double().sum()
        return {"ok": ok, "crps": crps, "abs_err": abs_err, "covered": covered}


# --------------------------------------------------------------------------------------- bins and weights
def density_edges(cfg) -> np.ndarray:
    lo, hi = (float(v) for v in cfg.density_range)
    w = float(cfg.density_bin_width)
    return np.round(np.arange(lo, hi + 0.5 * w, w), 6)


def layer_labels(edges: Sequence[float]):
    e = [f"{float(v):g}" for v in edges]
    return [f"<{e[0]}"] + [f"{a}-{b}" for a, b in zip(e[:-1], e[1:])] + [f">{e[-1]}"] if e else ["all"]


def loss_weights(cfg, names: Sequence[str]) -> Optional[np.ndarray]:
    """Weight of every predicted variable in the loss and the validation objective (sum 1); None = "o2" weighting."""
    if str(cfg.variable_weighting).lower() != "equal" or len(names) < 2:
        return None
    extra = dict(cfg.variable_weights or {})
    unknown = [k for k in extra if k not in names]
    if unknown:
        raise ValueError(f"variable_weights names {unknown} are not predicted variables {list(names)}")
    w = np.asarray([float(extra.get(n, 1.0)) for n in names], dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"variable_weights must be >= 0 with a positive sum, got {extra}")
    return w / w.sum()


def combine_losses(cfg, loss_o2, loss_aux, weights=None):
    """weights: sum of weight x loss; None: O2 loss + aux_loss_weight x mean auxiliary loss."""
    if loss_aux is None:
        return loss_o2
    if weights is None:
        return loss_o2 + cfg.aux_loss_weight * loss_aux.mean()
    return weights[0] * loss_o2 + (weights[1:] * loss_aux).sum()


# --------------------------------------------------------------------------------------- evaluator
def scores(s: Dict[str, np.ndarray], n: np.ndarray, std: np.ndarray, n_members: int) -> Dict[str, np.ndarray]:
    """Sums of the point metrics (variable, ...) and the number of points (...) -> scores in physical units."""
    std = std.reshape((-1,) + (1,) * n.ndim)
    corr = (n_members + 1.0) / n_members
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"count": np.broadcast_to(n, s["pinball"].shape).copy(), "crps": 2.0 * s["pinball"] / n * std,
                "mae": s["abs_err"] / n * std, "rmse": np.sqrt(s["sq_err"] / n) * std, "bias": s["bias"] / n * std,
                "coverage": s["covered"] / n, "spread": np.sqrt(corr * s["ens_var"] / n) * std,
                "rmse_ensmean": np.sqrt(s["sq_err_mean"] / n) * std,
                "spread_skill": np.sqrt(corr * s["ens_var"] / n) / np.sqrt(s["sq_err_mean"] / n)}


class Evaluator:
    """Accumulates every score of one evaluation. update() once per (batch, row); finish() -> arrays.

    A score is a sum over the scored points (a level of a profile), kept in bins. The same points are binned in
    several ways, the "views" (prefix of the arrays in the metrics file):
      v   row x model level                        d   row x bin of the true sigma0
      m   row x water-mass layer x map cell        t   row x water-mass layer x month of the record
      s   calendar month x model level             ms  season (DJF, MAM, JJA, SON) x map cell
    plus the rank histogram, the coherence of the members and the oxycline."""

    def __init__(self, cfg, grid: dict, stats, rows: Sequence[float], row_name: str, device):
        self.cfg, self.rows, self.row_name = cfg, list(rows), row_name
        n_rows = len(self.rows)
        self.depth = torch.as_tensor(np.asarray(grid["depth"]), dtype=torch.float32, device=device)
        feats = list(grid["features"])
        self.ti = int(grid["target_index"])
        aux_names, aux_feat = aux_feature_indices(cfg, grid)
        self.names, self.feat = [grid["target"]] + aux_names, [self.ti] + aux_feat
        self.std = np.array([float(stats["std"][f]) for f in self.feat])
        self.o2_mean, self.o2_std = float(stats["mean"][self.ti]), float(stats["std"][self.ti])
        self.tol = float(cfg.coverage_tol_mmol_m3) / self.o2_std  # standardised, the same for every variable
        self.n_members = int(cfg.fgn_eval_members)
        nv, nz = len(self.names), len(grid["depth"])
        # density bins and water-mass layers, from the true sigma0 of every point
        self.si = feats.index("sigma0") if "sigma0" in feats else None
        self.edges = density_edges(cfg) if self.si is not None else np.zeros(0)
        self.layer_edges = np.asarray(cfg.density_layers if self.si is not None else (), dtype=np.float64)
        if self.si is not None:
            self.sig_mean, self.sig_std = float(stats["mean"][self.si]), float(stats["std"][self.si])
            self._inner = torch.as_tensor(self.edges[1:-1], dtype=torch.float32, device=device)
            self._layers = torch.as_tensor(self.layer_edges, dtype=torch.float32, device=device)
        # map cells = blocks of whole model columns, edges halfway between grid points
        lat, lon = np.asarray(grid["lat"], dtype=np.float64), np.asarray(grid["lon"], dtype=np.float64)
        dlat, dlon = (float(np.median(np.diff(a))) if a.size > 1 else float(cfg.map_resolution_deg) for a in (lat, lon))
        self.res_lat = max(1, int(round(cfg.map_resolution_deg / dlat))) * dlat
        self.res_lon = max(1, int(round(cfg.map_resolution_deg / dlon))) * dlon
        self.lat0, self.lon0 = lat.min() - dlat / 2, lon.min() - dlon / 2
        self.ny = int(np.floor((lat.max() - self.lat0) / self.res_lat)) + 1
        self.nx = int(np.floor((lon.max() - self.lon0) / self.res_lon)) + 1
        # months of every processed year
        years = sorted(int(y) for y in grid["years"])
        self.month0, self.n_t = (years[0] - 1970) * 12, (years[-1] - years[0] + 1) * 12

        nl, nc = len(self.layer_edges) + 1, self.ny * self.nx
        self.shape = {"v": (n_rows, nz), "d": (n_rows, len(self.edges) - 1), "m": (n_rows, nl, nc),
                      "t": (n_rows, nl, self.n_t), "s": (12, nz), "ms": (4, nc)}
        self.keys = {"v": POINT_KEYS, "d": POINT_KEYS, "m": MAP_KEYS, "t": TIME_KEYS, "s": SEASON_KEYS, "ms": SEASON_KEYS}
        if self.si is None:
            del self.shape["d"]

        def zeros(*shape):
            return torch.zeros(*shape, dtype=torch.float64, device=device)

        self.sums = {view: {k: zeros(nv, int(np.prod(shape))) for k in self.keys[view]} for view, shape in self.shape.items()}
        self.count = {view: zeros(int(np.prod(shape))) for view, shape in self.shape.items()}
        self.rank = zeros(nv, n_rows * (self.n_members + 1))  # rank of the truth among the members
        self.coh_levels, self.coh_pairs = zeros(n_rows, 2, nv), zeros(n_rows, 2, nv * (nv - 1) // 2)
        self.oxy = OxyclineAccumulator(n_rows, device, cfg.oxycline_o2_threshold)
        self.map_oxy = {k: zeros(n_rows, nc) for k in ("crps", "abs_err", "covered", "count")}
        self.gen = torch.Generator(device=device)  # ties of the rank histogram, shuffling of the coherence score
        self.gen.manual_seed(int(cfg.seed))

    @torch.no_grad()
    def update(self, row, pred, aux, x, idx, mask, observed, valid_len, x_raw, lat, lon, month):
        """pred O2 members (B, M, T), aux (B, M, T, A) or None, x the model's (drift-shifted) standardised input,
        x_raw the same without drift removal. Returns O2 truth (B, Z) and members (B, M, T) in mmol m-3."""
        nz, nc, nl = self.depth.numel(), self.ny * self.nx, len(self.layer_edges) + 1
        sig_prof, layer = None, torch.zeros_like(idx)
        iy = torch.floor((lat.double() - self.lat0) / self.res_lat).long().clamp(0, self.ny - 1)
        ix = torch.floor((lon.double() - self.lon0) / self.res_lon).long().clamp(0, self.nx - 1)
        cell = (iy * self.nx + ix)[:, None]
        t = (month.long() - self.month0).clamp(0, self.n_t - 1)[:, None]
        cal = (month.long() % 12)[:, None]  # 0 = January
        bins = {}
        if self.si is not None:
            sig_prof = x_raw[..., self.si].float() * self.sig_std + self.sig_mean
            sig_t = sig_prof.gather(1, idx)
            layer = torch.bucketize(sig_t, self._layers, right=True)
            bins["d"] = row * self.shape["d"][1] + torch.bucketize(sig_t, self._inner, right=True)
        bins.update(v=row * nz + idx, m=(row * nl + layer) * nc + cell, t=(row * nl + layer) * self.n_t + t,
                    s=cal * nz + idx, ms=((cal + 1) % 12 // 3) * nc + cell)
        scored = mask.reshape(-1)
        bins = {view: b.expand_as(idx).reshape(-1)[scored] for view, b in bins.items()}  # bin of every scored point
        for view, b in bins.items():
            self.count[view].index_add_(0, b, torch.ones_like(b, dtype=torch.float64))
        for v, f in enumerate(self.feat):
            y = torch.gather(x[..., f], 1, idx).float()
            members = (pred if v == 0 else aux[..., v - 1]).float()
            pm = {k: val.reshape(-1)[scored].double() for k, val in point_metrics(members, y, self.tol).items()}
            for view, b in bins.items():
                for k in self.keys[view]:
                    self.sums[view][k][v].index_add_(0, b, pm[k])
            if self.n_members > 1:
                rank = truth_rank(members, y, self.tol, self.gen).reshape(-1)[scored]
                self.rank[v].index_add_(0, row * (self.n_members + 1) + rank, torch.ones_like(rank, dtype=torch.float64))
        if self.n_members > 1:
            every = pred.float().unsqueeze(-1) if aux is None else torch.cat([pred.float().unsqueeze(-1), aux.float()], dim=-1)
            feat = torch.as_tensor(self.feat, device=idx.device)
            truth_all = torch.gather(x.index_select(-1, feat), 1, idx[..., None].expand(-1, -1, len(self.feat))).float()
            levels, pairs = coherence_sums(every, truth_all, idx, mask, self.gen)
            self.coh_levels[row] += levels
            self.coh_pairs[row] += pairs
        truth = x[..., self.ti].float() * self.o2_std + self.o2_mean  # O2 is never drift-shifted
        members = pred.float() * self.o2_std + self.o2_mean
        sc = self.oxy.update(row, truth, members, idx, mask, observed, valid_len, self.depth, sig_prof)
        if sc is not None:
            for k, val in (("crps", sc["crps"]), ("abs_err", sc["abs_err"]), ("covered", sc["covered"]),
                           ("count", torch.ones_like(sc["crps"]))):
                self.map_oxy[k][row].index_add_(0, cell[sc["ok"], 0], val.double())
        return truth, members

    def finish(self) -> Dict[str, np.ndarray]:
        _all_reduce([t for group in self.sums.values() for t in group.values()] + list(self.count.values())
                    + [self.rank, self.coh_levels, self.coh_pairs] + list(self.oxy.sums.values())
                    + list(self.map_oxy.values()))
        nv, n_rows, grid2d = len(self.names), len(self.rows), (self.ny, self.nx)

        def sums(view):  # {key: (variable, *bins of the view)}
            return {k: t.cpu().numpy().reshape((nv,) + self.shape[view]) for k, t in self.sums[view].items()}

        def count(view):
            return self.count[view].cpu().numpy().reshape(self.shape[view])

        out = {"rows": np.asarray(self.rows), "row_name": np.asarray(self.row_name),
               "experiment": np.asarray(self.cfg.experiment), "level_depths": self.depth.cpu().numpy().astype(np.float64),
               "variables": np.asarray(self.names), "units": np.asarray([UNITS.get(n, "") for n in self.names]),
               "var_std": self.std, "n_members": np.asarray(self.n_members), "layer_edges": self.layer_edges,
               "v_loss": sums("v")["pinball"].sum(axis=(1, 2)) / max(count("v").sum(), 1.0)}
        for view in ("v", "d"):  # scores per row and level / density bin
            if view in self.shape:
                out.update({f"{view}_{k}": val for k, val in scores(sums(view), count(view), self.std, self.n_members).items()})
        if "d" in self.shape:
            out["density_edges"] = self.edges
        o = {k: v.cpu().numpy() for k, v in self.oxy.sums.items()}
        with np.errstate(invalid="ignore", divide="ignore"):
            out.update(oxy_count=o["count"], oxy_crps=o["crps"] / o["count"], oxy_mae=o["abs_err"] / o["count"],
                       oxy_coverage=o["covered"] / o["count"], oxy_crps_sigma=o["crps_sigma"] / o["count"],
                       oxy_mae_sigma=o["abs_err_sigma"] / o["count"], oxy_coverage_sigma=o["covered_sigma"] / o["count"])
        # maps: sums per cell (figures.py pools rows and layers before dividing)
        for k, val in sums("m").items():
            out[f"m_{k}"] = val.reshape(val.shape[:3] + grid2d).astype(np.float32)
        out["m_count"] = count("m").reshape((n_rows, -1) + grid2d).astype(np.float32)
        for k, val in self.map_oxy.items():
            out[f"m_oxy_{k}"] = val.cpu().numpy().reshape((n_rows,) + grid2d).astype(np.float32)
        for k, val in sums("ms").items():
            out[f"ms_{k}"] = val.reshape((nv, 4) + grid2d).astype(np.float32)
        out["ms_count"] = count("ms").reshape((4,) + grid2d).astype(np.float32)
        out["map_lat"] = self.lat0 + self.res_lat * (np.arange(self.ny) + 0.5)
        out["map_lon"] = self.lon0 + self.res_lon * (np.arange(self.nx) + 0.5)
        # months of the record that hold data, and the calendar months
        keep = np.nonzero(count("t").sum(axis=(0, 1)) > 0)[0]
        out["t_months"], out["t_count"] = (self.month0 + keep).astype(np.int64), count("t")[..., keep].astype(np.float32)
        for k, val in sums("t").items():
            out[f"t_{k}"] = val[..., keep].astype(np.float32)
        out.update({f"s_{k}": val for k, val in sums("s").items()}, s_count=count("s"))
        # calibration and coherence (ensembles only)
        out["rank"] = self.rank.cpu().numpy().reshape(nv, n_rows, self.n_members + 1)
        out["coh_levels"] = self.coh_levels.cpu().numpy()
        pairs = np.zeros((n_rows, 2, nv, nv))
        a, b = np.triu_indices(nv, 1)
        pairs[..., a, b] = pairs[..., b, a] = self.coh_pairs.cpu().numpy()
        out["coh_variables"] = pairs
        out["objective"] = np.asarray(objective(self.cfg, out))
        return out


# --------------------------------------------------------------------------------------- summaries
def objective(cfg, r) -> float:
    """What selects best.pt (standardised units): the weighted mean loss over variables ("equal"), else the O2 loss."""
    w = loss_weights(cfg, [str(n) for n in r["variables"]])
    return float(r["v_loss"][0]) if w is None else float((w * r["v_loss"]).sum())


def pooled(values: np.ndarray, count: np.ndarray, axis=None) -> np.ndarray:
    w = count.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.nan_to_num(values.astype(np.float64)) * w).sum(axis=axis) / w.sum(axis=axis)


def scalars(r: Dict[str, np.ndarray], prefix: str) -> Dict[str, float]:
    """Pooled scores of an evaluation, for the log and W&B."""
    out = {f"{prefix}/loss": float(r["v_loss"][0]), f"{prefix}/loss_total": float(r["objective"])}
    labels = layer_labels(r["layer_edges"])
    for v, name in enumerate(r["variables"]):
        n = r["v_count"][v]
        base = f"{prefix}/vars/{name}"
        out[f"{base}/crps"] = float(pooled(r["v_crps"][v], n))
        out[f"{base}/crps_rel"] = out[f"{base}/crps"] / float(r["var_std"][v])
        for k in ("mae", "bias", "coverage"):
            out[f"{base}/{k}"] = float(pooled(r[f"v_{k}"][v], n))
        out[f"{base}/spread_skill"] = float(np.sqrt(pooled(r["v_spread"][v] ** 2, n) / pooled(r["v_rmse_ensmean"][v] ** 2, n)))
        if len(labels) > 1:
            pin = r["m_pinball"][v].sum(axis=(0, 2, 3)).astype(np.float64)
            cnt = r["m_count"].sum(axis=(0, 2, 3)).astype(np.float64)
            for lab, p, c in zip(labels, pin, cnt):
                if c > 0:
                    out[f"{base}/crps_layer/{lab}"] = float(2.0 * p / c * r["var_std"][v])
    unit = str(r["row_name"]).rpartition("(")[2].rstrip(")")
    for i, row in enumerate(r["rows"]):
        out[f"{prefix}/crps_by_row/{float(row):g}{unit}"] = float(pooled(r["v_crps"][0][i], r["v_count"][0][i]))
    c = r["oxy_count"]
    if c.sum() > 0:
        for k in ("crps", "mae", "coverage", "crps_sigma", "mae_sigma", "coverage_sigma"):
            out[f"{prefix}/oxycline/{k}"] = float(pooled(r[f"oxy_{k}"], c))
    if "coh_levels" in r and int(r["n_members"]) > 1:  # see coherence()
        for name, value in zip(r["variables"], coherence(r["coh_levels"])):
            out[f"{prefix}/vars/{name}/coherence_levels"] = float(value)
        if len(r["variables"]) > 1:
            out[f"{prefix}/coherence_variables"] = float(coherence(r["coh_variables"].sum(axis=(2, 3))[..., None])[0])
    return out


def summary_line(r: Dict[str, np.ndarray]) -> str:
    s = scalars(r, "x")
    o2 = "x/vars/o2"
    line = (f"objective {s['x/loss_total']:.4f} | O2 CRPS {s[o2 + '/crps']:.2f} MAE {s[o2 + '/mae']:.2f} mmol m-3, "
            f"coverage {s[o2 + '/coverage']:.3f}, spread/skill {s[o2 + '/spread_skill']:.2f}")
    if "x/oxycline/crps" in s:
        line += f" | oxycline CRPS {s['x/oxycline/crps']:.1f} m, {s['x/oxycline/crps_sigma']:.3f} kg m-3"
    if o2 + "/coherence_levels" in s:
        line += f" | O2 coherence between levels {s[o2 + '/coherence_levels']:.2f}"
        line += f", between variables {s['x/coherence_variables']:.2f}" if "x/coherence_variables" in s else ""
    return line + " | CRPS " + ", ".join(f"{n} {s[f'x/vars/{n}/crps']:.3g}" for n in r["variables"])
