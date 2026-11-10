#!/usr/bin/env python
"""
Train or evaluate the FGN profile model, every experiment (config `experiment`), one GPU or DDP via torchrun.

    python train.py --set experiment=cutoff input_drift_mode=lagged        # observed surface -> D_input
    python train.py --set experiment=sparse input_drift_mode=shallowest    # observed every `spacing` m
    python train.py --set experiment=neighbours neighbour_max_km=50        # whole profile from nearby casts
    python train.py --eval-only --split val --max-profiles 50000 --set ckpt_dir=<run> processed_dir=<data>

Training resumes from <ckpt_dir>/last.pt. Every validation writes the scores (metrics.py) and the best checkpoint
best.pt; --eval-only writes <ckpt_dir>/<split>_metrics[_TAG].npz, and for "neighbours" also the scores of the
inverse-distance-weighted mean of the casts (<split>_metrics[_TAG]_idw.npz).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import FROZEN_ON_RESUME, Config, apply_overrides, config_from_checkpoint
from dataset import (DRIFT_MODES, cutoff_levels, drift_indices, load_grid, load_or_compute_basin_monthly,
                     load_or_compute_stats, make_dataset, max_input_levels, resolve_splits, subtract_shallowest)
from metrics import (Evaluator, combine_losses, crps_ensemble, ensemble_median, fill_columns, loss_weights,
                     mean_member_distance, scalars, summary_line)
from model import aux_feature_indices, build_model, gather_levels, observation_mask

log = logging.getLogger("train")
FGN_MAX_SEQUENCES = 19 * 1024  # profile copies per forward pass in evaluation when fgn_member_chunk = 0
NEIGHBOUR_KEYS = ("src_x", "src_len", "src_ctx", "src_dxy")


# --------------------------------------------------------------------------------------- infrastructure
def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(minutes=60))
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def amp_mode(cfg: Config, device) -> str:
    if device.type != "cuda":
        return "fp32"
    if cfg.amp_dtype != "auto":
        return cfg.amp_dtype
    major, _ = torch.cuda.get_device_capability(device)
    return "bf16" if major >= 8 else ("fp16" if major >= 7 else "fp32")


def autocast(cfg: Config, device):
    mode = amp_mode(cfg, device)
    return nullcontext() if mode == "fp32" else torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def make_loader(ds, cfg: Config, batch_size: int, train: bool, ddp: bool):
    sampler = DistributedSampler(ds, shuffle=train, drop_last=train, seed=cfg.seed) if ddp else None
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None)


def make_optimizer(model, cfg: Config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "token_type" in name else decay).append(p)
    fused = {"fused": True} if torch.cuda.is_available() else {}
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg.lr, betas=(0.9, 0.95), **fused)


def make_scheduler(opt, cfg: Config, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def init_wandb(cfg: Config, run_id: Optional[str], job_type: str, extra: dict):
    import wandb

    project = cfg.wandb_project or "blacksea-o2-" + {"cutoff": "profiles"}.get(cfg.experiment, cfg.experiment)
    return wandb.init(project=project, entity=cfg.wandb_entity, name=cfg.wandb_run_name, id=run_id,
                      resume="allow" if run_id else None, mode=os.environ.get("WANDB_MODE", cfg.wandb_mode),
                      job_type=job_type, dir=cfg.ckpt_dir, config={**cfg.to_dict(), **extra})


def wandb_log(r: dict, prefix: str, step: int, full: bool, examples: Optional[dict], extra: dict) -> None:
    """Scalars + figures (figures.py) of one evaluation in one wandb.log call."""
    import matplotlib.pyplot as plt
    import wandb

    import figures

    payload = {**scalars(r, prefix), **extra}
    for name, fig in figures.wandb_figures(r, full, examples).items():
        payload[f"{prefix}/{name}"] = wandb.Image(fig)
        plt.close(fig)
    wandb.log(payload, step=step)


class WeightEMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.model = copy.deepcopy(unwrap(model)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for avg, p in zip(self.model.parameters(), unwrap(model).parameters()):
            avg.lerp_(p.detach(), 1.0 - d)


def save_checkpoint(path: Path, **payload) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # a job killed mid-save never corrupts last.pt


def check_config(cfg: Config, depth) -> None:
    if cfg.experiment not in ("cutoff", "sparse", "neighbours"):
        raise ValueError(f"experiment must be 'cutoff', 'sparse' or 'neighbours', not {cfg.experiment!r}")
    if cfg.input_drift_mode not in DRIFT_MODES:
        raise ValueError(f"input_drift_mode must be one of {DRIFT_MODES}, not {cfg.input_drift_mode!r}")
    if cfg.fgn_train_members < 2 or cfg.fgn_eval_members < 2 or not 0.0 <= cfg.fgn_crps_alpha <= 1.0:
        raise ValueError("fgn_train_members and fgn_eval_members must be >= 2, fgn_crps_alpha in [0, 1]")
    if cfg.input_drift_mode == "lagged" and (cfg.drift_window_years < 1 or cfg.drift_lag_years < 0):
        raise ValueError("drift_window_years must be >= 1 and drift_lag_years >= 0")
    if cfg.experiment == "cutoff":
        ks, d_in = cutoff_levels(depth, cfg.val_cutoff_depths_m, cfg.min_input_levels)
        too_deep = [d for k, d in zip(ks, d_in) if k > max_input_levels(depth, cfg.max_input_depth_m)]
        if not ks or too_deep:
            raise ValueError(f"val_cutoff_depths_m give D_input {d_in}; none may exceed max_input_depth_m="
                             f"{cfg.max_input_depth_m} (the model never trains on deeper cutoffs)")
    elif cfg.experiment == "sparse" and (min(cfg.sparse_train_spacings_m) <= 0 or min(cfg.sparse_val_spacings_m) <= 0):
        raise ValueError("sparse spacings must be positive")
    elif cfg.experiment == "neighbours":
        if cfg.input_drift_mode == "shallowest":
            raise ValueError("experiment=neighbours takes input_drift_mode none or lagged")
        if cfg.neighbour_casts < 1 or not 1 <= (cfg.neighbour_eval_casts or 1) <= cfg.neighbour_casts:
            raise ValueError("neighbour_casts must be >= 1 and neighbour_eval_casts in [1, neighbour_casts]")
        if max(cfg.neighbour_val_km) < cfg.neighbour_max_km:
            raise ValueError("the last neighbour_val_km bin must reach neighbour_max_km")


# --------------------------------------------------------------------------------------- what the model is given
def eval_rows(cfg: Config, depth):
    """Validation rows: (key of every row for Task.scored / Task.inputs, row values, row name)."""
    if cfg.experiment == "sparse":
        spacings = [float(s) for s in cfg.sparse_val_spacings_m]
        return spacings, spacings, "spacing (m)"
    if cfg.experiment == "neighbours":
        km = sorted(float(v) for v in cfg.neighbour_val_km)
        return km, km, "distance to the nearest cast (km)"
    ks, d_in = cutoff_levels(depth, cfg.val_cutoff_depths_m, cfg.min_input_levels)
    return ks, d_in, "D_input (m)"


def sparse_obs(cfg: Config, net, valid_len, spacing):
    first = spacing if cfg.sparse_first_depth_m is None else torch.full_like(spacing, float(cfg.sparse_first_depth_m))
    return observation_mask(net.depth, valid_len, spacing, first, cfg.sparse_max_obs_depth_m)


def sample_cutoffs(valid_len, k_min: int, k_max: int):
    """Uniform number of observed levels k in [k_min, min(valid_len - 1, k_max)] per profile."""
    hi = (valid_len - 1).clamp(max=k_max)
    lo = torch.full_like(hi, k_min).minimum(hi)
    offset = (torch.rand(hi.shape, device=hi.device) * (hi - lo + 1).float()).long()
    return lo + torch.minimum(offset, hi - lo)


def nominal_observed(cfg: Config, depth, spacings) -> np.ndarray:
    """Sparse: observed levels of each spacing on a full-depth profile, for the plots."""
    d = torch.as_tensor(np.asarray(depth), dtype=torch.float32)
    net = type("Grid", (), {"depth": d})
    return np.stack([sparse_obs(cfg, net, torch.tensor([len(depth)]), torch.tensor([float(s)]))[0].numpy()
                     for s in spacings])


class Task:
    """What a run predicts and what its model is given: the only place where the three experiments differ.
    Used by training, evaluation and figures.py."""

    def __init__(self, cfg: Config, grid: dict, stats, device):
        self.cfg = cfg
        self.ti = int(grid["target_index"])
        aux_names, aux_feat = aux_feature_indices(cfg, grid)
        self.names = [grid["target"]] + aux_names  # every predicted variable, O2 first
        self.feat = torch.as_tensor([self.ti] + aux_feat, dtype=torch.long, device=device)
        self.aux_idx = self.feat[1:] if aux_feat else None
        lw = loss_weights(cfg, self.names)
        self.loss_w = None if lw is None else torch.as_tensor(lw, dtype=torch.float32, device=device)
        self.target_std = float(stats["std"][self.ti])
        idx = drift_indices(cfg, grid) if cfg.input_drift_mode == "shallowest" else []
        self.drift_idx = torch.as_tensor(idx, dtype=torch.long, device=device) if idx else None
        self.k_lim = (max(1, cfg.min_input_levels), max_input_levels(grid["depth"], cfg.max_input_depth_m))
        self.spacings = torch.as_tensor([float(s) for s in cfg.sparse_train_spacings_m], device=device)
        self.km_edges = torch.as_tensor(sorted(float(v) for v in cfg.neighbour_val_km), device=device)

    def inputs(self, net, b: dict, key=None):
        """Batch b -> (inputs of the model, x holding the targets, observed levels (B, Z), profiles that are given
        something (B,)). key=None: training, a random D_input / spacing per profile; else the key of a validation
        row: the number of observed levels ("cutoff") or the spacing in m ("sparse")."""
        x, valid_len = b["x"], b["valid_len"]
        if self.cfg.experiment == "neighbours":  # nearby casts; no level of the predicted profile is observed
            obs = torch.zeros_like(x[..., 0], dtype=torch.bool)
            return (*[b[k] for k in NEIGHBOUR_KEYS], valid_len, b["ctx"]), x, obs, (b["src_len"] > 0).any(dim=1)
        if self.cfg.experiment == "sparse":
            spacing = (self.spacings[torch.randint(len(self.spacings), valid_len.shape, device=x.device)] if key is None
                       else torch.full(valid_len.shape, float(key), device=x.device))
            obs = sparse_obs(self.cfg, net, valid_len, spacing)
        else:
            k = sample_cutoffs(valid_len, *self.k_lim)[:, None] if key is None else key
            obs = (torch.arange(x.shape[1], device=x.device)[None, :] < k).expand(x.shape[0], -1)
        if self.drift_idx is not None:
            x = subtract_shallowest(x, obs, valid_len, self.drift_idx)
        return (x, obs, valid_len, b["ctx"]), x, obs, obs.any(dim=1)

    def scored(self, b: dict, i: int, key) -> torch.Tensor:
        """Profiles of batch b scored in validation row i (B,) bool."""
        if self.cfg.experiment == "cutoff":
            return b["valid_len"] > key  # at least one level below D_input
        if self.cfg.experiment == "sparse":
            return torch.ones_like(b["valid_len"], dtype=torch.bool)
        km = b["src_dxy"].float().norm(dim=-1).masked_fill(b["src_len"] == 0, float("inf")).amin(dim=1)
        return torch.bucketize(km, self.km_edges) == i  # distance bin of the nearest cast (no cast: no row)

    def losses(self, pred, aux, x, target_idx, mask):
        """Half the (almost-)fair CRPS over the real targets: O2 () and every other variable (A,) or None."""
        alpha = self.cfg.fgn_crps_alpha
        y = torch.gather(x[..., self.ti], 1, target_idx).float()
        n_real = mask.sum().clamp(min=1.0)
        loss_o2 = (0.5 * crps_ensemble(pred.float(), y, alpha, dim=1) * mask).sum() / n_real
        if aux is None:
            return loss_o2, None
        per_feature = 0.5 * crps_ensemble(aux.float(), gather_levels(x, target_idx, self.aux_idx).float(), alpha, dim=1)
        return loss_o2, (per_feature * mask.unsqueeze(-1)).sum(dim=(0, 1)) / n_real


def fgn_members(net, inputs: tuple, n_members: int, chunk: int = 0, generator=None, shared_noise=None):
    """All members of one batch, drawn `chunk` at a time: O2 (B, M, T), idx, mask, other variables (B, M, T, A) or
    None. Every profile draws its own noise, unless shared_noise (M, noise_dim) is given: then member m uses the
    same draw in every profile, which makes it one coherent field (figures.py field)."""
    bsz = inputs[0].shape[0]
    casts = inputs[0].shape[1] + 1 if inputs[0].dim() == 4 else 1  # "neighbours": about (K + 1) x longer sequences
    step = max(1, min(n_members, FGN_MAX_SEQUENCES // max(bsz * casts, 1))) if chunk <= 0 else min(chunk, n_members)
    o2, aux, out = [], [], None
    for start in range(0, n_members, step):
        m = min(step, n_members - start)
        if shared_noise is not None:
            noise = shared_noise[None, start:start + m].expand(bsz, -1, -1)
        elif generator is not None:
            noise = torch.randn(bsz, m, net.noise_dim, generator=generator, device=generator.device)
        else:
            noise = None
        out = net(*inputs, n_members=m, noise=noise)
        o2.append(out[0])
        aux.append(out[3])
    return torch.cat(o2, dim=1), out[1], out[2], (torch.cat(aux, dim=1) if aux[0] is not None else None)


def idw_baseline(src_x, src_len, src_dxy, idx, feat):
    """Inverse-distance-squared mean of the casts at every level, a cast counting down to its bottom (deeper than
    every cast: the deepest cast's last value) -> (B, 1, T, len(feat)), as a one-member ensemble."""
    nz, nv = src_x.shape[2], feat.numel()
    lev = torch.arange(nz, device=src_x.device)
    last = torch.minimum(lev[None, None, :], (src_len - 1).clamp(min=0)[..., None])
    xs = src_x.index_select(-1, feat).gather(2, last[..., None].expand(-1, -1, -1, nv))
    w = (src_len > 0).float() / src_dxy.float().norm(dim=-1).clamp(min=1.0) ** 2
    wz = w[..., None] * (lev[None, None, :] < src_len[..., None]).float()
    deepest = (w * (src_len == src_len.amax(dim=1, keepdim=True)).float())[..., None].expand_as(wz)
    wz = torch.where(wz.sum(dim=1, keepdim=True) > 0, wz, deepest)
    est = (wz[..., None] * xs.float()).sum(dim=1) / wz.sum(dim=1)[..., None].clamp(min=1e-12)
    return est.gather(1, idx[..., None].expand(-1, -1, nv)).unsqueeze(1)


# --------------------------------------------------------------------------------------- training / evaluation
def train_one_epoch(model, loader, opt, sched, scaler, cfg: Config, task: Task, device, state, rank, ema=None):
    model.train()
    net = unwrap(model)
    n_batches = len(loader) if cfg.steps_per_epoch is None else min(len(loader), cfg.steps_per_epoch)
    for it, batch in enumerate(loader):
        if it >= n_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        inputs, x, _, given = task.inputs(net, b)
        with autocast(cfg, device):
            pred, target_idx, target_mask, aux = model(*inputs, n_members=cfg.fgn_train_members)
        mask = (target_mask & given[:, None]).float()
        loss_o2, loss_aux = task.losses(pred, aux, x, target_idx, mask)
        loss = combine_losses(cfg, loss_o2, loss_aux, task.loss_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler.is_enabled() or torch.isfinite(grad_norm):
            scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        state["step"] += 1
        if ema is not None:
            ema.update(model, state["step"])
        if rank == 0 and state["step"] % cfg.log_every == 0:
            import wandb

            spread = (mean_member_distance(pred.detach().float(), dim=1) * mask).sum() / mask.sum().clamp(min=1.0)
            wandb.log({"train/loss": loss_o2.item(), "train/loss_total": loss.item(),
                       "train/crps_mmol_m3": 2.0 * loss_o2.item() * task.target_std,
                       "train/member_spread_mmol_m3": spread.item() * task.target_std, "train/lr": sched.get_last_lr()[0],
                       "train/grad_norm": float(grad_norm), "epoch": state["epoch"] + it / n_batches}, step=state["step"])
            log.info("epoch %d step %d loss %.4f lr %.2e", state["epoch"], state["step"], loss_o2.item(),
                     sched.get_last_lr()[0])


def example_profiles(truth, members, idx, mask, obs, valid_len, n: int) -> dict:
    """The n deepest scored profiles of a batch as whole columns, for the W&B figure."""
    keep = mask.any(dim=1)
    order = torch.sort(valid_len[keep], descending=True, stable=True).indices[:n]
    t = truth[keep][order]
    cols = fill_columns(t, members[keep][order], idx[keep][order], mask[keep][order])
    return {"truth": t.cpu().numpy(), "observed": obs[keep][order].cpu().numpy(), "members": cols.cpu().numpy(),
            "center": ensemble_median(cols, dim=1).cpu().numpy(), "valid_len": valid_len[keep][order].cpu().numpy()}


@torch.no_grad()
def evaluate(model, loader, cfg: Config, grid: dict, stats, task: Task, device) -> tuple:
    """Scores of every row (D_input, spacing or distance to the nearest cast) -> (metrics arrays, example profiles
    for W&B, metrics arrays of the inverse-distance-weighted mean of the casts or None)."""
    net = unwrap(model)
    net.eval()
    keys, rows, row_name = eval_rows(cfg, grid["depth"])
    ev = Evaluator(cfg, grid, stats, rows, row_name, device)
    ev_idw = None
    if cfg.experiment == "neighbours":  # the baseline is scored like a one-member ensemble
        ev_idw = Evaluator(dataclasses.replace(cfg, fgn_eval_members=1), grid, stats, rows, row_name, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed) + 1000 * (dist.get_rank() if dist.is_available() and dist.is_initialized() else 0))
    example_row = len(keys) // 2 if cfg.experiment != "cutoff" else int(np.argmin([abs(r - 50.0) for r in rows]))
    examples = None
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        for i, key in enumerate(keys):
            sel = task.scored(b, i, key)
            if not bool(sel.any()):
                continue
            sub = {k: v[sel] for k, v in b.items()}
            inputs, x, obs, given = task.inputs(net, sub, key)
            with autocast(cfg, device):
                pred, idx, mask, aux = fgn_members(net, inputs, cfg.fgn_eval_members, cfg.fgn_member_chunk, gen)
            mask = mask & given[:, None]
            where = (x, idx, mask, obs, sub["valid_len"], sub["x_raw"], sub["lat"], sub["lon"], sub["month"])
            truth, members = ev.update(i, pred.float(), None if aux is None else aux.float(), *where)
            if ev_idw is not None:
                base = idw_baseline(sub["src_x"], sub["src_len"], sub["src_dxy"], idx, task.feat)
                ev_idw.update(i, base[..., 0], base[..., 1:] if aux is not None else None, *where)
            if examples is None and i == example_row and cfg.example_profiles > 0 and bool(mask.any()):
                examples = {**example_profiles(truth, members, idx, mask, obs, sub["valid_len"], cfg.example_profiles),
                            "row": rows[i], "rows": rows, "row_name": row_name, "depth": np.asarray(grid["depth"])}
    r = ev.finish()
    if cfg.experiment == "sparse":
        r["observed"] = nominal_observed(cfg, grid["depth"], rows)
    return r, examples, (ev_idw.finish() if ev_idw is not None else None)


# --------------------------------------------------------------------------------------- entry points
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--resume", default="auto", help="'auto' (<ckpt_dir>/last.pt if present), 'none' or a path")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--split", default="val", choices=["val", "test"], help="test: once, at the very end")
    p.add_argument("--ckpt", default=None, help="checkpoint for --eval-only (default <ckpt_dir>/best.pt)")
    p.add_argument("--max-profiles", type=int, default=None, help="random fixed subset of the split (--eval-only)")
    p.add_argument("--tag", default="", help="--eval-only output <split>_metrics_TAG.npz, e.g. 1cast")
    return p.parse_args()


def stats_from_ckpt(ckpt: dict):
    return {k: np.asarray(v, dtype=np.float64) for k, v in ckpt["stats"].items()}


def load_run(path: Path, overrides, device):
    """A trained run (any version of this code): config (+ overrides), grid, stats and the model with the weights
    that validation selected (the EMA copy if used), in evaluation mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = apply_overrides(config_from_checkpoint(ckpt), overrides)
    grid, stats = ckpt["grid"], stats_from_ckpt(ckpt)
    model = build_model(cfg, grid, stats)
    model.load_state_dict(ckpt["ema"] if cfg.ema_decay > 0 and ckpt.get("ema") is not None else ckpt["model"])
    return cfg, grid, stats, model.to(device).eval()


def save_metrics(path: Path, r: dict, r_idw: Optional[dict]) -> None:
    """One file per evaluation; "neighbours" writes the scores of its baseline next to it (<name>_idw.npz)."""
    np.savez_compressed(path, **r)
    if r_idw is not None:
        np.savez_compressed(path.with_name(path.stem + "_idw.npz"), **r_idw)


def run_eval(args, ddp: bool, rank: int, device) -> None:
    path = Path(args.ckpt) if args.ckpt else Path(apply_overrides(Config(), args.set).ckpt_dir) / "best.pt"
    cfg, grid, stats, model = load_run(path, args.set, device)
    check_config(cfg, grid["depth"])
    years = resolve_splits(cfg, load_grid(cfg)["years"])[args.split]
    ds = make_dataset(cfg, grid, stats, years, False, max_profiles=args.max_profiles, seed=cfg.seed,
                      in_memory=cfg.in_memory, extras=True)
    r, examples, r_idw = evaluate(model, make_loader(ds, cfg, cfg.eval_batch_size, False, ddp), cfg, grid, stats,
                                  Task(cfg, grid, stats, device), device)
    if rank == 0:
        out = Path(cfg.ckpt_dir) / f"{args.split}_metrics{'_' + args.tag if args.tag else ''}.npz"
        save_metrics(out, r, r_idw)
        log.info("%s: %s -> %s", args.split, summary_line(r), out)
        if r_idw is not None:
            log.info("%s, inverse-distance-weighted casts: %s", args.split, summary_line(r_idw))
        run = init_wandb(cfg, None, f"eval-{args.split}", {"checkpoint": str(path), "tag": args.tag})
        wandb_log(r, args.split, 0, True, examples, scalars(r_idw, f"{args.split}_idw") if r_idw is not None else {})
        run.finish()


def main() -> None:
    args = parse_args()
    ddp, rank, world, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = apply_overrides(Config(), args.set)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        run_eval(args, ddp, rank, device)
        if ddp:
            dist.destroy_process_group()
        return
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    ckpt = None
    resume = Path(cfg.ckpt_dir) / "last.pt" if args.resume == "auto" else (None if args.resume == "none" else Path(args.resume))
    if resume is not None and resume.exists():
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        saved = config_from_checkpoint(ckpt)
        for key in FROZEN_ON_RESUME:  # a resumed run keeps the loss / outputs it started with
            if str(getattr(saved, key)) != str(getattr(cfg, key)):
                log.warning("Resuming: keeping %s=%r of the checkpoint (not %r)", key, getattr(saved, key), getattr(cfg, key))
                setattr(cfg, key, getattr(saved, key))
        log.info("Resuming from %s (epoch %d, step %d)", resume, ckpt["state"]["epoch"], ckpt["state"]["step"])

    grid = load_grid(cfg)
    check_config(cfg, grid["depth"])
    splits = resolve_splits(cfg, grid["years"])
    log.info("Splits: %s", {k: f"{v[0]}-{v[-1]}" if v else "-" for k, v in splits.items()})
    if ckpt is not None:
        stats = stats_from_ckpt(ckpt)
    else:
        if rank == 0:
            load_or_compute_stats(cfg, grid, splits["train"])
        if ddp:
            dist.barrier()
        stats = load_or_compute_stats(cfg, grid, splits["train"])
    if cfg.input_drift_mode == "lagged":  # archive basin means, computed once by rank 0
        if rank == 0:
            load_or_compute_basin_monthly(cfg, grid)
        if ddp:
            dist.barrier()
    task = Task(cfg, grid, stats, device)
    log.info("Experiment %s | targets %s | loss weights %s | drift %s on %s", cfg.experiment, task.names,
             None if task.loss_w is None else [round(w, 3) for w in task.loss_w.tolist()], cfg.input_drift_mode,
             cfg.input_drift_vars)

    train_ds = make_dataset(cfg, grid, stats, splits["train"], True, in_memory=cfg.in_memory)
    val_ds = make_dataset(cfg, grid, stats, splits["val"], False, max_profiles=cfg.val_max_profiles, seed=cfg.seed,
                          in_memory=cfg.in_memory, extras=True)
    train_loader = make_loader(train_ds, cfg, cfg.batch_size, True, ddp)
    val_loader = make_loader(val_ds, cfg, cfg.eval_batch_size, False, ddp)
    steps_per_epoch = len(train_loader) if cfg.steps_per_epoch is None else min(len(train_loader), cfg.steps_per_epoch)

    model = build_model(cfg, grid, stats).to(device)
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg, total_steps=cfg.epochs * steps_per_epoch)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_mode(cfg, device) == "fp16")
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp_mode(cfg, device) == "fp16")
    state = {"epoch": 0, "step": 0, "best_val": float("inf"), "wandb_id": None}
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state.update(ckpt["state"])
    ema = WeightEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    if ema is not None and ckpt is not None and ckpt.get("ema") is not None:
        ema.model.load_state_dict(ckpt["ema"])
    del ckpt
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model %.2f M parameters | device %s | precision %s", n_params / 1e6,
             torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", amp_mode(cfg, device))
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if rank == 0:
        state["wandb_id"] = init_wandb(cfg, state["wandb_id"], "train",
                                       {"n_params": n_params, "world_size": world, "splits": splits}).id

    for epoch in range(state["epoch"], cfg.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, opt, sched, scaler, cfg, task, device, state, rank, ema)
        state["epoch"] = epoch + 1
        r, examples, r_idw = evaluate(ema.model if ema is not None else model, val_loader, cfg, grid, stats, task, device)
        is_best = float(r["objective"]) < state["best_val"]
        if is_best:
            state["best_val"] = float(r["objective"])
        if rank == 0:
            log.info("epoch %d | %s%s", epoch + 1, summary_line(r), "  *best*" if is_best else "")
            wandb_log(r, "val", state["step"], False, examples, {"val/best_loss": state["best_val"], "epoch": epoch + 1,
                                                                  **(scalars(r_idw, "val_idw") if r_idw else {})})
            last = Path(cfg.ckpt_dir) / "last.pt"
            save_checkpoint(last, model=unwrap(model).state_dict(), ema=ema.model.state_dict() if ema else None,
                            optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                            state=state, cfg=cfg.to_dict(), grid=grid, splits=splits,
                            stats={k: np.asarray(v).tolist() for k, v in stats.items()})
            if is_best:
                shutil.copyfile(last, Path(cfg.ckpt_dir) / "best.pt")
                save_metrics(Path(cfg.ckpt_dir) / "val_metrics_best.npz", r, r_idw)
        if ddp:
            dist.barrier()
    if rank == 0:
        import wandb

        wandb.finish()
        log.info("Finished. Best validation objective %.4f", state["best_val"])
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
