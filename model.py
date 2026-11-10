"""
FGN profile transformer (Alet et al. 2025, as in WeatherNext 2/3). Every LayerNorm is modulated by a noise vector z
drawn once per member and shared by every level and every variable, so one draw is one coherent profile of all the
predicted variables (O2 + the auxiliary ones, all from the same head). Giving the same z to several profiles makes
the member coherent between them too (train.fgn_members, figures.py field).

  encoder  [context] [observed level] ...        observed levels: surface -> D_input ("cutoff") or anywhere ("sparse"),
                                                 or every level of nearby casts ("neighbours")
  decoder  [query level] ...                     every unobserved valid level
  head     small MLP: O2 and every auxiliary variable per level and member
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


def aux_feature_indices(cfg, grid: dict) -> Tuple[List[str], List[int]]:
    """Auxiliary target names and feature indices: aux_target_vars present in the data, without the target."""
    feats = list(grid["features"])
    names = [v for v in dict.fromkeys(cfg.aux_target_vars or ()) if v != grid["target"] and v in feats]
    return names, [feats.index(v) for v in names]


def gather_levels(x: torch.Tensor, level_idx: torch.Tensor, feature_idx: torch.Tensor) -> torch.Tensor:
    """x (B, Z, F), level_idx (B, T), feature_idx (A,) -> (B, T, A)."""
    xa = x.index_select(-1, feature_idx)
    return torch.gather(xa, 1, level_idx.unsqueeze(-1).expand(-1, -1, xa.shape[-1]))


def pack_levels(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indices of the True levels of every row, shallowest first, padded: idx (B, M), pad (B, M) [True = padding]."""
    nz = mask.shape[1]
    lev = torch.arange(nz, device=mask.device)
    n = mask.sum(dim=1)
    idx = torch.where(mask, lev[None, :], lev[None, :] + nz).argsort(dim=1)[:, :max(1, int(n.max()))]
    return idx, torch.arange(idx.shape[1], device=mask.device)[None, :] >= n[:, None]


def observation_mask(depth: torch.Tensor, valid_len: torch.Tensor, spacing: torch.Tensor, first: torch.Tensor,
                     max_obs_depth: Optional[float] = None) -> torch.Tensor:
    """Sparse experiment: the levels nearest to first, first + spacing, ... down to the bottom (B, Z) bool."""
    bottom = depth[(valid_len - 1).clamp(min=0)].float()
    limit = bottom if max_obs_depth is None else bottom.clamp(max=float(max_obs_depth))
    n_samples = int(torch.ceil((depth[-1] - first.min().clamp(max=float(depth[-1]))) / spacing.min()).item()) + 1
    t = first.float()[:, None] + torch.arange(n_samples, device=depth.device, dtype=torch.float32)[None] * spacing.float()[:, None]
    near = torch.minimum((depth[None, None, :] - t[:, :, None]).abs().argmin(dim=-1), (valid_len - 1).clamp(min=0)[:, None])
    hits = torch.zeros(valid_len.shape[0], depth.numel(), dtype=torch.int32, device=depth.device)
    hits.scatter_add_(1, near, (t <= limit[:, None] + 1e-6).to(torch.int32))
    return hits > 0


class FourierDepthEncoding(nn.Module):
    """Depth (or depth difference) in metres -> sin/cos of log-spaced wavelengths + signed log depth -> MLP."""

    def __init__(self, d_model: int, n_freqs: int = 16, min_scale_m: float = 2.0, max_scale_m: float = 4000.0):
        super().__init__()
        wavelengths = torch.logspace(math.log10(min_scale_m), math.log10(max_scale_m), n_freqs)
        self.register_buffer("omega", 2.0 * math.pi / wavelengths, persistent=False)
        self.log_norm = math.log1p(max_scale_m)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freqs + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        angle = z.unsqueeze(-1) * self.omega
        log_z = (torch.sign(z) * torch.log1p(z.abs()) / self.log_norm).unsqueeze(-1)
        return self.mlp(torch.cat([torch.sin(angle), torch.cos(angle), log_z], dim=-1))


class NoiseConditionedLayerNorm(nn.Module):
    """LayerNorm(h) * (1 + W_s z) + W_b z."""

    def __init__(self, d_model: int, noise_dim: int, init_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.film = nn.Linear(noise_dim, 2 * d_model)
        nn.init.normal_(self.film.weight, std=init_scale / math.sqrt(noise_dim))
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(z).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(h) * (1.0 + scale) + shift


class MemberDropout(nn.Module):
    """Dropout with one mask per profile shared by its members in training (members on consecutive rows);
    off at evaluation, or independent per member with epistemic=True."""

    def __init__(self, p: float, epistemic: bool = False):
        super().__init__()
        self.p, self.epistemic = float(p), bool(epistemic)

    def forward(self, x: torch.Tensor, members: int) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        if self.training:
            keep = torch.rand((x.shape[0] // members,) + tuple(x.shape[1:]), device=x.device) >= self.p
            return x * (keep.to(x.dtype).repeat_interleave(members, dim=0) / (1.0 - self.p))
        return F.dropout(x, self.p, training=True) if self.epistemic else x


class FGNBlock(nn.Module):
    """Pre-LN block: self-attention, cross-attention (decoder), MLP; every norm noise-conditioned."""

    def __init__(self, d_model, n_heads, d_ff, dropout, noise_dim, cross, epistemic):
        super().__init__()
        self.norm_self = NoiseConditionedLayerNorm(d_model, noise_dim)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross = cross
        if cross:
            self.norm_cross = NoiseConditionedLayerNorm(d_model, noise_dim)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = NoiseConditionedLayerNorm(d_model, noise_dim)
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.drop = MemberDropout(dropout, epistemic)

    def forward(self, h, z, pad, members, memory=None, memory_pad=None):
        a = self.norm_self(h, z)
        h = h + self.drop(self.self_attn(a, a, a, key_padding_mask=pad, need_weights=False)[0], members)
        if self.cross:
            a = self.norm_cross(h, z)
            h = h + self.drop(self.cross_attn(a, memory, memory, key_padding_mask=memory_pad, need_weights=False)[0],
                              members)
        a = self.norm_ff(h, z)
        return h + self.drop(self.ff_out(self.drop(F.gelu(self.ff_in(a)), members)), members)


class _FGNBase(nn.Module):
    CONTEXT, OBSERVED, QUERY = 0, 1, 2

    def __init__(self, n_features, n_context, depth_levels, d_model=192, n_heads=8, n_enc_layers=4, n_dec_layers=4,
                 d_ff=768, dropout=0.1, n_depth_freqs=16, depth_min_scale_m=2.0, depth_max_scale_m=4000.0,
                 lower_bound=None, n_aux=0, noise_dim=32, epistemic_dropout=False, aux_lower_bounds=None):
        super().__init__()
        self.register_buffer("depth", torch.as_tensor(np.asarray(depth_levels), dtype=torch.float32))
        self.noise_dim, self.n_aux = int(noise_dim), int(n_aux)

        def mlp(n_in):
            return nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        self._depth_args = (d_model, n_depth_freqs, depth_min_scale_m, depth_max_scale_m)
        self.feature_embed = mlp(n_features)
        self.context_embed = mlp(n_context)
        self.abs_depth = self._enc()
        self.token_type = nn.Embedding(3, d_model)
        block = dict(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, noise_dim=noise_dim,
                     epistemic=epistemic_dropout)
        self.encoder = nn.ModuleList([FGNBlock(cross=False, **block) for _ in range(n_enc_layers)])
        self.encoder_norm = NoiseConditionedLayerNorm(d_model, noise_dim)
        self.decoder = nn.ModuleList([FGNBlock(cross=True, **block) for _ in range(n_dec_layers)])
        self.decoder_norm = NoiseConditionedLayerNorm(d_model, noise_dim)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1 + self.n_aux))
        self.use_lower_bound = lower_bound is not None
        self.register_buffer("lower_bound", torch.tensor(float(lower_bound) if self.use_lower_bound else 0.0))
        bounds = list(aux_lower_bounds or [None] * self.n_aux)  # derived from the config: not saved
        self.register_buffer("aux_bounded", torch.tensor([b is not None for b in bounds], dtype=torch.bool),
                             persistent=False)
        self.register_buffer("aux_lb", torch.tensor([0.0 if b is None else float(b) for b in bounds]), persistent=False)
        self.any_aux_bounded = any(b is not None for b in bounds)
        with torch.no_grad():  # untrained members start at the training mean (lb + softplus(raw) = 0)
            last = self.head[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            for j, b in enumerate([lower_bound] + bounds):
                if b is not None:
                    last.bias[j] = float(_inv_softplus(torch.tensor(max(-float(b), 1e-2))))

    def _enc(self) -> FourierDepthEncoding:
        return FourierDepthEncoding(*self._depth_args)

    def _source(self, tokens, obs_pad, ctx):
        """Context token + observed-level tokens, and their padding mask."""
        ctx_token = (self.context_embed(ctx) + self.token_type.weight[self.CONTEXT]).unsqueeze(1)
        src_pad = torch.cat([torch.zeros(ctx.shape[0], 1, dtype=torch.bool, device=ctx.device), obs_pad], dim=1)
        return torch.cat([ctx_token, tokens], dim=1), src_pad

    def _members(self, src, src_pad, queries, query_pad, n_members, noise):
        """-> O2 members (B, M, T) and auxiliary members (B, M, T, A) or None, fp32."""
        bsz, n_q, m = src.shape[0], queries.shape[1], int(n_members)
        if noise is None:
            noise = torch.randn(bsz, m, self.noise_dim, device=src.device)
        z = noise.reshape(bsz * m, self.noise_dim).float()
        h, h_pad = src.repeat_interleave(m, dim=0), src_pad.repeat_interleave(m, dim=0)
        for blk in self.encoder:
            h = blk(h, z, h_pad, m)
        memory = self.encoder_norm(h, z)
        q, q_pad = queries.repeat_interleave(m, dim=0), query_pad.repeat_interleave(m, dim=0)
        for blk in self.decoder:
            q = blk(q, z, q_pad, m, memory, h_pad)
        out = self.head(self.decoder_norm(q, z)).float().reshape(bsz, m, n_q, 1 + self.n_aux)
        o2 = self.lower_bound + F.softplus(out[..., 0]) if self.use_lower_bound else out[..., 0]
        if not self.n_aux:
            return o2, None
        aux = out[..., 1:]
        if self.any_aux_bounded:
            aux = torch.where(self.aux_bounded, self.aux_lb + F.softplus(aux), aux)
        return o2, aux


class CutoffFGN(_FGNBase):
    """Observed from the surface down to D_input; members for every valid level below it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_depth = self._enc()

    def forward(self, x, obs, valid_len, ctx, n_members=2, noise=None):
        """x (B, Z, F), obs (B, Z) bool = levels [0, k), valid_len (B,), ctx (B, C) -> O2 members (B, M, T),
        target level indices (B, T), target mask (B, T), auxiliary members (B, M, T, A) or None."""
        nz = x.shape[1]
        k = obs.sum(dim=1)
        k_max = int(k.max())
        obs_pad = torch.arange(k_max, device=x.device)[None, :] >= k[:, None]
        x_obs = x[:, :k_max] * (~obs_pad).unsqueeze(-1).to(x.dtype)  # nothing below D_input can leak
        tokens = (self.feature_embed(x_obs) + self.abs_depth(self.depth[:k_max])[None]
                  + self.token_type.weight[self.OBSERVED])
        src, src_pad = self._source(tokens, obs_pad, ctx)
        offsets = torch.arange(int((valid_len - k).max()), device=x.device)
        target_mask = offsets[None, :] < (valid_len - k)[:, None]
        target_idx = (k[:, None] + offsets[None, :]).clamp(max=nz - 1)
        z_query = self.depth[target_idx]
        queries = (self.abs_depth(z_query) + self.rel_depth(z_query - self.depth[(k - 1).clamp(min=0)][:, None])
                   + self.token_type.weight[self.QUERY])
        o2, aux = self._members(src, src_pad, queries, ~target_mask, n_members, noise)
        return o2, target_idx, target_mask, aux


class SparseFGN(_FGNBase):
    """Observed at a set of levels anywhere in the column; members for every other valid level. Each query encodes
    its distance to the nearest observation above and below (a learned embedding when there is none)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_above = self._enc()
        self.dist_below = self._enc()
        self.no_obs = nn.Embedding(2, self.token_type.weight.shape[1])

    def forward(self, x, obs, valid_len, ctx, n_members=2, noise=None):
        lev = torch.arange(x.shape[1], device=x.device)
        valid = lev[None, :] < valid_len[:, None]
        obs = obs & valid
        tgt = valid & ~obs
        obs_idx, obs_pad = pack_levels(obs)
        x_obs = torch.gather(x, 1, obs_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])) * (~obs_pad).unsqueeze(-1).to(x.dtype)
        tokens = self.feature_embed(x_obs) + self.abs_depth(self.depth[obs_idx]) + self.token_type.weight[self.OBSERVED]
        src, src_pad = self._source(tokens, obs_pad, ctx)
        tgt_idx, tgt_pad = pack_levels(tgt)
        z_q = self.depth[tgt_idx]
        dep = self.depth[None, None, :]
        above = obs[:, None, :] & (dep < z_q[:, :, None])
        below = obs[:, None, :] & (dep > z_q[:, :, None])
        ha, hb = above.any(dim=-1), below.any(dim=-1)
        d_above = torch.where(ha, z_q - torch.where(above, dep, torch.full_like(dep, -1e9)).amax(dim=-1), 0.0)
        d_below = torch.where(hb, torch.where(below, dep, torch.full_like(dep, 1e9)).amin(dim=-1) - z_q, 0.0)
        ha, hb = ha.unsqueeze(-1).float(), hb.unsqueeze(-1).float()
        queries = (self.abs_depth(z_q) + self.dist_above(d_above) * ha + self.dist_below(d_below) * hb
                   + self.no_obs.weight[0] * (1.0 - ha) + self.no_obs.weight[1] * (1.0 - hb)
                   + self.token_type.weight[self.QUERY])
        attn_pad = tgt_pad.clone()  # a row without targets keeps one dummy query (all-masked attention gives NaN)
        attn_pad[:, 0] &= tgt.any(dim=1)
        o2, aux = self._members(src, src_pad, queries, attn_pad, n_members, noise)
        return o2, tgt_idx, ~tgt_pad, aux


class NeighbourFGN(_FGNBase):
    """Nearby casts of the same time step; members for every valid level of the profile at the target position.
    Tokens: [target context] [cast context + offset] [cast level + offset]; offset = learned embedding of the
    position of the cast relative to the target (km east, km north, distance)."""

    OFFSET_SCALE_KM = 50.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.token_type.weight.shape[1]
        self.offset_embed = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, src_x, src_len, src_ctx, src_dxy, valid_len, ctx, n_members=2, noise=None):
        """src_x (B, K, Z, F), src_len (B, K) [0 = no cast], src_ctx (B, K, C), src_dxy (B, K, 2) km, valid_len (B,),
        ctx (B, C) of the target -> as CutoffFGN."""
        bsz, k = src_len.shape
        zs, zt = max(1, int(src_len.max())), int(valid_len.max())
        lev = torch.arange(max(zs, zt), device=src_x.device)
        dxy = src_dxy.float() / self.OFFSET_SCALE_KM
        off = self.offset_embed(torch.cat([dxy, dxy.norm(dim=-1, keepdim=True)], dim=-1))  # (B, K, d)
        casts = self.context_embed(src_ctx) + off + self.token_type.weight[self.CONTEXT]
        levels = (self.feature_embed(src_x[:, :, :zs]) + self.abs_depth(self.depth[:zs])[None, None] + off[:, :, None]
                  + self.token_type.weight[self.OBSERVED]).reshape(bsz, k * zs, -1)
        pad = torch.cat([src_len == 0, (lev[None, None, :zs] >= src_len[..., None]).reshape(bsz, k * zs)], dim=1)
        src, src_pad = self._source(torch.cat([casts, levels], dim=1), pad, ctx)
        target_mask = lev[None, :zt] < valid_len[:, None]
        queries = (self.abs_depth(self.depth[:zt]) + self.token_type.weight[self.QUERY])[None].expand(bsz, -1, -1)
        o2, aux = self._members(src, src_pad, queries, ~target_mask, n_members, noise)
        return o2, lev[None, :zt].expand(bsz, -1), target_mask, aux


def aux_lower_bounds(cfg, grid: dict, stats) -> List[Optional[float]]:
    """Physical 0 in standardised units for the bounded auxiliary variables (nonnegative_vars, not drift-shifted,
    not log-transformed), else None."""
    names, feat = aux_feature_indices(cfg, grid)
    if not cfg.enforce_nonnegative:
        return [None] * len(names)
    shifted = set(cfg.input_drift_vars or ()) if cfg.input_drift_mode != "none" else set()
    skip = shifted | set(grid.get("log_transform_vars", ()))
    return [-float(stats["mean"][f]) / float(stats["std"][f]) if (n in cfg.nonnegative_vars and n not in skip) else None
            for n, f in zip(names, feat)]


def build_model(cfg, grid: dict, stats) -> _FGNBase:
    ti = grid["target_index"]
    lower = -float(stats["mean"][ti]) / float(stats["std"][ti]) if cfg.enforce_nonnegative else None
    cls = {"sparse": SparseFGN, "neighbours": NeighbourFGN}.get(cfg.experiment, CutoffFGN)
    return cls(n_features=len(grid["features"]), n_context=len(grid["context"]), depth_levels=grid["depth"],
               d_model=cfg.d_model, n_heads=cfg.n_heads, n_enc_layers=cfg.n_enc_layers, n_dec_layers=cfg.n_dec_layers,
               d_ff=cfg.d_ff, dropout=cfg.dropout, n_depth_freqs=cfg.n_depth_freqs,
               depth_min_scale_m=cfg.depth_min_scale_m, depth_max_scale_m=cfg.depth_max_scale_m, lower_bound=lower,
               n_aux=len(aux_feature_indices(cfg, grid)[0]), noise_dim=cfg.fgn_noise_dim,
               epistemic_dropout=cfg.fgn_epistemic_dropout, aux_lower_bounds=aux_lower_bounds(cfg, grid, stats))
