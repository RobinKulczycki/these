#!/usr/bin/env python
"""
Figures from the metrics files (<run>/val_metrics_best.npz, <run>/<split>_metrics.npz) and the processed data.

  data      O2 map + profiles of one day, the same profiles against depth and density, spread left per coordinate
  compare   1-3 runs, every variable: CRPS / bias / coverage / spread-skill against depth and density, heat maps per
            row (D_input, spacing or distance to the nearest cast), difference of two runs, per-row profiles,
            oxycline, summary.csv
            time:        bias and CRPS per month of the record, per calendar month (also per water-mass layer) and
                         per calendar month x depth
            calibration: rank histogram
            coherence:   what the coherence of the members is worth, between levels and between variables
  maps      error maps of every variable per water-mass layer and per season, the oxycline, and the difference with
            another run
  ensemble  members of every variable for a few casts of one run (the same member keeps its colour in every
            panel) + ensemble_coherence.csv: density inversions, distance from the equation of state
  field     "neighbours" only: a whole map predicted from a few stations, with members coherent between profiles
  drift     data only: how the column drifts between years and the bias each drift reference leaves
  all       compare + maps (+ difference) + ensemble (+ field) for the given runs, and data + drift with
            --processed-dir

    python figures.py all --out figs --processed-dir $D "new=$R/v4_cutoff" "old=$R/v3_fgn_lag"
A run may be given as its folder (its <split>_metrics.npz, else val_metrics_best.npz) or as a metrics file.
"neighbours" runs: the inverse-distance-weighted mean of the same casts (<metrics file>_idw.npz) is added as a run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import LONG_NAMES, SHORT_NAMES, UNITS, coherence, layer_labels, pooled  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
VAR_COLORS = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f5ad6", "#d6457a", "#7a6a4f", "#4fa3a8"]
INK, INK2, MUTED, GRID, SURFACE, NODATA = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#d9d8d3"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#9ec5f4", "#ecebe7", "#f6b89c", "#eb6834"])
MEMBER_COLORS = ["#eb6834", "#1baf7a", "#8f5ad6"]  # the same member in every panel of the ensemble figure
LABELS = {"crps": "CRPS", "mae": "MAE of the median", "rmse": "RMSE", "bias": "bias (median - truth)",
          "coverage": "coverage of the member range", "spread_skill": "spread / skill", "crps_rel": "CRPS / std"}
MONTHS = "JFMAMJJASOND"
SEASONS = ("winter (DJF)", "spring (MAM)", "summer (JJA)", "autumn (SON)")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.8})
Runs = Sequence[Tuple[str, Dict[str, np.ndarray]]]


# --------------------------------------------------------------------------------------- helpers
def metrics_file(spec: str, split: str = "val") -> Path:
    """'LABEL=path' or 'path' -> the metrics file: the path itself, or of a run folder its <split>_metrics.npz
    (else val_metrics_best.npz, written during training)."""
    p = Path(spec.partition("=")[2] or spec)
    if p.is_dir():
        p = next((q for q in (p / f"{split}_metrics.npz", p / "val_metrics_best.npz") if q.exists()), p / f"{split}_metrics.npz")
    return p


def load(spec: str, split: str = "val") -> Tuple[str, Dict[str, np.ndarray]]:
    label, _, path = spec.partition("=")
    p = metrics_file(spec, split)
    if not p.exists():
        raise SystemExit(f"{p} not found (a run of older code needs: EVAL=only bash submit.sh NAME)")
    with np.load(p) as f:
        m = {k: f[k] for k in f.files}
    if "v_crps" not in m or "rows" not in m:
        raise SystemExit(f"{p}: not a metrics file of this code version (re-run train.py --eval-only)")
    return label if path else p.parent.name, m


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def names(m) -> List[str]:
    return [str(v) for v in m["variables"]]


def unit(m, v: int, metric: str = "crps") -> str:
    u = str(m["units"][v])
    return u if metric in ("crps", "mae", "rmse", "bias") and u != "-" else ""


def vertical(m, coord: str):
    """(positions, cell edges, array prefix, axis label) of the vertical coordinate."""
    if coord == "depth":
        z = np.asarray(m["level_depths"], dtype=np.float64)
        return z, np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1]), [z[-1] + (z[-1] - z[-2]) / 2]]), "v_", "depth (m)"
    e = np.asarray(m["density_edges"], dtype=np.float64)
    return 0.5 * (e[1:] + e[:-1]), e, "d_", "$\\sigma_0$ of the predicted level (kg m$^{-3}$)"


def coords(m) -> List[str]:
    return ["depth"] + (["density"] if "density_edges" in m else [])


def is_ensemble(m) -> bool:
    """False for the one-member baseline (IDW), which has no coverage, spread, rank or coherence."""
    return int(m["n_members"]) > 1


def pooled_rows(m, prefix: str, metric: str, v: int, rows=None):
    """Metric of variable v against the vertical coordinate, pooled over rows -> (values, count)."""
    sel = slice(None) if rows is None else list(rows)
    w = m[prefix + "count"][v][sel].astype(np.float64)
    if metric == "crps_rel":
        return pooled(m[prefix + "crps"][v][sel], w, 0) / float(m["var_std"][v]), w.sum(0)
    if metric == "spread_skill":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(pooled(m[prefix + "spread"][v][sel] ** 2, w, 0) /
                           pooled(m[prefix + "rmse_ensmean"][v][sel] ** 2, w, 0)), w.sum(0)
    return pooled(m[prefix + metric][v][sel], w, 0), w.sum(0)


def pooled_vertical(m, metric: str, v: int) -> np.ndarray:
    """Metric of variable v per row, pooled over levels."""
    w = m["v_count"][v].astype(np.float64)
    if metric == "crps_rel":
        return pooled(m["v_crps"][v], w, 1) / float(m["var_std"][v])
    if metric == "spread_skill":
        return np.sqrt(pooled(m["v_spread"][v] ** 2, w, 1) / pooled(m["v_rmse_ensmean"][v] ** 2, w, 1))
    return pooled(m[f"v_{metric}"][v], w, 1)


def row_ticks(m) -> List[str]:
    """Row labels: the value, or for "neighbours" the distance bin ('0-10', '10-20', ...)."""
    rows = [float(r) for r in m["rows"]]
    if "nearest cast" in str(m["row_name"]):
        return [f"{a:g}-{b:g}" for a, b in zip([0.0] + rows[:-1], rows)]
    return [f"{r:g}" for r in rows]


def row_text(m, i: int) -> str:
    """'D_input 50 m', 'spacing 20 m', 'nearest cast 10-20 km' for row i."""
    name, _, unit = str(m["row_name"]).partition(" (")
    name = name.replace("distance to the nearest cast", "nearest cast")
    return f"{name} {row_ticks(m)[i]} {unit.rstrip(')')}".strip()


def grid_of(n: int, ncol: Optional[int] = None):
    ncol = ncol or (n if n <= 4 else 3)
    return int(np.ceil(n / ncol)), ncol


def subplots(n: int, w: float, h: float, ncol=None, **kw):
    nrow, ncol = grid_of(n, ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol + 0.6, h * nrow + 0.6), squeeze=False, **kw)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def vlim(m, coord: str, max_depth: float):
    if coord == "depth":
        return 0.0, float(min(max_depth, m["level_depths"][-1]))
    _, edges, _, _ = vertical(m, coord)
    ok = np.nonzero(m["d_count"].sum(axis=(0, 1)) > 0)[0]
    return (float(edges[ok[0]]), float(edges[ok[-1] + 1])) if ok.size else (float(edges[0]), float(edges[-1]))


def finish(fig, axes, title: str, legend: bool) -> None:
    if legend:
        h, lab = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=max(1, len(lab)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03 if legend else 0, 1, 1))


def ref_line(ax, metric: str, m, vertical_axis: bool = True) -> None:
    nominal = (int(m["n_members"]) - 1) / (int(m["n_members"]) + 1)
    ref = {"coverage": nominal, "spread_skill": 1.0, "bias": 0.0}.get(metric)
    if ref is not None:
        (ax.axvline if vertical_axis else ax.axhline)(ref, color=INK2, lw=1, ls="--")


def norm_for(metric: str, data: np.ndarray, m):
    finite = data[np.isfinite(data)]
    if metric == "bias" or metric == "change":
        lim = max(float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0, 1e-9)
        return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), DIV, np.clip(data, -lim, lim)
    if metric == "spread_skill":
        return TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0), DIV.reversed(), np.clip(data, 0, 2)
    if metric == "coverage":
        nominal = (int(m["n_members"]) - 1) / (int(m["n_members"]) + 1)
        return TwoSlopeNorm(vmin=min(nominal - 0.05, float(finite.min()) if finite.size else 0.5), vcenter=nominal,
                            vmax=1.0), DIV.reversed(), data
    lo = float(np.percentile(finite, 2)) if finite.size else 0.0
    hi = float(np.percentile(finite, 98)) if finite.size else 1.0
    return plt.Normalize(min(lo, 0.0) if metric in ("crps", "mae") else lo, max(hi, lo + 1e-9)), SEQ, data


# --------------------------------------------------------------------------------------- per-variable figures
def fig_profiles(runs: Runs, coord: str, metric: str, rows=None, max_depth: float = 300.0):
    """One panel per variable: metric against depth or density (pooled over rows), one line per run."""
    m0 = runs[0][1]
    fig, axes = subplots(len(names(m0)), 3.4, 3.7, sharey=True)
    top, bottom = vlim(m0, coord, max_depth)
    for v, name in enumerate(names(m0)):
        ax = axes.flat[v]
        for (label, m), color in zip(runs, SERIES):
            if name in names(m) and (is_ensemble(m) or metric not in ("coverage", "spread_skill")):
                y, _, prefix, _ = vertical(m, coord)
                val, n = pooled_rows(m, prefix, metric, names(m).index(name), rows)
                ok = (n > 0) & np.isfinite(val)
                ax.plot(val[ok], y[ok], color=color, label=label)
        ref_line(ax, metric, m0)
        ax.set_title(LONG_NAMES.get(name, name))
        u = unit(m0, v, metric)
        ax.set_xlabel(LABELS.get(metric, metric) + (f" ({u})" if u else ""), fontsize=9)
    axes[0, 0].set_ylim(bottom, top)
    for row in axes:
        row[0].set_ylabel(vertical(m0, coord)[3], fontsize=9)
    finish(fig, axes, f"{LABELS.get(metric, metric)} against {coord}, every variable", len(runs) > 1)
    return fig


def fig_relative(m, coord: str, max_depth: float = 300.0):
    """Every variable on one axis: CRPS / training std (the scale of the equal-weight loss)."""
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    y, _, prefix, label = vertical(m, coord)
    for v, name in enumerate(names(m)):
        val, n = pooled_rows(m, prefix, "crps_rel", v)
        ok = (n > 0) & np.isfinite(val)
        ax.plot(val[ok], y[ok], color=VAR_COLORS[v % len(VAR_COLORS)], lw=2.4 if v == 0 else 1.5,
                label=LONG_NAMES.get(name, name))
    top, bottom = vlim(m, coord, max_depth)
    ax.set_ylim(bottom, top)
    ax.set_xlim(left=0)
    ax.set(xlabel="CRPS / training standard deviation", ylabel=label, title="Relative error of every variable")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def fig_by_row(runs: Runs, metric: str = "crps"):
    """One panel per variable: metric against D_input / spacing."""
    m0 = runs[0][1]
    rows = np.asarray(m0["rows"])
    fig, axes = subplots(len(names(m0)), 3.5, 2.9, sharex=True)
    for v, name in enumerate(names(m0)):
        ax = axes.flat[v]
        for (label, m), color in zip(runs, SERIES):
            if name in names(m):
                ax.plot(np.arange(len(rows)), pooled_vertical(m, metric, names(m).index(name)), color=color, marker="o",
                        ms=4, label=label)
        u = unit(m0, v, metric)
        ax.set_title(LONG_NAMES.get(name, name) + (f" ({u})" if u else ""), fontsize=11)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(row_ticks(m0), fontsize=8)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel(str(m0["row_name"]))
    finish(fig, axes, f"{LABELS.get(metric, metric)} against {str(m0['row_name']).split(' (')[0]}", len(runs) > 1)
    return fig


def fig_heatmaps(m, coord: str, metric: str = "crps", other=None, max_depth: float = 300.0):
    """One heat map per variable: rows x depth or density. other: CRPS of m minus CRPS of other (blue = m better)."""
    rows = np.asarray(m["rows"])
    _, edges, prefix, ylabel = vertical(m, coord)
    top, bottom = vlim(m, coord, max_depth)
    fig, axes = subplots(len(names(m)), 3.8, 3.8, sharey=True)
    for v, name in enumerate(names(m)):
        ax = axes.flat[v]
        data = m[prefix + metric][v].astype(np.float64)
        if other is not None:
            data = data - other[prefix + metric][names(other).index(name)]
        data = np.where(m[prefix + "count"][v] > 0, data, np.nan).T
        inside = (edges[1:] > top) & (edges[:-1] < bottom)
        norm, cmap, _ = norm_for("change" if other is not None else metric, data[inside], m)
        ax.set_facecolor("white")
        mesh = ax.pcolormesh(np.arange(len(rows) + 1) - 0.5, edges, np.ma.masked_invalid(data), cmap=cmap, norm=norm)
        ax.set_ylim(bottom, top)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(row_ticks(m), fontsize=8)
        ax.grid(False)
        ax.set_title(LONG_NAMES.get(name, name), fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02, label=unit(m, v, metric))
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(str(m["row_name"]))
    what = "CRPS difference (blue = first run better)" if other is not None else LABELS.get(metric, metric)
    finish(fig, axes, f"{what} per {str(m['row_name']).split(' (')[0]} and {coord} (white = nothing predicted)", False)
    return fig


def fig_rows(runs: Runs, v: int, coord: str, metric: str = "crps", rows_sel=None, max_depth: float = 300.0):
    """One panel per row (D_input or spacing) for one variable, one line per run."""
    m0 = runs[0][1]
    rows = np.asarray(m0["rows"])
    picks = list(range(len(rows))) if rows_sel is None else rows_sel
    fig, axes = subplots(len(picks), 3.0, 5.0, ncol=min(len(picks), 5), sharey=True)
    top, bottom = vlim(m0, coord, max_depth)
    for ax, r in zip(axes.flat, picks):
        for (label, m), color in zip(runs, SERIES):
            y, _, prefix, _ = vertical(m, coord)
            val = np.where(m[prefix + "count"][v][r] > 0, m[prefix + metric][v][r], np.nan)
            if coord == "depth" and "observed" in m:  # sparse: observed levels are known exactly
                val = np.where(m["observed"][r], 0.0, val)
            ax.plot(val, y, color=color, label=label)
        if coord == "depth" and "observed" in m0:
            ax.plot(np.zeros(int(m0["observed"][r].sum())), m0["level_depths"][m0["observed"][r]], "o", color=INK, ms=3)
        elif coord == "depth" and str(m0["experiment"]) == "cutoff":
            ax.axhspan(0, rows[r], color=GRID, lw=0)
        ax.set_title(row_text(m0, r), fontsize=10)
        ax.set_xlim(left=0)
    axes[0, 0].set_ylim(bottom, top)
    axes[0, 0].set_ylabel(vertical(m0, coord)[3], fontsize=9)
    name = names(m0)[v]
    finish(fig, axes, f"{LONG_NAMES.get(name, name)}: {LABELS.get(metric, metric)} ({unit(m0, v, metric)}) per "
                      f"{str(m0['row_name']).split(' (')[0]}", len(runs) > 1)
    return fig


def fig_oxycline(runs: Runs):
    """Oxycline CRPS as a depth (m) and as a density (kg m-3) against D_input / spacing."""
    m0 = runs[0][1]
    rows = np.asarray(m0["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
    for ax, key, lab in zip(axes[0], ("oxy_crps", "oxy_crps_sigma"), ("CRPS (m)", "CRPS (kg m$^{-3}$)")):
        for (label, m), color in zip(runs, SERIES):
            ax.plot(np.arange(len(rows)), m[key], color=color, marker="o", label=label)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(row_ticks(m0))
        ax.set(xlabel=str(m0["row_name"]), ylabel=lab)
        ax.set_ylim(bottom=0)
    axes[0, 0].set_title("as a depth")
    axes[0, 1].set_title("as a density")
    finish(fig, axes, "Oxycline (O$_2$ < threshold) from whole member profiles", len(runs) > 1)
    return fig


def fig_time(runs: Runs, metric: str = "bias", layers=None):
    """One panel per variable: monthly metric through the evaluation years, one line per run."""
    m0 = runs[0][1]
    fig, axes = subplots(len(names(m0)), 4.3, 2.7, sharex=True)
    lsel = slice(None) if layers is None else list(layers)
    for v, name in enumerate(names(m0)):
        ax = axes.flat[v]
        for (label, m), color in zip(runs, SERIES):
            vi = names(m).index(name)
            n = m["t_count"][:, lsel].sum(axis=(0, 1)).astype(np.float64)
            s = m["t_pinball" if metric == "crps" else "t_bias"][vi][:, lsel].sum(axis=(0, 1)).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(n > 0, s / n * float(m["var_std"][vi]) * (2.0 if metric == "crps" else 1.0), np.nan)
            t = 1970 + m["t_months"] / 12.0 + 1 / 24.0
            ax.plot(t, val, color=color, marker="o", ms=2.5, lw=1.4, label=label)
        if metric == "bias":
            ax.axhline(0, color=INK2, lw=0.8)
        y0, y1 = int(np.floor(t.min())), int(np.ceil(t.max()))
        for y in range(y0 + 1, y1):
            ax.axvline(y, color=NODATA, lw=0.8, zorder=0)
        ax.set_xticks([y + 0.5 for y in range(y0, y1)])
        ax.set_xticklabels([str(y) for y in range(y0, y1)], fontsize=8)
        ax.grid(axis="x", visible=False)
        u = unit(m0, v, metric)
        ax.set_title(LONG_NAMES.get(name, name) + (f" ({u})" if u else ""), fontsize=11)
    lay = "" if layers is None else ", $\\sigma_0$ " + ", ".join(layer_labels(m0["layer_edges"])[i] for i in layers)
    finish(fig, axes, f"{LABELS.get(metric, metric)} per month{lay}", len(runs) > 1)
    return fig


def calendar_months(m, v: int, metric: str, layers=None) -> np.ndarray:
    """CRPS or bias of variable v per calendar month (12,), pooled over the rows, the years and the layers."""
    lsel = slice(None) if layers is None else list(layers)
    cal = (m["t_months"] % 12).astype(int)
    n = np.bincount(cal, weights=m["t_count"][:, lsel].sum(axis=(0, 1)), minlength=12)
    s = np.bincount(cal, weights=m["t_pinball" if metric == "crps" else "t_bias"][v][:, lsel].sum(axis=(0, 1)), minlength=12)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n * float(m["var_std"][v]) * (2.0 if metric == "crps" else 1.0), np.nan)


def fig_season(runs: Runs, metric: str = "crps", by_layer: bool = False):
    """One panel per variable: metric per calendar month. One line per run, or (by_layer) per water-mass layer of
    the first run."""
    m0 = runs[0][1]
    fig, axes = subplots(len(names(m0)), 3.6, 2.6, sharex=True)
    for v, name in enumerate(names(m0)):
        ax = axes.flat[v]
        if by_layer:
            ax.plot(np.arange(12), calendar_months(m0, v, metric), color=INK, marker="o", ms=3, label="whole column")
            for i, lab in enumerate(layer_labels(m0["layer_edges"])):
                ax.plot(np.arange(12), calendar_months(m0, v, metric, [i]), color=SERIES[i % len(SERIES)], marker="o", ms=3,
                        lw=1.4, label=f"$\\sigma_0$ {lab}")
        else:
            for (label, m), color in zip(runs, SERIES):
                if name in names(m):
                    ax.plot(np.arange(12), calendar_months(m, names(m).index(name), metric), color=color, marker="o", ms=3,
                            label=label)
        if metric == "bias":
            ax.axhline(0, color=INK2, lw=0.8)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(list(MONTHS), fontsize=8)
        u = unit(m0, v, metric)
        ax.set_title(LONG_NAMES.get(name, name) + (f" ({u})" if u else ""), fontsize=11)
    finish(fig, axes, f"{LABELS.get(metric, metric)} per calendar month" + (", per water-mass layer" if by_layer else ""),
           by_layer or len(runs) > 1)
    return fig


def fig_season_depth(m, metric: str = "crps", max_depth: float = 300.0):
    """One heat map per variable: calendar month x depth (pooled over the rows and the years)."""
    _, edges, _, ylabel = vertical(m, "depth")
    top, bottom = vlim(m, "depth", max_depth)
    inside = (edges[1:] > top) & (edges[:-1] < bottom)
    n = m["s_count"].astype(np.float64)
    fig, axes = subplots(len(names(m)), 3.8, 3.4, sharey=True)
    for v, name in enumerate(names(m)):
        ax = axes.flat[v]
        with np.errstate(invalid="ignore", divide="ignore"):
            data = (m["s_pinball" if metric == "crps" else "s_bias"][v] / n * float(m["var_std"][v])
                    * (2.0 if metric == "crps" else 1.0)).T  # (level, month)
        norm, cmap, _ = norm_for(metric, data[inside], m)
        ax.set_facecolor("white")
        mesh = ax.pcolormesh(np.arange(13) - 0.5, edges, np.ma.masked_invalid(data), cmap=cmap, norm=norm)
        ax.set_ylim(bottom, top)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(list(MONTHS), fontsize=8)
        ax.grid(False)
        ax.set_title(LONG_NAMES.get(name, name), fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02, label=unit(m, v, metric))
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=9)
    finish(fig, axes, f"{LABELS.get(metric, metric)} per calendar month and depth (white = nothing predicted)", False)
    return fig


def fig_rank(runs: Runs):
    """Rank histogram, one panel per variable: where the truth falls among the sorted members. Flat (dashed) =
    calibrated; U-shape = members too close together; dome = too far apart; slope = bias."""
    runs = [(label, m) for label, m in runs if "rank" in m and is_ensemble(m)]
    m0 = runs[0][1]
    fig, axes = subplots(len(names(m0)), 3.4, 2.6, sharex=True)
    for v, name in enumerate(names(m0)):
        ax = axes.flat[v]
        for (label, m), color in zip(runs, SERIES):
            if name in names(m):
                h = m["rank"][names(m).index(name)].sum(axis=0)
                freq = h / max(h.sum(), 1.0) * len(h)  # 1 = as often as a calibrated ensemble
                ax.stairs(freq, np.linspace(0, 1, len(h) + 1), color=color, lw=1.6, label=label, fill=len(runs) == 1,
                          alpha=0.55 if len(runs) == 1 else 1.0)
        ax.axhline(1.0, color=INK2, lw=1, ls="--")
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["below all", "middle", "above all"], fontsize=8)
        ax.set_title(LONG_NAMES.get(name, name), fontsize=11)
    for row in axes:
        row[0].set_ylabel("frequency / calibrated", fontsize=9)
    finish(fig, axes, "Rank of the truth among the members (dashed = calibrated)", len(runs) > 1)
    return fig


def fig_coherence(runs: Runs):
    """Coherence of the members (metrics.coherence_sums): how much better they predict the difference between two
    consecutive levels, between two neighbouring predicted columns ("neighbours" only) and between two variables
    (right, first run) than the same members shuffled.
    0 = no better than independent draws, 1 = perfect, below 0 = wrong coherence."""
    runs = [(label, m) for label, m in runs if "coh_levels" in m and is_ensemble(m)]
    m0 = runs[0][1]
    nv = len(names(m0))
    bars = [("coh_levels", "Between consecutive levels")]
    bars += [("coh_columns", "Between neighbouring columns")] if "coh_columns" in m0 else []
    fig, axes = plt.subplots(1, len(bars) + 1, figsize=(6.6 * len(bars) + 6.6, 5.2), squeeze=False)
    width = 0.8 / len(runs)
    for ax, (key, title) in zip(axes[0], bars):
        for i, ((label, m), color) in enumerate(zip(runs, SERIES)):
            if key in m:
                val = coherence(m[key])
                shown = [n for n in names(m) if n in names(m0)]
                xs = [names(m0).index(n) + (i - (len(runs) - 1) / 2) * width for n in shown]
                ax.bar(xs, [val[names(m).index(n)] for n in shown], width * 0.92, color=color, label=label)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(np.arange(nv))
        ax.set_xticklabels([SHORT_NAMES.get(n, n) for n in names(m0)])
        ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.0)
        ax.set(ylabel="coherence (1 - CRPS as predicted / CRPS shuffled)", title=title)
        if len(runs) > 1:
            ax.legend(fontsize=9)
    ax2 = axes[0, -1]
    mat = coherence(m0["coh_variables"]) if nv > 1 else np.full((1, 1), np.nan)
    ax2.set_facecolor(NODATA)
    mesh = ax2.pcolormesh(np.arange(nv + 1) - 0.5, np.arange(nv + 1) - 0.5, np.ma.masked_invalid(mat), cmap=DIV.reversed(),
                          norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
    for i in range(nv):
        for j in range(nv):
            if np.isfinite(mat[i, j]):
                ax2.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax2.set_xticks(np.arange(nv))
    ax2.set_yticks(np.arange(nv))
    ax2.set_xticklabels([SHORT_NAMES.get(n, n) for n in names(m0)], fontsize=9)
    ax2.set_yticklabels([SHORT_NAMES.get(n, n) for n in names(m0)], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    ax2.grid(False)
    ax2.set_title("Between two variables" + (f" ({runs[0][0]})" if len(runs) > 1 else ""))
    fig.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.03)
    fig.suptitle("Coherence of the members: 0 = no better than independent draws, 1 = perfect", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def summary_rows(label: str, m) -> List[list]:
    labs = layer_labels(m["layer_edges"])
    out = []
    for v, name in enumerate(names(m)):
        w = m["v_count"][v].astype(np.float64)
        row = [label, name, unit(m, v).replace("$", "")]
        row += [float(pooled(m[f"v_{k}"][v], w)) for k in ("crps", "mae", "bias", "coverage")]
        row.append(row[3] / float(m["var_std"][v]))
        row.append(float(np.sqrt(pooled(m["v_spread"][v] ** 2, w) / pooled(m["v_rmse_ensmean"][v] ** 2, w))))
        row.append(float(coherence(m["coh_levels"])[v]) if "coh_levels" in m else float("nan"))
        row.append(float(coherence(m["coh_columns"])[v]) if "coh_columns" in m else float("nan"))
        if not is_ensemble(m):  # a single estimate has no coverage, spread or coherence
            row[6], row[8], row[9] = float("nan"), float("nan"), float("nan")
        if "density_edges" in m and len(labs) > 1:
            e = m["density_edges"]
            lay = np.searchsorted(m["layer_edges"], 0.5 * (e[1:] + e[:-1]), side="right")
            c = m["d_count"][v].astype(np.float64)
            for i in range(len(labs)):
                sel = (lay == i)[None, :] & (c > 0)
                row.append(float((m["d_crps"][v][sel] * c[sel]).sum() / c[sel].sum()) if sel.any() else float("nan"))
        out.append(row)
    return out


def cmd_compare(runs: Runs, out: Path, max_depth: float = 300.0) -> None:
    m0 = runs[0][1]
    for coord in coords(m0):
        for metric in ("crps", "bias", "coverage", "spread_skill"):
            save(fig_profiles(runs, coord, metric, max_depth=max_depth), out, f"{metric}_{coord}")
        save(fig_relative(m0, coord, max_depth), out, f"relative_crps_{coord}")
        save(fig_heatmaps(m0, coord, max_depth=max_depth), out, f"heatmap_crps_{coord}")
        if len(runs) > 1:
            save(fig_heatmaps(m0, coord, other=runs[1][1], max_depth=max_depth), out, f"heatmap_change_{coord}")
        save(fig_rows(runs, 0, coord, max_depth=max_depth), out, f"o2_per_row_{coord}")
    save(fig_by_row(runs, "crps_rel"), out, "crps_by_row")
    save(fig_oxycline(runs), out, "oxycline_by_row")
    for metric in ("bias", "crps"):
        save(fig_time(runs, metric), out, f"time_{metric}")
        for i, lab in enumerate(layer_labels(m0["layer_edges"]) if len(m0["layer_edges"]) else []):
            save(fig_time(runs, metric, [i]), out, f"time_{metric}_sigma_{lab.replace('<', 'lt').replace('>', 'gt')}")
        save(fig_season(runs, metric), out, f"season_{metric}")
        if len(m0["layer_edges"]):
            save(fig_season(runs, metric, by_layer=True), out, f"season_{metric}_layers")
        if "s_count" in m0:  # metrics files written before the seasonal scores existed have none
            save(fig_season_depth(m0, metric, max_depth), out, f"season_depth_{metric}")
    if "rank" in m0:
        save(fig_rank(runs), out, "rank_histogram")
        save(fig_coherence(runs), out, "coherence")
    labs = layer_labels(m0["layer_edges"]) if "density_edges" in m0 and len(m0["layer_edges"]) else []
    with open(out / "summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["run", "variable", "unit", "CRPS", "MAE", "bias", "coverage", "CRPS/std", "spread/skill",
                     "coherence between levels", "coherence between columns"] + [f"CRPS sigma0 {lab}" for lab in labs])
        for label, m in runs:
            wr.writerows([r[:3] + [f"{x:.4g}" for x in r[3:]] for r in summary_rows(label, m)])
    print((out / "summary.csv").read_text())


# --------------------------------------------------------------------------------------- maps
def map_field(m, v: int, metric: str, rows=None, layers=None, min_count: float = 1.0) -> np.ndarray:
    """Per-cell value pooled over rows / layers: crps, mae, bias, coverage, spread_skill, count of variable v;
    oxycline_crps, oxycline_mae, oxycline_coverage, oxycline_count (m). NaN where count < min_count."""
    rsel = slice(None) if rows is None else list(rows)
    if metric.startswith("oxycline_"):
        key = {"oxycline_crps": "crps", "oxycline_mae": "abs_err", "oxycline_coverage": "covered",
               "oxycline_count": "count"}[metric]
        n = m["m_oxy_count"][rsel].sum(axis=0).astype(np.float64)
        s = n if key == "count" else m[f"m_oxy_{key}"][rsel].sum(axis=0).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= min_count, s if key == "count" else s / n, np.nan)
    lsel = slice(None) if layers is None else list(layers)
    n = m["m_count"][rsel][:, lsel].sum(axis=(0, 1)).astype(np.float64)

    def total(key):
        return m[f"m_{key}"][v][rsel][:, lsel].sum(axis=(0, 1)).astype(np.float64)

    std = float(m["var_std"][v])
    nm = int(m["n_members"])
    with np.errstate(invalid="ignore", divide="ignore"):
        val = {"crps": lambda: 2.0 * total("pinball") / n * std, "mae": lambda: total("abs_err") / n * std,
               "bias": lambda: total("bias") / n * std, "coverage": lambda: total("covered") / n, "count": lambda: n,
               "spread_skill": lambda: np.sqrt((nm + 1.0) / nm * total("ens_var") / total("sq_err_mean"))}[metric]()
    return np.where(n >= min_count, val, np.nan)


def draw_map(ax, m, field: np.ndarray, metric: str, title: str, label: str = "", fig=None, scale=None):
    """scale: the values that set the colour scale (default: the field itself), to share it between maps."""
    n = m["m_count"].sum(axis=(0, 1))
    ys, xs = np.nonzero(n > 0)
    sy = slice(max(ys.min() - 1, 0), ys.max() + 2) if ys.size else slice(None)
    sx = slice(max(xs.min() - 1, 0), xs.max() + 2) if xs.size else slice(None)
    lat, lon, data = m["map_lat"][sy], m["map_lon"][sx], field[sy, sx]
    if scale is None:
        norm, cmap, data = norm_for(metric, data, m)
    else:
        norm, cmap, _ = norm_for(metric, scale, m)

    def edges(c):
        step = float(c[1] - c[0]) if len(c) > 1 else 0.2
        return np.concatenate([c - step / 2, [c[-1] + step / 2]])

    ax.set_facecolor(NODATA)
    mesh = ax.pcolormesh(edges(lon), edges(lat), np.ma.masked_invalid(data), cmap=cmap, norm=norm)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(lat)))))
    ax.grid(False)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)
    (fig or ax.figure).colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label=label)


def fig_maps(m, metric: str = "crps", rows=None, layers=None, min_count: float = 20, other=None):
    """One map per variable (other: CRPS of m minus CRPS of other, blue = m better)."""
    fig, axes = subplots(len(names(m)), 5.2, 2.8)
    for v, name in enumerate(names(m)):
        field = map_field(m, v, metric, rows, layers, min_count)
        if other is not None:
            field = field - map_field(other, names(other).index(name), metric, rows, layers, min_count)
        draw_map(axes.flat[v], m, field, "change" if other is not None else metric, LONG_NAMES.get(name, name),
                 unit(m, v, metric), fig)
    what = "CRPS difference (blue = first run better)" if other is not None else LABELS.get(metric, metric)
    finish(fig, axes, f"{what} per location", False)
    return fig


def fig_maps_layers(m, v: int, rows=None, min_count: float = 20):
    labs = layer_labels(m["layer_edges"])
    fig, axes = subplots(len(labs) + 1, 5.2, 2.8, ncol=3)
    for i, lab in enumerate(labs):
        draw_map(axes.flat[i], m, map_field(m, v, "crps", rows, [i], min_count), "crps", f"$\\sigma_0$ {lab}",
                 unit(m, v), fig)
    draw_map(axes.flat[len(labs)], m, map_field(m, v, "count", rows), "count", "scored points", "", fig)
    name = names(m)[v]
    finish(fig, axes, f"{LONG_NAMES.get(name, name)}: CRPS per location and water-mass layer", False)
    return fig


def fig_maps_variable(m, v: int, rows=None, min_count: float = 20):
    fig, axes = subplots(4, 5.2, 2.8, ncol=2)
    for ax, metric in zip(axes.flat, ("crps", "bias", "coverage", "spread_skill")):
        draw_map(ax, m, map_field(m, v, metric, rows, None, min_count), metric, LABELS[metric], unit(m, v, metric), fig)
    name = names(m)[v]
    finish(fig, axes, f"{LONG_NAMES.get(name, name)} per location", False)
    return fig


def fig_maps_oxycline(m, rows=None, min_count: float = 5):
    fig, axes = subplots(4, 5.2, 2.8, ncol=2)
    for ax, (metric, lab) in zip(axes.flat, (("oxycline_crps", "CRPS (m)"), ("oxycline_mae", "MAE (m)"),
                                              ("oxycline_coverage", "coverage"), ("oxycline_count", "profiles"))):
        kind = "coverage" if metric.endswith("coverage") else "count"
        draw_map(ax, m, map_field(m, 0, metric, rows, min_count=1 if kind == "count" else min_count), kind, lab, "", fig)
    finish(fig, axes, "Oxycline depth per location", False)
    return fig


def fig_maps_seasons(m, v: int, min_count: float = 20):
    """CRPS (top) and bias (bottom) of one variable per location and season, every row and layer pooled; the four
    seasons share their colour scale."""
    fig, axes = plt.subplots(2, 4, figsize=(5.2 * 4 + 0.6, 2.8 * 2 + 0.6), squeeze=False)
    n = m["ms_count"].astype(np.float64)
    for r, metric in enumerate(("crps", "bias")):
        with np.errstate(invalid="ignore", divide="ignore"):
            data = (m["ms_pinball" if metric == "crps" else "ms_bias"][v] / n * float(m["var_std"][v])
                    * (2.0 if metric == "crps" else 1.0))
        data = np.where(n >= min_count, data, np.nan)
        for s, season in enumerate(SEASONS):
            draw_map(axes[r, s], m, data[s], metric, f"{season}: {LABELS[metric]}", unit(m, v, metric), fig, scale=data)
    name = names(m)[v]
    finish(fig, axes, f"{LONG_NAMES.get(name, name)} per location and season", False)
    return fig


def cmd_maps(m, out: Path, other=None, rows=None, min_count: float = 20) -> None:
    for metric in ("crps", "bias", "spread_skill"):
        save(fig_maps(m, metric, rows, min_count=min_count), out, f"maps_{metric}")
    if other is not None:
        save(fig_maps(m, "crps", rows, min_count=min_count, other=other), out, "maps_crps_change")
    for v, name in enumerate(names(m)):
        save(fig_maps_layers(m, v, rows, min_count), out, f"maps_{name}_layers")
        save(fig_maps_variable(m, v, rows, min_count), out, f"maps_{name}")
        if "ms_count" in m:
            save(fig_maps_seasons(m, v, min_count), out, f"maps_{name}_seasons")
    save(fig_maps_oxycline(m, rows), out, "maps_oxycline")


# --------------------------------------------------------------------------------------- W&B
def fig_examples(ex: dict, max_depth: float = 250.0, max_members: int = 10):
    n = len(ex["truth"])
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n + 0.6, 5.0), sharey=True, squeeze=False)
    for i, ax in enumerate(axes[0]):
        v = int(ex["valid_len"][i])
        z = ex["depth"][:v]
        for j in range(min(max_members, ex["members"].shape[1])):
            ax.plot(ex["members"][i, j, :v], z, color=SERIES[0], lw=0.7, alpha=0.5, label="members" if j == 0 else None)
        ax.plot(ex["center"][i, :v], z, color=SERIES[1], lw=1.4, ls="--", label="median")
        ax.plot(ex["truth"][i, :v], z, color=INK, lw=1.5, label="truth")
        ob = ex["observed"][i, :v].astype(bool)
        ax.scatter(ex["truth"][i, :v][ob], z[ob], s=9, color=INK, zorder=3, label="observed")
        ax.set_xlabel("O$_2$ (mmol m$^{-3}$)")
    axes[0, 0].set_ylim(min(max_depth, float(ex["depth"][-1])), 0)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    i = [float(r) for r in ex["rows"]].index(float(ex["row"]))
    fig.suptitle(f"O$_2$ members, {row_text(ex, i)}", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def wandb_figures(r, full: bool, examples: Optional[dict]) -> dict:
    """Figures logged to W&B: a few every validation round, the full set with --eval-only."""
    runs = [("", r)]
    figs = {"crps_depth": lambda: fig_profiles(runs, "depth", "crps"), "heatmap_crps_depth": lambda: fig_heatmaps(r, "depth")}
    if "density_edges" in r:
        figs["crps_density"] = lambda: fig_profiles(runs, "density", "crps")
    if examples is not None:
        figs["example_profiles"] = lambda: fig_examples(examples)
    if full:
        figs.update({"coverage_depth": lambda: fig_profiles(runs, "depth", "coverage"),
                     "spread_skill_depth": lambda: fig_profiles(runs, "depth", "spread_skill"),
                     "crps_by_row": lambda: fig_by_row(runs, "crps_rel"), "oxycline": lambda: fig_oxycline(runs),
                     "time_bias": lambda: fig_time(runs, "bias"), "season_crps": lambda: fig_season(runs, "crps"),
                     "season_depth_crps": lambda: fig_season_depth(r, "crps"), "rank_histogram": lambda: fig_rank(runs),
                     "coherence": lambda: fig_coherence(runs), "maps_crps": lambda: fig_maps(r, "crps"),
                     "maps_o2_layers": lambda: fig_maps_layers(r, 0), "maps_oxycline": lambda: fig_maps_oxycline(r)})
        if "density_edges" in r:
            figs.update({"heatmap_crps_density": lambda: fig_heatmaps(r, "density"),
                         "relative_crps_density": lambda: fig_relative(r, "density")})
    out = {}
    for name, make in figs.items():
        try:
            out[name] = make()
        except Exception as exc:  # a figure must never cost an evaluation
            print(f"[wandb_figures] {name} skipped: {exc!r}")
    return out


# --------------------------------------------------------------------------------------- data
def oxycline_np(o2, valid_len, depth, thr):
    n, nz = o2.shape
    below = (o2 < thr) & (np.arange(nz)[None, :] < valid_len[:, None])
    j = np.where(below.any(axis=1), below.argmax(axis=1), -1)
    out = np.full(n, np.nan)
    ok = j > 0
    a, b = o2[ok, j[ok] - 1], o2[ok, j[ok]]
    out[ok] = depth[j[ok] - 1] + np.clip((a - thr) / np.maximum(a - b, 1e-9), 0, 1) * (depth[j[ok]] - depth[j[ok] - 1])
    return out


def cmd_data(processed_dir: str, year: int, step: int, out: Path, thr: float = 20.0, max_depth: float = 300.0,
             bin_width: float = 0.05) -> None:
    """The problem on one day of the data, and why density coordinates (remark of the teachers)."""
    root = Path(processed_dir)
    grid = json.loads((root / "grid.json").read_text())
    depth, ti, feats = np.asarray(grid["depth"]), int(grid["target_index"]), list(grid["features"])
    ydir = root / f"year_{year}"
    prof = np.load(ydir / "profiles.npy", mmap_mode="r")
    vlen = np.load(ydir / "valid_len.npy").astype(int)
    meta = np.load(ydir / "meta.npz")
    steps = np.unique(meta["time"])
    t = steps[min(step, len(steps) - 1)]
    sel = np.nonzero(meta["time"] == t)[0]
    date = str(np.datetime64(int(t), str(meta["time_unit"])))
    ox = oxycline_np(np.asarray(prof[sel, :, ti], dtype=np.float64), vlen[sel], depth, thr)
    deep = sel[vlen[sel] >= np.searchsorted(depth, max_depth)]
    pick = np.random.default_rng(0).choice(deep, size=min(150, len(deep)), replace=False) if len(deep) else sel[:0]

    fig, (ax_map, ax_z, ax_s) = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.9, 1, 1]})
    no = np.isnan(ox)
    ax_map.scatter(meta["lon"][sel][no], meta["lat"][sel][no], s=4, c=GRID, marker="s", lw=0, label="no oxycline")
    sc = ax_map.scatter(meta["lon"][sel][~no], meta["lat"][sel][~no], s=4, c=ox[~no], cmap=SEQ.reversed(), marker="s",
                        lw=0, vmin=np.nanpercentile(ox, 2), vmax=np.nanpercentile(ox, 98))
    ax_map.set_aspect(1.0 / np.cos(np.deg2rad(43.5)))
    ax_map.grid(False)
    ax_map.set_title(f"Oxycline depth (O$_2$ < {thr:g}), {date}")
    fig.colorbar(sc, ax=ax_map, fraction=0.03, pad=0.02, label="m").ax.invert_yaxis()
    si = feats.index("sigma0") if "sigma0" in feats else None
    zc, sgc = [], []
    for i in pick:
        n = vlen[i]
        ax_z.plot(prof[i, :n, ti], depth[:n], color=MUTED, lw=0.5, alpha=0.5)
        z = oxycline_np(np.asarray(prof[i, :n, ti], dtype=np.float64)[None], np.array([n]), depth[:n], thr)[0]
        if si is not None:
            ax_s.plot(prof[i, :n, ti], prof[i, :n, si], color=MUTED, lw=0.5, alpha=0.5)
            if np.isfinite(z):
                zc.append(z)
                sgc.append(np.interp(z, depth[:n], prof[i, :n, si]))
    ax_z.set_ylim(max_depth, 0)
    ax_z.set(xlabel="O$_2$ (mmol m$^{-3}$)", ylabel="depth (m)")
    if zc:
        ax_z.set_title(f"against depth: oxycline {np.percentile(zc, 5):.0f}-{np.percentile(zc, 95):.0f} m")
        ax_s.set_title(f"against density: oxycline {np.percentile(sgc, 5):.2f}-{np.percentile(sgc, 95):.2f}")
        sig_all = np.concatenate([prof[i, :vlen[i], si] for i in pick])
        ax_s.set_ylim(float(np.nanpercentile(sig_all, 99.5)), float(np.nanpercentile(sig_all, 1)))
        ax_s.set(xlabel="O$_2$ (mmol m$^{-3}$)", ylabel="$\\sigma_0$ (kg m$^{-3}$)")
    for ax in (ax_z, ax_s):
        ax.axvline(thr, color=INK2, lw=1, ls=":")
    fig.tight_layout()
    save(fig, out, "data_overview")
    if si is None:
        return
    # spread between the profiles of that day left at a fixed depth and at a fixed density, every variable
    zk = int(np.searchsorted(depth, max_depth, side="right"))
    P = np.asarray(prof[sel, :zk], dtype=np.float64)
    valid = np.arange(zk)[None, :] < vlen[sel][:, None]
    rows = []
    for f, name in enumerate(feats):
        v = P[..., f][valid]
        if name == "sigma0" or v.size < 10 or np.nanstd(v) <= 0:
            continue

        def within(ids):
            _, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
            return float(np.sum((v - (np.bincount(inv, weights=v) / cnt)[inv]) ** 2) / v.size)

        rows.append((name, np.sqrt(within(np.broadcast_to(np.arange(zk)[None], valid.shape)[valid]) / np.var(v)),
                     np.sqrt(within(np.floor(P[..., si][valid] / bin_width)) / np.var(v))))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r[1] for r in rows], 0.38, color=SERIES[0], label="at a fixed depth")
    ax.bar(xs + 0.2, [r[2] for r in rows], 0.38, color=SERIES[1], label=f"at a fixed density ({bin_width:g} kg m$^{{-3}}$ bins)")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT_NAMES.get(r[0], r[0]) for r in rows])
    ax.set(ylabel="spread left / total spread", ylim=(0, 1.05), title=f"Spread between profiles, upper {max_depth:g} m, {date}")
    ax.legend(loc="upper right", fontsize=10)
    save(fig, out, "variability_depth_vs_density")
    print("spread left (fixed depth | fixed density): " + ", ".join(f"{r[0]} {r[1]:.2f}|{r[2]:.2f}" for r in rows))


# --------------------------------------------------------------------------------------- members
def predict(cfg, grid: dict, stats, model, b: dict, key, n_members: int, noise=None):
    """Members of every variable for the batch b (as given by the dataset), in physical units: names, truth (N, Z, V),
    members as whole columns (N, M, Z, V) [the truth at the observed levels], observed levels (N, Z).
    key: as Task.inputs. noise: as fgn_members (default: every profile draws its own)."""
    import torch

    from metrics import fill_columns
    from train import Task, fgn_members

    task = Task(cfg, grid, stats, b["x"].device)
    mean, std = (torch.as_tensor(np.asarray(stats[k]), dtype=torch.float32, device=b["x"].device)[task.feat]
                 for k in ("mean", "std"))
    gen = torch.Generator(device=b["x"].device)
    gen.manual_seed(cfg.seed)
    truth, members, observed = [], [], []
    for a in range(0, len(b["x"]), cfg.eval_batch_size):
        sub = {k: v[a:a + cfg.eval_batch_size] for k, v in b.items()}
        inputs, x, obs, _ = task.inputs(model, sub, key)
        with torch.no_grad():
            z = noise[a:a + cfg.eval_batch_size] if noise is not None and noise.dim() == 3 else noise
            o2, idx, mask, aux = fgn_members(model, inputs, n_members, 0, gen, z)
        every = o2.unsqueeze(-1) if aux is None else torch.cat([o2.unsqueeze(-1), aux], dim=-1)  # (n, M, T, V)
        shift = (sub["x_raw"] - x).index_select(-1, task.feat)  # the drift reference removed from the model's values
        t = (sub["x_raw"].index_select(-1, task.feat) * std + mean).float()
        cols = [fill_columns(t[..., v], ((every[..., v] + shift[..., v].gather(1, idx).unsqueeze(1)) * std[v] + mean[v]).float(),
                             idx, mask) for v in range(len(task.names))]
        truth.append(t.cpu().numpy())
        members.append(torch.stack(cols, dim=-1).cpu().numpy())
        observed.append(obs.cpu().numpy())
    return task.names, np.concatenate(truth), np.concatenate(members), np.concatenate(observed)


def physical_coherence(var_names: List[str], truth, members, observed, valid_len) -> List[list]:
    """Is a member one physically possible water column? Rows [what, members, truth], on the predicted levels:
    density inversions (sigma0 decreasing downwards by more than 0.01 kg m-3) and, with gsw installed, the distance
    between a member's sigma0 and the equation of state applied to its own temperature and salinity."""
    if "sigma0" not in var_names:
        return []
    predicted = (np.arange(truth.shape[1])[None, :] < np.asarray(valid_len)[:, None]) & ~observed
    pair = predicted[:, 1:] & predicted[:, :-1]
    sig_t, sig_m = truth[..., var_names.index("sigma0")], members[..., var_names.index("sigma0")]

    def inversions(sig, pairs):
        down = np.diff(sig, axis=-1) < -0.01
        return 100.0 * float((down & pairs).sum()) / max(int(np.broadcast_to(pairs, down.shape).sum()), 1)

    rows = [["density inversions (% of level pairs)", inversions(sig_m, pair[:, None]), inversions(sig_t, pair)]]
    rows.append(["spread of sigma0 between the members (kg m-3)", float(sig_m.std(axis=1)[predicted].mean()), ""])
    try:
        import gsw
    except ImportError:
        return rows
    if "thetao" in var_names and "so" in var_names:
        def mismatch(a):  # sigma0 minus sigma0(T, S), as in preprocess.compute_sigma0
            sr = gsw.SR_from_SP(a[..., var_names.index("so")])
            return a[..., var_names.index("sigma0")] - gsw.sigma0(sr, gsw.CT_from_pt(sr, a[..., var_names.index("thetao")]))

        rows.append(["sigma0 minus sigma0(T, S) of the same member, RMS (kg m-3)",
                     float(np.sqrt(np.mean(mismatch(members)[np.broadcast_to(predicted[:, None], sig_m.shape)] ** 2))),
                     float(np.sqrt(np.mean(mismatch(truth)[predicted] ** 2)))])
    return rows


def cmd_ensemble(run: str, processed_dir: Optional[str], out: Path, row: Optional[float] = None, n_casts: int = 3,
                 n_members: int = 50, split: str = "val", max_depth: float = 250.0, show: int = 20) -> None:
    """Members of every variable for a few profiles (three members keep their colour in every panel: a member is
    one water column), and ensemble_coherence.csv from 200 profiles (physical_coherence).
    row: D_input / spacing (m), or for "neighbours" the number of casts given (grey; default as in training)."""
    import torch

    from dataset import cutoff_levels, load_grid, make_dataset, resolve_splits
    from train import load_run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, grid, stats, model = load_run(Path(run) / "best.pt", [f"processed_dir={processed_dir}"] if processed_dir else [],
                                       device)
    depth = np.asarray(grid["depth"])
    nb = cfg.experiment == "neighbours"
    if nb:
        cfg.neighbour_eval_casts, key = (None if row is None else int(row)), None
    else:
        row = row if row is not None else (20.0 if cfg.experiment == "sparse" else 50.0)
        key = float(row) if cfg.experiment == "sparse" else cutoff_levels(depth, [row], 1)[0][0]
    ds = make_dataset(cfg, grid, stats, resolve_splits(cfg, load_grid(cfg)["years"])[split], False, max_profiles=2000,
                      seed=12345, in_memory=False, extras=True, **({"pairs": False} if nb else {}))
    b = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
    keep = (b["valid_len"] > int(np.searchsorted(depth, max_depth))) & ((b["src_len"] > 0).any(dim=1) if nb else True)
    deep = {k: v[keep].to(device) for k, v in b.items()}  # profiles that reach max_depth

    some = {k: v[:200] for k, v in deep.items()}
    var_names, truth, members, observed = predict(cfg, grid, stats, model, some, key, n_members)
    numbers = physical_coherence(var_names, truth, members, observed, some["valid_len"].cpu().numpy())
    if numbers:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "ensemble_coherence.csv", "w", newline="") as fh:
            csv.writer(fh).writerows([["", "members", "truth"]] + [[r[0]] + [f"{x:.4g}" if x != "" else "" for x in r[1:]]
                                                                   for r in numbers])
        print((out / "ensemble_coherence.csv").read_text())

    pick = np.sort(np.random.default_rng(0).choice(int(keep.sum()), size=min(n_casts, int(keep.sum())), replace=False))
    b = {k: v[torch.as_tensor(pick, device=device)] for k, v in deep.items()}
    var_names, truth, members, observed = predict(cfg, grid, stats, model, b, key, n_members)
    fig, axes = plt.subplots(len(pick), len(var_names), figsize=(2.3 * len(var_names) + 0.6, 2.9 * len(pick) + 1.2),
                             sharey="row", squeeze=False)
    for v, name in enumerate(var_names):
        f = list(grid["features"]).index(name)
        mean, std = float(stats["mean"][f]), float(stats["std"][f])
        for c in range(len(pick)):
            ax, n = axes[c, v], int(b["valid_len"][c])
            if nb:  # the given casts, down to the bottom of the predicted profile (same drift reference: same day)
                shift = (b["x_raw"][c, :, f] - b["x"][c, :, f]).cpu().numpy()
                for s in range(b["src_len"].shape[1]):
                    ns = min(int(b["src_len"][c, s]), n)
                    ax.plot((b["src_x"][c, s, :ns, f].cpu().numpy() + shift[:ns]) * std + mean, depth[:ns], color=MUTED,
                            lw=1.0, label="given casts" if s == 0 else None)
            for j, mbr in enumerate(members[c, :show, :n, v]):
                ax.plot(mbr, depth[:n], color=SERIES[0], lw=0.6, alpha=0.4, label="members" if j == 0 else None)
            for j, color in enumerate(MEMBER_COLORS[:members.shape[1]]):
                ax.plot(members[c, j, :n, v], depth[:n], color=color, lw=1.1,
                        label="three of the members (same colour = same member)" if j == 0 else None)
            ax.plot(np.median(members[c, :, :n, v], axis=0), depth[:n], color=SERIES[0], lw=1.6, ls="--", label="median")
            ax.plot(truth[c, :n, v], depth[:n], color=INK, lw=1.4, label="truth")
            o = observed[c, :n]
            ax.plot(truth[c, :n, v][o], depth[:n][o], "o", color=INK, ms=2.5, label="observed" if o.any() else None)
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_title(LONG_NAMES.get(name, name), fontsize=10)
    for c in range(len(pick)):
        axes[c, 0].set_ylim(max_depth, 0)
        if nb:
            km = sorted(float(d) for d, n in zip(b["src_dxy"][c].norm(dim=-1).cpu(), b["src_len"][c].cpu()) if n > 0)
            axes[c, 0].set_ylabel(f"profile {c + 1}: casts at\n{', '.join(f'{d:.0f}' for d in km)} km\ndepth (m)", fontsize=9)
        else:
            axes[c, 0].set_ylabel(f"cast {c + 1}\ndepth (m)")
    what = "" if nb else ("spacing" if cfg.experiment == "sparse" else "D_input") + f" {row:g} m"
    fig.suptitle("Members of every variable" + (f", {what}" if what else ""), x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "ensemble_all_variables" + (f"_{int(row)}casts" if nb and row is not None else ""))


def map_correlation(anomaly: np.ndarray, lag_y: int = 1, lag_x: int = 1) -> float:
    """Correlation between the values of a map (M, ny, nx) at two grid points lag_y rows or lag_x columns apart
    (NaN = no value)."""
    num, a2, b2 = 0.0, 0.0, 0.0
    for a, b in ((anomaly[:, :, :-lag_x], anomaly[:, :, lag_x:]), (anomaly[:, :-lag_y], anomaly[:, lag_y:])):
        ok = np.isfinite(a) & np.isfinite(b)
        num, a2, b2 = num + float((a[ok] * b[ok]).sum()), a2 + float((a[ok] ** 2).sum()), b2 + float((b[ok] ** 2).sum())
    return num / max(np.sqrt(a2 * b2), 1e-30)


def smooth_noise(n_members: int, dim: int, shape: Tuple[int, int], cells: Tuple[float, float], seed: int) -> np.ndarray:
    """Noise that varies smoothly over the map (members, dim, ny, nx): white noise smoothed over `cells` grid cells
    (Gaussian) and rescaled, so that every grid point still gets a standard normal draw."""
    from scipy.ndimage import gaussian_filter

    pad = [int(3 * c) + 1 for c in cells]  # margins, cut off below: the two edges of the map stay independent
    full = (shape[0] + 2 * pad[0], shape[1] + 2 * pad[1])
    white = np.random.default_rng(seed).standard_normal((n_members, dim) + full).astype(np.float32)
    kernel = np.zeros(full, dtype=np.float32)
    kernel[0, 0] = 1.0
    scale = 1.0 / np.sqrt((gaussian_filter(kernel, cells, mode="wrap") ** 2).sum())
    smooth = gaussian_filter(white, (0, 0) + tuple(cells), mode="wrap") * scale
    return smooth[:, :, pad[0]:pad[0] + shape[0], pad[1]:pad[1] + shape[1]]


def cmd_field(run: str, processed_dir: Optional[str], out: Path, split: str = "val", step: int = 0,
              station_km: float = 60.0, var: str = "o2", depth_m: float = 50.0, noise_km: Optional[float] = None,
              n_members: int = 19) -> None:
    """"neighbours": one time step mapped from stations about station_km apart, with members that are coherent
    between profiles: a member uses ONE noise draw for the whole map (as it does for the whole column), or with
    noise_km a noise that varies smoothly over that distance. The last panel draws the noise per profile instead.
    Also prints how alike two profiles are in the members and in the real error (their correlation over the map)."""
    import torch

    from dataset import NeighbourDataset, load_grid, nearest_index, resolve_splits
    from train import load_run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, grid, stats, model = load_run(Path(run) / "best.pt", [f"processed_dir={processed_dir}"] if processed_dir else [],
                                       device)
    if cfg.experiment != "neighbours":
        raise SystemExit("figures.py field needs a run of experiment=neighbours")
    year = resolve_splits(cfg, load_grid(cfg)["years"])[split][0]
    ds = NeighbourDataset(cfg, grid, stats, [year], train=False, in_memory=False, extras=True, pairs=False)
    b, st_lat, st_lon = ds.field(0, step, station_km)
    b = {k: v.to(device) for k, v in b.items()}
    lat_ax, lon_ax = np.asarray(grid["lat"]), np.asarray(grid["lon"])
    iy, ix = (nearest_index(ax, b[k].cpu().numpy().astype(float)) for ax, k in ((lat_ax, "lat"), (lon_ax, "lon")))
    if noise_km:  # (N, M, noise_dim): the smooth noise map read at every profile
        cells = tuple(noise_km / km for km in ds.cell_km)
        z = smooth_noise(n_members, model.noise_dim, (len(lat_ax), len(lon_ax)), cells, cfg.seed)[:, :, iy, ix]
        noise = torch.as_tensor(z, device=device).permute(2, 0, 1)
    else:  # (M, noise_dim): one draw per member
        noise = torch.randn(n_members, model.noise_dim, generator=torch.Generator().manual_seed(cfg.seed)).to(device)
    var_names, truth, coherent, _ = predict(cfg, grid, stats, model, b, None, n_members, noise)
    own = predict(cfg, grid, stats, model, b, None, n_members)[2]  # every profile draws its own noise

    v, k = var_names.index(var), int(np.argmin(np.abs(np.asarray(grid["depth"]) - depth_m)))
    reach = b["valid_len"].cpu().numpy() > k

    def on_grid(values):  # (N,) or (N, M) at level k -> (ny, nx) or (M, ny, nx), NaN where nothing is predicted
        g = np.full(values.shape[1:] + (len(lat_ax), len(lon_ax)), np.nan)
        g[..., iy[reach], ix[reach]] = np.moveaxis(values[reach], 0, -1)
        return g

    t, s, o = on_grid(truth[:, k, v]), on_grid(coherent[:, :, k, v]), on_grid(own[:, :, k, v])
    mean = s.mean(axis=0)
    level = f"{LONG_NAMES.get(var, var)} at {grid['depth'][k]:.0f} m"
    far = [max(1, int(round(cfg.neighbour_max_km / km))) for km in ds.cell_km]
    lines = []
    for where, lag in ((f"next to each other (~{np.mean(ds.cell_km):.0f} km)", (1, 1)),
                       (f"{cfg.neighbour_max_km:g} km apart", far)):
        c = [map_correlation(a, *lag) for a in ((t - mean)[None], s - mean, o - o.mean(axis=0))]
        lines.append(f"{where}: real error {c[0]:.2f}, members {c[1]:.2f}, members with noise drawn per profile {c[2]:.2f}")
    print(f"{level}, correlation between two profiles\n  " + "\n  ".join(lines))

    # top left: the field; the other panels: departures from the mean of the members, which show the coherence
    panels = [("truth", t), ("mean of the members", mean), ("truth - mean (the real error)", t - mean),
              ("member 1 - mean", s[0] - mean), ("member 2 - mean", s[1] - mean),
              ("member 1 - mean, noise drawn per profile", o[0] - o.mean(axis=0))]
    lim = max(float(np.nanpercentile(np.abs(np.stack([d for _, d in panels[2:]])), 98)), 1e-9)
    scales = [(plt.Normalize(float(np.nanpercentile(t, 2)), float(np.nanpercentile(t, 98)) + 1e-9), SEQ)] * 2 \
        + [(TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), DIV)] * 4

    def edges(c):
        return np.concatenate([[c[0] - (c[1] - c[0]) / 2], 0.5 * (c[1:] + c[:-1]), [c[-1] + (c[-1] - c[-2]) / 2]])

    ys, xs = np.nonzero(np.isfinite(t))
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 7.0), squeeze=False)
    for ax, (title, data), (norm, cmap) in zip(axes.flat, panels, scales):
        ax.set_facecolor(NODATA)
        mesh = ax.pcolormesh(edges(lon_ax), edges(lat_ax), np.ma.masked_invalid(data), cmap=cmap, norm=norm)
        ax.plot(st_lon, st_lat, ".", color=INK, ms=2.5)
        ax.set_xlim(edges(lon_ax)[xs.min()], edges(lon_ax)[xs.max() + 1])
        ax.set_ylim(edges(lat_ax)[ys.min()], edges(lat_ax)[ys.max() + 1])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(lat_ax.mean()))))
        ax.grid(False)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)
        fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label=UNITS.get(var, ""))
    how = f"noise smooth over {noise_km:g} km" if noise_km else "one noise draw per member for the whole map"
    fig.suptitle(f"{level} mapped from stations {station_km:g} km apart (dots); {how}\nCorrelation between two profiles "
                 + "; ".join(lines).replace("members with noise drawn per profile", "noise per profile"),
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, out, f"field_{var}_{grid['depth'][k]:.0f}m" + (f"_noise{noise_km:g}km" if noise_km else ""))


# --------------------------------------------------------------------------------------- drift diagnostic
def cmd_drift(processed_dir: str, out: Path, lagged: Sequence[str], model_runs: Runs, overrides=(),
              eval_split: str = "val", max_depth: float = 500.0) -> None:
    """Bias each drift reference leaves in the evaluation years, from the basin-mean profiles alone: a model learns
    the training-years mean of its target, so it is biased by mean_train(target) - mean_eval(target)."""
    from config import Config, apply_overrides
    from dataset import lagged_years, load_or_compute_basin_monthly, resolve_splits

    cfg = apply_overrides(Config(), [f"processed_dir={processed_dir}"] + list(overrides))
    grid = json.loads((Path(processed_dir) / "grid.json").read_text())
    splits = resolve_splits(cfg, grid["years"])
    train, ev = splits["train"], splits[eval_split]
    depth, feats = np.asarray(grid["depth"], dtype=np.float64), list(grid["features"])
    bm = load_or_compute_basin_monthly(cfg, grid)
    years = [int(y) for y in bm["years"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        annual = np.nanmean(bm["mean"], axis=1)  # (n_years, Z, F)
        yi = {y: i for i, y in enumerate(years)}

        def bias(target):
            return np.nanmean(target[[yi[y] for y in train]], axis=0) - np.nanmean(target[[yi[y] for y in ev]], axis=0)

        refs = {"absolute (none)": bias(annual), "relative to the surface (shallowest)": bias(annual - annual[:, :1])}
        for spec in lagged:
            lag, window = (int(v) for v in spec.split("/"))
            ref = np.stack([np.nanmean(annual[[yi[y] for y in lagged_years(y, years, lag, window)]], axis=0) for y in years])
            refs[f"lagged {lag} y, {window}-y window"] = bias(annual - ref)
        clim = np.nanmean(annual[[yi[y] for y in train]], axis=0)
    shown = depth <= max_depth
    fig, axes = subplots(len(feats), 4.8, 3.3, sharey=True)
    edges_y = np.concatenate([[0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1]]])
    for f, name in enumerate(feats):
        ax = axes.flat[f]
        anom = (annual[:, :, f] - clim[None, :, f]).T
        norm, cmap, anom = norm_for("bias", anom, {})
        mesh = ax.pcolormesh(np.arange(len(years) + 1) - 0.5 + years[0], edges_y, np.ma.masked_invalid(anom), cmap=cmap,
                             norm=norm)
        for b in (max(train) + 0.5, max(splits["val"]) + 0.5):
            ax.axvline(b, color=INK, lw=1, ls="--")
        ax.set_ylim(max_depth, 0)
        ax.grid(False)
        ax.set_title(LONG_NAMES.get(name, name), fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02, label=UNITS.get(name, "").replace("-", ""))
    finish(fig, axes, "Basin-mean anomaly from the training-years mean (dashed: end of training / validation)", False)
    save(fig, out, "drift_anomaly")
    fig, axes = subplots(len(feats), 4.4, 3.8, sharey=True)
    colors = [MUTED, SERIES[1], SERIES[0], "#1c5cab", "#86b6ef", SERIES[2], SERIES[3]]
    for f, name in enumerate(feats):
        ax = axes.flat[f]
        for (label, b), c, ls in zip(refs.items(), colors, [":"] + ["-"] * 6):
            ax.plot(b[shown, f], depth[shown], color=c, ls=ls, label=label)
        for (label, m), mk in zip(model_runs, ["o", "s", "^"]):
            if name in names(m) and m["experiment"] == "cutoff":
                v = names(m).index(name)
                ax.plot(pooled(m["v_bias"][v], m["v_count"][v], 0)[shown], depth[shown], mk, color=INK, ms=3,
                        label=f"{label} (measured)")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_ylim(max_depth, 0)
        ax.set_title(LONG_NAMES.get(name, name), fontsize=11)
        ax.set_xlabel("bias" + (f" ({UNITS.get(name)})" if UNITS.get(name, "-") != "-" else ""), fontsize=9)
    finish(fig, axes, f"Bias each drift reference leaves in {ev[0]}-{ev[-1]} (training {train[0]}-{train[-1]})", True)
    save(fig, out, "drift_expected_bias")
    deep = shown & (depth >= 200)
    with open(out / "drift_expected_bias.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variable", "reference", "RMS bias column", "RMS bias 200 m and deeper"])
        for f, name in enumerate(feats):
            for label, b in refs.items():
                wr.writerow([name, label, f"{np.sqrt(np.nanmean(b[shown, f] ** 2)):.4g}",
                             f"{np.sqrt(np.nanmean(b[deep, f] ** 2)):.4g}"])
    print((out / "drift_expected_bias.csv").read_text())


# --------------------------------------------------------------------------------------- command line
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "maps", "ensemble", "field", "all"):
        s = sub.add_parser(name)
        s.add_argument("runs", nargs="+", help="LABEL=run folder or metrics file (first = reference)")
        s.add_argument("--out", default="figs")
        s.add_argument("--split", default="val")
        s.add_argument("--max-depth", type=float, default=300.0)
        s.add_argument("--rows", type=float, nargs="*", default=None, help="maps: pool only these D_input / spacings")
        s.add_argument("--min-count", type=float, default=20)
        s.add_argument("--row", type=float, default=None,
                       help="ensemble: D_input or spacing (default 50 / 20 m); neighbours: number of casts")
        s.add_argument("--processed-dir", default=None)
        s.add_argument("--step", type=int, default=0, help="field: index of the day (or month) in the first year of the split")
        s.add_argument("--station-km", type=float, default=60.0, help="field: distance between the stations")
        s.add_argument("--var", default="o2", help="field: variable shown")
        s.add_argument("--depth", type=float, default=50.0, help="field: depth shown (m)")
        s.add_argument("--noise-km", type=float, default=None,
                       help="field: noise smooth over this distance (default: one draw for the whole map)")
    d = sub.add_parser("data")
    d.add_argument("--processed-dir", required=True)
    d.add_argument("--year", type=int, required=True)
    d.add_argument("--step", type=int, default=0, help="index of the day (or month) in the year")
    d.add_argument("--out", default="figs/data")
    d.add_argument("--max-depth", type=float, default=300.0)
    dr = sub.add_parser("drift")
    dr.add_argument("--processed-dir", required=True)
    dr.add_argument("--out", default="figs/drift")
    dr.add_argument("--lagged", nargs="*", default=["1/1", "1/3", "1/5", "2/3"], metavar="LAG/WINDOW")
    dr.add_argument("--runs", nargs="*", default=[], help="LABEL=run: overlay the bias a trained model shows")
    dr.add_argument("--set", nargs="*", default=[], help="config overrides (split years ...)")
    a = p.parse_args()
    if a.cmd == "data":
        return cmd_data(a.processed_dir, a.year, a.step, Path(a.out), max_depth=a.max_depth)
    if a.cmd == "drift":
        return cmd_drift(a.processed_dir, Path(a.out), a.lagged, [load(s) for s in a.runs], a.set)
    out, run_dirs = Path(a.out), [Path(s.partition("=")[2] or s) for s in a.runs]
    if a.cmd == "ensemble":
        return cmd_ensemble(str(run_dirs[0]), a.processed_dir, out, a.row, split=a.split)
    if a.cmd == "field":
        return cmd_field(str(run_dirs[0]), a.processed_dir, out, a.split, a.step, a.station_km, a.var, a.depth, a.noise_km)
    runs = [load(s, a.split) for s in a.runs]
    first = metrics_file(a.runs[0], a.split)
    idw = first.with_name(first.stem + "_idw.npz")
    if idw.exists():  # "neighbours": the inverse-distance-weighted mean of the same casts, scored the same way
        runs.append(load(f"IDW={idw}"))
    for label, m in runs[1:]:
        if not np.array_equal(m["rows"], runs[0][1]["rows"]):
            raise SystemExit(f"'{label}' is scored on other rows ({m['row_name']}: {m['rows'].tolist()}) than "
                             f"'{runs[0][0]}': compare runs of the same experiment, evaluated on the same rows")
    rows = None if a.rows is None else [int(np.argmin(np.abs(runs[0][1]["rows"] - r))) for r in a.rows]
    if a.cmd in ("compare", "all"):
        cmd_compare(runs, out / "compare" if a.cmd == "all" else out, a.max_depth)
    if a.cmd in ("maps", "all"):
        cmd_maps(runs[0][1], out / "maps" if a.cmd == "all" else out, runs[1][1] if len(runs) > 1 else None, rows,
                 a.min_count)
    if a.cmd == "all" and (run_dirs[0] / "best.pt").exists():
        cmd_ensemble(str(run_dirs[0]), a.processed_dir, out, a.row, split=a.split)
        if str(runs[0][1]["experiment"]) == "neighbours":
            cmd_field(str(run_dirs[0]), a.processed_dir, out, a.split, a.step, a.station_km, a.var, a.depth, a.noise_km)
    if a.cmd == "all" and a.processed_dir:
        splits, year = [], int(json.loads((Path(a.processed_dir) / "grid.json").read_text())["years"][-1])
        if (run_dirs[0] / "best.pt").exists():  # the split years of the first run
            import torch

            c = torch.load(run_dirs[0] / "best.pt", map_location="cpu", weights_only=False)["cfg"]
            splits = [f"{k}_years={tuple(c[k + '_years'])}".replace(" ", "") for k in ("train", "val", "test")]
            year = int(list(c["val_years"])[0])
        cmd_data(a.processed_dir, year, 60, out / "data")
        cmd_drift(a.processed_dir, out / "drift", ["1/1", "1/3", "1/5", "2/3"], runs, splits)


if __name__ == "__main__":
    main()
