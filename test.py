#!/usr/bin/env python
"""
Checks that the code works end to end, on synthetic data and on CPU (~8 min): every experiment and drift mode,
resume, evaluation, DDP and every figure command.

    python test.py [--keep FOLDER]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ENV = {**os.environ, "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}


def run(*args, ok=True, cwd=CODE, prefix=(sys.executable,)) -> str:
    p = subprocess.run([*prefix, *map(str, args)], cwd=cwd, env=ENV, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if (p.returncode == 0) != ok:
        print(out[-4000:])
        raise SystemExit(f"FAILED ({'exit ' + str(p.returncode)}): {' '.join(map(str, args))}")
    return out


def check(cond, what: str) -> None:
    if not cond:
        raise SystemExit(f"FAILED: {what}")
    print(f"  ok  {what}")


def synthetic_data(out: Path, years=range(2000, 2008)) -> None:
    """Black-Sea-like profiles: isopycnals dome in two gyres, O2 / NO3 / PO4 follow sigma0, the upper layer drifts.
    As in the reanalysis, what differs from one profile to the next varies slowly over the sea and with depth, so
    that neighbouring profiles (and levels) resemble each other."""
    rng = np.random.default_rng(0)
    depth = np.array([2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 115, 130, 150, 175, 200, 250, 300,
                      400, 500])
    lat, lon = np.arange(41.0, 46.6, 0.4), np.arange(28.0, 41.6, 0.4)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    ell = ((LO - 34.7) / 6.8) ** 2 + ((LA - 43.3) / 1.9) ** 2
    bottom = np.where(ell < 0.55, 2000.0, np.clip(2000 * (1 - ell) / 0.45, 20, 2000))
    gyre = sum(np.exp(-(((LO - x) / 1.6) ** 2 + ((LA - 43.2) / 0.9) ** 2)) for x in (31.5, 37.5))
    feats = ["o2", "thetao", "so", "sigma0", "chl", "no3", "po4", "dissic", "ph"]

    def smooth():  # a random map (std ~1) made of a few long waves, plus a little noise per profile
        waves = sum(rng.standard_normal() * np.sin(kx * LO + ky * LA + phase)
                    for kx, ky, phase in rng.uniform([-1.5, -3.0, 0.0], [1.5, 3.0, 6.3], (4, 3)))
        return 0.65 * waves + 0.1 * rng.standard_normal(LO.shape)

    for year in years:
        rows = []
        for month in range(12):
            summer = max(np.cos(2 * np.pi * (month - 7) / 12), 0)
            dome, oxycline, wiggle1, wiggle2 = smooth(), smooth(), smooth(), smooth()
            for iy, ix in np.argwhere(ell < 1.0):
                g = gyre[iy, ix] + 0.15 * dome[iy, ix]
                sig = 17.0 - 5.2 * np.exp(-depth * (1 + 0.6 * g) / 70) - 1.5 * summer * np.exp(-depth / 15)
                sig = sig + 0.03 * (wiggle1[iy, ix] * np.sin(depth / 30.0) + wiggle2[iy, ix] * np.cos(depth / 70.0))
                o2 = 280 * np.clip((15.7 + 0.08 * oxycline[iy, ix] - sig) / 1.6, 0, 1) ** 1.3
                drift = (year - years[0]) * np.exp(-depth / 80)
                arr = np.stack([o2, 8.9 + 15 * summer * np.exp(-depth / 12) + 0.05 * drift,
                                18 + 0.85 * (sig - 11) + 0.04 * drift, sig,
                                np.log10(0.05 + 1.2 * np.exp(-((depth - 30) / 20) ** 2) + 1e-3),
                                6 * np.exp(-((sig - 15.4) / 0.3) ** 2) + 0.1, 0.05 + 6 / (1 + np.exp(-(sig - 16) / 0.15)),
                                3 + 0.12 * (sig - 11) + 0.004 * drift, 8.3 - 0.13 * (sig - 11)], axis=-1)
                vl = max(int(np.searchsorted(depth, bottom[iy, ix], side="right")), 3)
                arr[vl:] = 0
                rows.append((arr, vl, [lat[iy], lon[ix], np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                                       np.log1p(depth[vl - 1]), 20 + 30 * (1 - summer), 0.05 * (1 - g)],
                             (year - 1970) * 12 + month, lat[iy], lon[ix]))
        d = out / f"year_{year}"
        d.mkdir(parents=True)
        np.save(d / "profiles.npy", np.array([r[0] for r in rows], dtype=np.float32))
        np.save(d / "valid_len.npy", np.array([r[1] for r in rows], dtype=np.int16))
        np.save(d / "context.npy", np.array([r[2] for r in rows], dtype=np.float32))
        np.savez(d / "meta.npz", time=np.array([r[3] for r in rows]), time_unit="M", lat=np.array([r[4] for r in rows]),
                 lon=np.array([r[5] for r in rows]))
        (d / "_DONE").touch()
    (out / "grid.json").write_text(json.dumps({
        "depth": depth.tolist(), "features": feats, "context": ["lat", "lon", "month_sin", "month_cos", "log_bottom_depth",
                                                                "mlotst", "zos"],
        "ssh_anomaly": True, "target": "o2", "target_index": 0, "log_transform_vars": ["chl"], "spatial_stride": 1,
        "max_depth_m": 500.0, "lat": lat.tolist(), "lon": lon.tolist(), "years": list(years)}))


def metrics(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def same_metrics(a: dict, b: dict) -> float:
    """Largest relative difference between two metrics files (inf if their keys or shapes differ)."""
    if a.keys() != b.keys():
        return float("inf")
    worst = 0.0
    for k in a:
        if a[k].dtype.kind in "fiu":
            if a[k].shape != b[k].shape or not np.array_equal(np.isfinite(a[k]), np.isfinite(b[k])):
                return float("inf")
            x, y = a[k][np.isfinite(a[k])].astype(np.float64), b[k][np.isfinite(b[k])].astype(np.float64)
            if x.size:
                worst = max(worst, float((np.abs(x - y) / np.maximum(np.abs(x), 1e-6)).max()))
    return worst


def synthetic(tmp: Path) -> None:
    import torch

    data, runs, figs = tmp / "data", tmp / "runs", tmp / "figs"
    print("synthetic data ...")
    synthetic_data(data)
    common = [f"processed_dir={data}", "train_years=(2000,2001,2002,2003,2004)", "val_years=(2005,)", "test_years=(2006,2007)",
              "d_model=32", "d_ff=64", "n_heads=4", "n_enc_layers=1", "n_dec_layers=1", "batch_size=32",
              "eval_batch_size=128", "steps_per_epoch=8", "warmup_steps=4", "val_max_profiles=200", "num_workers=0",
              "log_every=4", "fgn_eval_members=5", "input_drift_vars=('thetao','so','sigma0','dissic','ph')",
              "drift_window_years=2"]

    def train(name, *settings, epochs=2, ok=True):
        return run("train.py", "--set", *common, f"ckpt_dir={runs / name}", f"epochs={epochs}", *settings, ok=ok)

    def evaluate(name, split="val", n=200, *settings):
        run("train.py", "--eval-only", "--split", split, "--max-profiles", n, "--set", f"processed_dir={data}",
            f"ckpt_dir={runs / name}", *settings)
        return metrics(runs / name / f"{split}_metrics.npz")

    print("experiment 1 (cutoff), lagged drift")
    train("cut", "experiment=cutoff", "input_drift_mode=lagged")
    check(all((runs / "cut" / f).exists() for f in ("last.pt", "best.pt", "val_metrics_best.npz")), "checkpoints + scores")
    r = metrics(runs / "cut" / "val_metrics_best.npz")
    check(np.isfinite(r["objective"]) and (r["v_count"].sum(axis=(1, 2)) > 0).all(), "every variable scored")
    check(np.isfinite(r["v_crps"][r["v_count"] > 0]).all() and r["m_count"].sum() > 0 and r["t_count"].sum() > 0,
          "depth / density / map / monthly scores")
    n = r["v_count"][0].sum()
    check(r["s_count"].sum() == n and r["ms_count"].sum() == n and np.allclose(r["rank"].sum(axis=(1, 2)), n),
          "calendar-month, seasonal-map and rank-histogram scores count the same points")
    check(np.isfinite(r["coh_levels"]).all() and (r["coh_levels"][:, 1] > 0).all() and r["coh_variables"].shape[2:] == (9, 9),
          "coherence scores (between levels, between variables)")
    check(any(f.startswith("basin_monthly_") for f in os.listdir(data)), "lagged reference cached")
    out = train("cut", "experiment=cutoff", "input_drift_mode=lagged", "variable_weighting=o2", epochs=3)
    ck = torch.load(runs / "cut" / "last.pt", map_location="cpu", weights_only=False)
    check("Resuming from" in out and ck["state"]["epoch"] == 3 and ck["state"]["step"] == 24, "resume: epoch 3, step 24")
    check(ck["cfg"]["variable_weighting"] == "equal", "resume keeps the loss of the checkpoint")
    check(same_metrics(evaluate("cut"), metrics(runs / "cut" / "val_metrics_best.npz")) < 1e-5,
          "--eval-only of best.pt reproduces its validation scores")
    evaluate("cut", "test", 100)

    print("experiment 2 (sparse), shallowest drift")
    train("sparse", "experiment=sparse", "input_drift_mode=shallowest")
    r = evaluate("sparse")
    check(str(r["experiment"]) == "sparse" and r["observed"].shape == (4, 24), "sparse scores per spacing")
    check(same_metrics(r, metrics(runs / "sparse" / "val_metrics_best.npz")) < 1e-5, "sparse --eval-only reproduces")

    print("experiment 3 (neighbours), lagged drift")
    nb = ["experiment=neighbours", "input_drift_mode=lagged", "neighbour_max_km=100", "neighbour_val_km=(40,70,100)"]
    train("nb", *nb)
    r, idw = metrics(runs / "nb" / "val_metrics_best.npz"), metrics(runs / "nb" / "val_metrics_best_idw.npz")
    check(str(r["experiment"]) == "neighbours" and (r["v_count"][0].sum(axis=1) > 0).all(), "scores per distance row")
    check(np.array_equal(r["v_count"], idw["v_count"]) and int(idw["n_members"]) == 1,
          "IDW baseline scored on the same points")
    check(same_metrics(evaluate("nb"), r) < 1e-5 and same_metrics(metrics(runs / "nb" / "val_metrics_idw.npz"), idw) < 1e-5,
          "neighbours --eval-only reproduces (same casts drawn)")
    run("train.py", "--eval-only", "--split", "val", "--max-profiles", 200, "--tag", "1cast", "--set", f"processed_dir={data}",
        f"ckpt_dir={runs / 'nb'}", "neighbour_eval_casts=1")
    check((runs / "nb" / "val_metrics_1cast_idw.npz").exists(), "evaluation with exactly one cast (--tag)")
    sys.path.insert(0, str(CODE))
    from config import Config, apply_overrides
    from dataset import NeighbourDataset, load_grid, load_or_compute_stats

    cfg = apply_overrides(Config(), [f"processed_dir={data}", "neighbour_max_km=100", "neighbour_eval_casts=1"])
    ds = [NeighbourDataset(cfg, load_grid(cfg), load_or_compute_stats(cfg, load_grid(cfg), [2000, 2001, 2002, 2003, 2004]),
                           [2005], train=train, in_memory=False) for train in (True, False)]
    casts = [[int((d[i]["src_len"] > 0).sum()) for i in range(0, 400, 7)] for d in ds]
    check(max(casts[0]) > 1 and max(casts[1]) == 1, "neighbour_eval_casts applies to evaluation only")
    train("nb_bad", *nb, "input_drift_mode=shallowest", epochs=1, ok=False)
    check(True, "neighbours + shallowest drift is refused")

    print("no drift, O2-weighted loss, EMA, one variable weighted twice")
    train("none", "input_drift_mode=none", "variable_weighting=o2", "ema_decay=0.9", epochs=1)
    train("w2", "input_drift_mode=shallowest", "variable_weights={'o2':2.0}", epochs=1)
    train("bad", "input_drift_mode=basin", epochs=1, ok=False)
    check(True, "input_drift_mode=basin is refused")

    print("DDP (2 CPU processes)")
    run("--nproc_per_node=2", "train.py", "--set", *common, f"ckpt_dir={runs / 'ddp'}", "epochs=1", "experiment=sparse",
        "input_drift_mode=lagged", prefix=(sys.executable, "-m", "torch.distributed.run"))
    check((runs / "ddp" / "val_metrics_best.npz").exists(), "torchrun --nproc_per_node=2")

    print("figures")
    run("figures.py", "all", f"new={runs / 'cut'}", f"other={runs / 'none'}", "--out", figs / "cut", "--processed-dir", data)
    run("figures.py", "all", f"sparse={runs / 'sparse'}", f"ddp={runs / 'ddp'}", "--out", figs / "sparse")
    run("figures.py", "maps", runs / "cut" / "test_metrics.npz", "--rows", "10", "50", "--out", figs / "test_maps")
    run("figures.py", "all", f"model={runs / 'nb'}", "--out", figs / "nb")
    run("figures.py", "compare", f"1 cast={runs / 'nb' / 'val_metrics_1cast.npz'}", f"1-4 casts={runs / 'nb'}",
        "--out", figs / "nb_casts")
    run("figures.py", "ensemble", runs / "nb", "--row", 1, "--out", figs / "nb")
    out = run("figures.py", "field", runs / "nb", "--station-km", 90, "--out", figs / "nb_field")
    run("figures.py", "field", runs / "nb", "--station-km", 90, "--noise-km", 150, "--out", figs / "nb_field")
    line = next(x for x in out.splitlines() if "next to each other" in x)  # real error, members, noise per profile
    c = [float(x) for x in re.findall(r"-?\d+\.\d\d", line.split("):")[1])]
    check(c[1] > 0.5 > abs(c[2]), f"field: members sharing their noise are coherent between profiles ({c[1]:.2f}), "
                                  f"members with their own noise are not ({c[2]:.2f})")
    run("figures.py", "layouts", runs / "nb", "--every", 2, 3, "--steps", 0, 6, "--out", figs / "nb_layouts")
    table = (figs / "nb_layouts" / "layouts.csv").read_text().splitlines()
    check(len(table) == 5 and table[1].split(",")[2] == table[2].split(",")[2],
          "layouts: grid and Argo-like stations, the same number of each")
    run("figures.py", "drift", "--processed-dir", data, "--out", figs / "drift", "--runs", f"cut={runs / 'cut'}",
        "--set", "train_years=(2000,2001,2002,2003,2004)", "val_years=(2005,)", "test_years=(2006,2007)")
    n = {d.name: len(list(d.rglob("*.png"))) for d in figs.iterdir()}
    check(n["cut"] >= 73 and n["sparse"] >= 69 and n["test_maps"] >= 31 and n["drift"] == 2 and n["nb"] >= 71
          and n["nb_casts"] >= 36 and n["nb_field"] == 2 and n["nb_layouts"] == 3,
          f"figures written {n}")
    import figures

    figs_wandb = figures.wandb_figures(metrics(runs / "cut" / "val_metrics.npz"), True, None)
    check(len(figs_wandb) >= 16, f"{len(figs_wandb)} W&B figures")
    print(f"\nALL CHECKS PASSED (figures in {figs})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep", default=None, help="folder for the test (default: a temporary one, deleted)")
    a = p.parse_args()
    if a.keep:
        shutil.rmtree(a.keep, ignore_errors=True)
        return synthetic(Path(a.keep))
    with tempfile.TemporaryDirectory() as tmp:
        synthetic(Path(tmp))


if __name__ == "__main__":
    main()
