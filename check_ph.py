#!/usr/bin/env python
"""
Check of the pH profiles of a processed folder: do they jump back and forth below ~100 m, how often, where, in
which years, and is it already in the raw Copernicus files? Reads the data only, changes nothing.

    python check_ph.py > check_ph.txt        (login node, a few minutes; also writes check_ph.png)
    python check_ph.py --year 2019 --lat 43 --lon 34 --out check_ph.png [--set processed_dir=... raw_dir_daily=...]
The folder is processed_dir of config.py (pH corrected by preprocess.py: fix_ph); to see the product as it is,
give the folder made before the correction with --set processed_dir=...

A "big step" of a variable = a change of more than 20 % of its range between two consecutive levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import Config, apply_overrides, raw_root  # noqa: E402

BELOW = 100.0  # m: steps are counted below this depth (above, the seasonal layer has real sharp gradients)


def sample(folder: Path, n_max: int = 20_000):
    """Evenly spread profiles of one year: values (n, Z, F) with NaN below the seafloor, and their indices."""
    prof = np.load(folder / "profiles.npy", mmap_mode="r")
    idx = np.arange(0, prof.shape[0], max(1, prof.shape[0] // n_max))
    x = np.asarray(prof[idx], dtype=np.float64)
    valid_len = np.load(folder / "valid_len.npy")[idx].astype(np.int64)
    x[np.arange(x.shape[1])[None, :] >= valid_len[:, None]] = np.nan
    return x, idx


def big_steps(x: np.ndarray, depth: np.ndarray, threshold: float):
    """x (n, Z) -> per profile and level below BELOW: the next level is higher / lower by more than threshold."""
    step = np.diff(x, axis=1)
    deep = (depth[:-1] >= BELOW)[None, :]
    return (step > threshold) & deep, (step < -threshold) & deep


def value_range(x: np.ndarray) -> float:
    lo, hi = np.nanpercentile(x, [1, 99])
    return float(hi - lo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--lat", type=float, default=43.0)
    p.add_argument("--lon", type=float, default=34.0)
    p.add_argument("--out", default="check_ph.png")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    cfg = apply_overrides(Config(), a.set)
    root = Path(cfg.processed_dir)
    grid = json.loads((root / "grid.json").read_text())
    depth, feats = np.asarray(grid["depth"], dtype=np.float64), list(grid["features"])
    ph, dic = feats.index("ph"), feats.index("dissic")
    print(f"processed data {root}: {len(depth)} levels down to {depth[-1]:.0f} m, years {grid['years'][0]}-"
          f"{grid['years'][-1]}, " + ", ".join(f"{k}={grid.get(k)}" for k in ("temporal_resolution", "days_per_week",
                                                                              "kept_weekdays", "spatial_stride")))

    folder = root / f"year_{a.year}"
    x, idx = sample(folder)
    print(f"\n1. {a.year}, {len(x)} profiles: share of the profiles (%) with a big step up AND a big step down below "
          f"{BELOW:.0f} m\n   (0 for a smooth variable)")
    for f, name in enumerate(feats):
        up, down = big_steps(x[:, :, f], depth, 0.2 * value_range(x[:, :, f]))
        print(f"   {name:8s} {100 * (up.any(axis=1) & down.any(axis=1)).mean():6.2f}")

    threshold = 0.2 * value_range(x[:, :, ph])
    up, down = big_steps(x[:, :, ph], depth, threshold)
    print(f"\n2. pH in {a.year} per level (a big step = more than {threshold:.2f}): median, 1 % and 99 % of the profiles, "
          "share (%) of\n   the profiles with a big step up / down between this level and the next")
    print("   depth (m)  median    1 %   99 %     up   down   profiles")
    for z in range(len(depth) - 1):
        col = x[:, z, ph][np.isfinite(x[:, z, ph])]
        both = np.isfinite(x[:, z + 1, ph])
        if col.size:
            print(f"   {depth[z]:9.1f} {np.median(col):7.3f} {np.percentile(col, 1):6.3f} {np.percentile(col, 99):6.3f} "
                  f"{100 * up[both, z].mean() if both.any() else 0:6.1f} {100 * down[both, z].mean() if both.any() else 0:6.1f} "
                  f"{col.size:9d}")

    deep = x[:, depth >= 150.0, ph]
    deep = deep[np.isfinite(deep)]
    print(f"\n3. pH at 150 m and deeper in {a.year}: share of the values (%) per 0.1 of pH")
    if deep.size:
        edges = np.round(np.arange(np.floor(deep.min() * 10) / 10, deep.max() + 0.1, 0.1), 1)
        for lo, n in zip(edges[:-1], np.histogram(deep, edges)[0]):
            print(f"   {lo:4.1f}-{lo + 0.1:4.1f} {100 * n / deep.size:6.1f}")

    print(f"\n4. per year: share (%) of the profiles reaching 200 m with a big step up of pH below {BELOW:.0f} m")
    per_year = {}
    for year in grid["years"]:
        xy, _ = sample(root / f"year_{year}", 5_000)
        reach = np.isfinite(xy[:, np.searchsorted(depth, 200.0), ph])
        per_year[int(year)] = 100 * big_steps(xy[reach][:, :, ph], depth, threshold)[0].any(axis=1).mean() if reach.any() else np.nan
        print(f"   {year} {per_year[int(year)]:6.1f}")

    meta = np.load(folder / "meta.npz")
    unit = str(meta["time_unit"]) if "time_unit" in meta.files else "M"
    lat, lon, time = meta["lat"].astype(np.float64), meta["lon"].astype(np.float64), meta["time"].astype(np.int64)
    dist = (lat - a.lat) ** 2 + (lon - a.lon) ** 2
    here = np.nonzero(dist == dist.min())[0]  # this column at every time step of the year
    here = here[np.argsort(time[here])]
    prof = np.load(folder / "profiles.npy", mmap_mode="r")
    column = np.asarray(prof[here], dtype=np.float64)
    z200 = int(np.argmin(np.abs(depth - 200.0)))
    print(f"\n5. the column at {lat[here[0]]:.3f} N, {lon[here[0]]:.3f} E, at {depth[z200]:.0f} m, first time steps of {a.year}")
    print("   time         pH   DIC")
    for j in range(min(20, len(here))):
        print(f"   {np.datetime64(int(time[here[j]]), unit)}  {column[j, z200, ph]:6.3f} {column[j, z200, dic]:6.3f}")

    day = np.datetime64(int(time[here[0]]), unit)
    print(f"\n6. the same column on {day}: the raw file next to the processed profile")
    try:
        import xarray as xr

        from preprocess import group_year_files

        for path in group_year_files(Path(raw_root(cfg)), "bgc_car", a.year):
            with xr.open_dataset(path) as ds:
                t = np.nonzero(ds["time"].values.astype(f"datetime64[{unit}]") == day)[0]
                if t.size == 0:
                    continue
                raw = ds.isel(time=int(t[0])).sel(latitude=lat[here[0]], longitude=lon[here[0]], method="nearest")
                print(f"   {path}\n   raw point {float(raw['latitude']):.3f} N, {float(raw['longitude']):.3f} E, "
                      f"time {ds['time'].values[int(t[0])]}")
                print("   pH attributes:", dict(ds["ph"].attrs))
                print("   pH storage:", {k: v for k, v in ds["ph"].encoding.items()
                                         if k in ("dtype", "scale_factor", "add_offset", "_FillValue")})
                print("   file:", {k: ds.attrs[k] for k in ("title", "source", "institution", "references") if k in ds.attrs})
                names = [v for v in ("dissic", "talk", "ph") if v in ds]
                print("   depth (m) " + " ".join(f"raw {v:>7s}" for v in names) + "   processed dissic      ph")
                for zr, d in enumerate(raw["depth"].values):
                    if d > depth[-1] + 1:
                        break
                    z = int(np.argmin(np.abs(depth - d)))
                    same = abs(depth[z] - d) < 0.01
                    print(f"   {d:9.2f} " + " ".join(f"{float(raw[v].values[zr]):11.4f}" for v in names)
                          + (f"   {column[0, z, dic]:16.4f} {column[0, z, ph]:7.4f}" if same else ""))
                break
        else:
            print(f"   no raw file of bgc_car holds {day}")
    except Exception as e:  # the first five parts do not need the raw files
        print(f"   the raw file could not be read: {type(e).__name__}: {e}")

    first = np.nonzero(time == time[here[0]])[0]  # every profile of that time step
    vlen = np.load(folder / "valid_len.npy").astype(np.int64)
    full = first[vlen[first] == len(depth)]
    shown = full[:: max(1, len(full) // 40)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, f, name in ((axes[0], ph, "pH"), (axes[1], dic, "DIC")):
        ax.plot(np.asarray(prof[shown])[:, :, f].T, depth, lw=0.6, color="#2a78d6", alpha=0.6)
        ax.invert_yaxis()
        ax.set(title=f"{name}, {len(shown)} deep columns, {day}", xlabel=name, ylabel="depth (m)")
    deep_here = first[vlen[first] > z200]
    sc = axes[2].scatter(lon[deep_here], lat[deep_here], c=np.asarray(prof[deep_here])[:, z200, ph], s=5, marker="s")
    fig.colorbar(sc, ax=axes[2])
    axes[2].set(title=f"pH at {depth[z200]:.0f} m, {day}", xlabel="longitude", ylabel="latitude")
    axes[3].bar(list(per_year), list(per_year.values()), color="#2a78d6")
    axes[3].set(title=f"profiles with a big step up of pH below {BELOW:.0f} m", xlabel="year", ylabel="%", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"\nfigure: {a.out}")


if __name__ == "__main__":
    main()
