#!/usr/bin/env python
"""
Raw Copernicus NetCDF -> per-year profile arrays on the BGC depth levels (one profile per time step x strided
ocean point): <processed_dir>/year_YYYY/{profiles.npy (N,Z,F), valid_len.npy, context.npy, meta.npz}, grid.json.
Physics is interpolated onto the BGC levels and extended at most max_bottom_extend_levels towards the seafloor,
interior gaps of <= max_interior_gap_levels are filled, and each profile stops at its first missing value.

    python preprocess.py --workers 4 [--years 1993 1994] [--set spatial_stride=1 max_depth_m=500]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Set, Tuple

import numpy as np
import xarray as xr

from config import (DATASET_GROUPS, DERIVED_FEATURES, PHYSICAL_FEATURES, REFERENCE_GROUP, Config,
                    apply_overrides, kept_weekdays, raw_root)

log = logging.getLogger("preprocess")
CHUNK = 32_768  # profiles per chunk for the vectorised gap filling (bounds peak memory)
CONTEXT_BASE = ["lat", "lon", "month_sin", "month_cos", "log_bottom_depth"]


# --------------------------------------------------------------------------------------- bookkeeping
def variable_to_group() -> Dict[str, str]:
    return {v: name for name, g in DATASET_GROUPS.items() for v in g.variables}


def required_variables(cfg: Config) -> Tuple[Set[str], Set[str]]:
    """Raw 3-D and 2-D variables needed for the configured features."""
    v2g = variable_to_group()
    raw3d: Set[str] = set()
    for feat in cfg.profile_features:
        raw3d |= set(DERIVED_FEATURES.get(feat, (feat,)))
    raw2d = set(cfg.context_vars_2d)
    if cfg.target_var not in cfg.profile_features:
        raise ValueError("target_var must be one of profile_features (it is also an input above D_input)")
    if cfg.target_var in cfg.log_transform_vars:
        raise ValueError("Do not log-transform the target; the non-negativity bound assumes linear units")
    for v in raw3d | raw2d:
        if v not in v2g:
            raise KeyError(f"Variable '{v}' is not provided by any group in config.DATASET_GROUPS")
    for v in raw3d:
        if not DATASET_GROUPS[v2g[v]].is_3d:
            raise ValueError(f"'{v}' is 2-D; list it in context_vars_2d instead")
    for v in raw2d:
        if DATASET_GROUPS[v2g[v]].is_3d:
            raise ValueError(f"'{v}' is 3-D; list it in profile_features instead")
    return raw3d, raw2d


# Raw file names: <group>_<my|myint>_<YYYY>.nc (monthly means, one file per year) or
# <group>_<my|myint>_<YYYY>-<MM>.nc (daily means, one file per month).
def _raw_file_regex(group: str, year: str = r"\d{4}") -> re.Pattern:
    return re.compile(rf"^{re.escape(group)}_(my|myint)_({year})(?:-\d{{2}})?\.nc$")


def available_years(raw_dir: Path, group: str) -> Set[int]:
    pattern = _raw_file_regex(group)
    years = set()
    for f in (raw_dir / group).glob("*.nc"):
        m = pattern.match(f.name)
        if m:
            years.add(int(m.group(2)))
    return years


def group_year_files(raw_dir: Path, group: str, year: int) -> list:
    """All raw files of one group and year, sorted so that "my" files come before "myint" files."""
    pattern = _raw_file_regex(group, str(year))
    files = sorted(f for f in (raw_dir / group).glob(f"{group}_*_{year}*.nc") if pattern.match(f.name))
    if not files:
        raise FileNotFoundError(f"No raw files for {group} {year} in {raw_dir / group}")
    return files


def time_unit(cfg: Config) -> str:
    return "D" if cfg.temporal_resolution == "daily" else "M"


def open_group_year(cfg: Config, raw_dir: Path, group: str, year: int, ref: dict) -> xr.Dataset:
    """Open all files of one group/year ("my" + possibly "myint"), round time to the month (monthly means) or
    the day (daily means), keep only the chosen weekdays (daily), and pick the reference grid points.

    Day selection and horizontal regridding are applied to each file BEFORE the files are joined, while the
    data is still lazy, so only the kept days and grid points are ever read into memory. Duplicate time steps
    keep their "my" value, because "my" files sort first."""
    unit = time_unit(cfg)
    keep = np.asarray(kept_weekdays(cfg)) if unit == "D" else None
    parts = []
    for f in group_year_files(raw_dir, group, year):
        ds = xr.open_dataset(f)
        stamps = ds["time"].values.astype(f"datetime64[{unit}]")
        ds = ds.assign_coords(time=stamps)
        if keep is not None:  # days since 1970-01-01, modulo 7 = position in the week
            ds = ds.isel(time=np.nonzero(np.isin(stamps.astype(np.int64) % 7, keep))[0])
        parts.append(to_horizontal_grid(ds, ref))
    ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time", data_vars="minimal",
                                                    coords="minimal", compat="override")
    _, first = np.unique(ds["time"].values, return_index=True)  # first occurrence of each, in time order
    return ds.isel(time=first)


def reference_grid(cfg: Config, raw_dir: Path, year: int) -> dict:
    with xr.open_dataset(group_year_files(raw_dir, REFERENCE_GROUP, year)[0]) as ds:  # coordinates only
        lat, lon = ds["latitude"].values, ds["longitude"].values
        depth = ds["depth"].values.astype(np.float64)
    if cfg.max_depth_m is not None:
        depth = depth[depth <= cfg.max_depth_m]
    native_res = float(np.median(np.abs(np.diff(lat))))
    s = cfg.spatial_stride
    return {"lat": lat[::s], "lon": lon[::s], "depth": depth, "tol": 0.51 * native_res}


# --------------------------------------------------------------------------------------- regridding
def to_horizontal_grid(da: xr.DataArray, ref: dict) -> xr.DataArray:
    # Both reanalyses use the same 1/40 deg mesh, so nearest-neighbour with a half-cell tolerance is exact;
    # anything further away (e.g. a different mesh) becomes NaN instead of silently mixing cells.
    return da.reindex(latitude=ref["lat"], longitude=ref["lon"], method="nearest", tolerance=ref["tol"])


def to_vertical_grid(da: xr.DataArray, depth: np.ndarray) -> xr.DataArray:
    try:  # same vertical grid (other BGC variables): exact selection
        return da.sel(depth=depth, method="nearest", tolerance=1e-2)
    except KeyError:  # physics: linear interpolation, constant extrapolation above the first level
        src = da["depth"].values
        out = da.interp(depth=np.clip(depth, src.min(), src.max()), method="linear")
        return out.assign_coords(depth=depth)


def compute_sigma0(theta: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Potential density anomaly sigma_theta (kg m-3, ref. 0 dbar) with TEOS-10.

    The Black Sea's ionic composition (strong riverine influence) is not represented by the TEOS-10
    absolute-salinity-anomaly atlas, so Reference Salinity is used (delta S_A = 0). Plenty accurate for a
    model feature: in the Black Sea the oxic/suboxic interface follows isopycnals (sigma_theta ~ 15.4-16.2)
    far more closely than it follows depth.
    """
    import gsw

    sa = gsw.SR_from_SP(sp)
    ct = gsw.CT_from_pt(sa, theta)
    return np.asarray(gsw.sigma0(sa, ct), dtype=np.float32)


# --------------------------------------------------------------------------------------- missing values
def _chunked(arr: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    out = np.empty_like(arr)
    for a in range(0, len(arr), CHUNK):
        out[a:a + CHUNK] = fn(arr[a:a + CHUNK])
    return out


def fill_interior_gaps(a: np.ndarray, depth: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation in depth across NaN runs of <= max_gap levels bounded by valid values. a: (n,Z,F)."""
    if max_gap <= 0:
        return a
    _, nz, _ = a.shape
    valid = np.isfinite(a)
    lev = np.arange(nz, dtype=np.int32)[None, :, None]
    prev = np.maximum.accumulate(np.where(valid, lev, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, lev, nz)[:, ::-1], axis=1)[:, ::-1]
    fill = ~valid & (prev >= 0) & (nxt < nz) & ((nxt - prev - 1) <= max_gap)
    if not fill.any():
        return a
    p, q = np.clip(prev, 0, nz - 1), np.clip(nxt, 0, nz - 1)
    vp, vq = np.take_along_axis(a, p, axis=1), np.take_along_axis(a, q, axis=1)
    zp, zq = depth[p], depth[q]
    w = (depth[None, :, None] - zp) / np.maximum(zq - zp, 1e-6)
    return np.where(fill, vp + w * (vq - vp), a).astype(a.dtype)


def ffill_levels(a: np.ndarray, limit: int) -> np.ndarray:
    """Forward-fill downwards by at most `limit` levels. a: (n,Z,F)."""
    if limit <= 0:
        return a
    _, nz, _ = a.shape
    valid = np.isfinite(a)
    lev = np.arange(nz, dtype=np.int32)[None, :, None]
    prev = np.maximum.accumulate(np.where(valid, lev, -1), axis=1)
    fill = ~valid & (prev >= 0) & ((lev - prev) <= limit)
    if not fill.any():
        return a
    return np.where(fill, np.take_along_axis(a, np.clip(prev, 0, None), axis=1), a)


# --------------------------------------------------------------------------------------- one year
def process_year(cfg: Config, year: int, raw_dir: Path, out_dir: Path, ref: dict) -> int:
    ydir = out_dir / f"year_{year}"
    if (ydir / "_DONE").exists():
        return int(np.load(ydir / "valid_len.npy", mmap_mode="r").shape[0])

    raw3d, raw2d = required_variables(cfg)
    v2g = variable_to_group()
    dsets = {g: open_group_year(cfg, raw_dir, g, year, ref) for g in sorted({v2g[v] for v in raw3d | raw2d})}
    # xarray may store the rounded time axis as datetime64[ns]; keep both representations
    times = reduce(np.intersect1d, [d["time"].values for d in dsets.values()])
    stamps = times.astype(f"datetime64[{time_unit(cfg)}]")  # months or days
    if stamps.size == 0:
        log.warning("%d: no time step common to all datasets, skipped", year)
        return 0
    depth = ref["depth"]
    nz = len(depth)

    fields: Dict[str, np.ndarray] = {}
    for v in sorted(raw3d):
        da = to_horizontal_grid(dsets[v2g[v]][v].sel(time=times), ref)
        da = to_vertical_grid(da, depth)
        fields[v] = da.transpose("time", "depth", "latitude", "longitude").values.astype(np.float32)
    ctx2d: Dict[str, np.ndarray] = {}
    for v in sorted(raw2d):
        da = to_horizontal_grid(dsets[v2g[v]][v].sel(time=times), ref)
        if "depth" in da.dims:
            da = da.isel(depth=0)
        ctx2d[v] = da.transpose("time", "latitude", "longitude").values.astype(np.float32)
    for d in dsets.values():
        d.close()

    # SSH -> anomaly from the basin mean of each month, averaged over the BGC ocean points only (the same
    # domain as the profiles). Keeps the spatial doming signal; removes the basin-wide seasonal cycle and the
    # sea-level trend, which only let the model tell years apart.
    if cfg.ssh_anomaly and "zos" in ctx2d:
        ocean2d = np.isfinite(fields[cfg.target_var][:, 0])  # (T, Y, X): surface O2 present
        zos = ctx2d["zos"]
        n_ok = (np.isfinite(zos) & ocean2d).sum(axis=(1, 2), keepdims=True)
        total = np.where(ocean2d & np.isfinite(zos), zos, 0.0).sum(axis=(1, 2), keepdims=True, dtype=np.float64)
        basin_mean = np.where(n_ok > 0, total / np.maximum(n_ok, 1), np.nan)
        ctx2d["zos"] = (zos - basin_mean).astype(np.float32)

    if "sigma0" in cfg.profile_features:
        fields["sigma0"] = compute_sigma0(fields["thetao"], fields["so"])
    for v in cfg.log_transform_vars:
        if v in fields:
            fields[v] = np.log10(np.clip(fields[v], 0.0, None) + cfg.log_eps).astype(np.float32)

    feats = list(cfg.profile_features)
    nt, _, ny, nx = fields[feats[0]].shape
    arr = np.stack([fields[f] for f in feats], axis=-1)  # (T, Z, Y, X, F)
    del fields
    arr = arr.transpose(0, 2, 3, 1, 4).reshape(nt * ny * nx, nz, len(feats))

    # 1) drop land / outside the BGC domain
    ti = feats.index(cfg.target_var)
    ocean = np.isfinite(arr[:, 0, ti])
    flat = np.nonzero(ocean)[0]
    arr = arr[ocean]
    # 2) bridge the seafloor mismatch of the two vertical grids (physics only), 3) fill short interior gaps
    phys = [i for i, f in enumerate(feats) if f in PHYSICAL_FEATURES]
    if phys and cfg.max_bottom_extend_levels > 0:
        arr[:, :, phys] = _chunked(arr[:, :, phys], lambda a: ffill_levels(a, cfg.max_bottom_extend_levels))
    arr = _chunked(arr, lambda a: fill_interior_gaps(a, depth, cfg.max_interior_gap_levels))
    # 4) truncate at the first level where any feature is missing
    valid = np.isfinite(arr).all(axis=2)
    valid_len = np.where(valid.all(axis=1), nz, np.argmin(valid, axis=1)).astype(np.int32)
    keep = valid_len >= cfg.min_valid_levels
    arr, valid_len, flat = arr[keep], valid_len[keep], flat[keep]
    arr[np.arange(nz)[None, :] >= valid_len[:, None]] = 0.0

    # context: position, season, bathymetry proxy, 2-D physics
    t_idx, y_idx, x_idx = np.unravel_index(flat, (nt, ny, nx))
    lat, lon = ref["lat"][y_idx], ref["lon"][x_idx]
    if time_unit(cfg) == "D":  # season from the day of the year (columns keep their names month_sin/cos)
        day0 = (stamps[t_idx] - stamps[t_idx].astype("datetime64[Y]")).astype(np.int64)  # 0 = 1 January
        angle = 2.0 * np.pi * day0 / 365.25
    else:
        month0 = stamps[t_idx].astype(np.int64) % 12  # 0 = January
        angle = 2.0 * np.pi * month0 / 12.0
    bottom = depth[valid_len - 1]
    cols = [lat, lon, np.sin(angle), np.cos(angle), np.log1p(bottom)]
    cols += [ctx2d[v][t_idx, y_idx, x_idx] for v in cfg.context_vars_2d]
    context = np.stack(cols, axis=-1).astype(np.float32)

    ydir.mkdir(parents=True, exist_ok=True)
    np.save(ydir / "profiles.npy", arr.astype(np.float32))
    np.save(ydir / "valid_len.npy", valid_len.astype(np.int16))
    np.save(ydir / "context.npy", context)
    np.savez(ydir / "meta.npz", time=stamps[t_idx].astype(np.int64), time_unit=time_unit(cfg), lat=lat.astype(np.float32),
             lon=lon.astype(np.float32))
    (ydir / "_DONE").touch()
    log.info("%d: %d time steps, %d profiles (median %d levels)", year, len(stamps), len(valid_len),
             int(np.median(valid_len)) if len(valid_len) else 0)
    return len(valid_len)


# --------------------------------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1, help="years processed in parallel (RAM ~5-10 GB each)")
    parser.add_argument("--years", type=int, nargs="*", default=None, help="restrict to these years")
    parser.add_argument("--force", action="store_true", help="overwrite grid.json if the configuration changed")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(Config(), args.set)
    raw_dir, out_dir = Path(raw_root(cfg)), Path(cfg.processed_dir)
    record = raw_dir / "download_weekdays.json"
    if cfg.temporal_resolution == "daily" and record.exists():
        have = set(json.loads(record.read_text())["kept_weekdays"])
        want = set(kept_weekdays(cfg))
        if not want <= have:
            raise SystemExit(f"days_per_week={cfg.days_per_week} (week_day_offset={cfg.week_day_offset}) needs weekday "
                             f"codes {sorted(want)}, but {raw_dir} only holds {sorted(have)}. Choose days_per_week "
                             "and week_day_offset so the codes are among the downloaded ones.")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw3d, raw2d = required_variables(cfg)
    v2g = variable_to_group()
    groups = sorted({v2g[v] for v in raw3d | raw2d} | {REFERENCE_GROUP})
    years = sorted(set.intersection(*[available_years(raw_dir, g) for g in groups]))
    if args.years:
        years = [y for y in years if y in set(args.years)]
    if not years:
        raise SystemExit(f"No year has raw files for all of {groups} in {raw_dir}")
    log.info("Years with complete raw data: %d-%d (%d)", years[0], years[-1], len(years))

    ref = reference_grid(cfg, raw_dir, years[0])
    grid = {
        "depth": ref["depth"].tolist(),
        "features": list(cfg.profile_features),
        "context": CONTEXT_BASE + list(cfg.context_vars_2d),
        # Recorded so that a folder processed with absolute SSH is refused instead of silently reused.
        "ssh_anomaly": bool(cfg.ssh_anomaly and "zos" in cfg.context_vars_2d),
        **({"temporal_resolution": "daily", "days_per_week": int(cfg.days_per_week),
            "kept_weekdays": kept_weekdays(cfg)} if cfg.temporal_resolution == "daily" else {}),
        "target": cfg.target_var,
        "target_index": list(cfg.profile_features).index(cfg.target_var),
        "log_transform_vars": [v for v in cfg.log_transform_vars if v in cfg.profile_features],
        "spatial_stride": cfg.spatial_stride,
        "max_depth_m": cfg.max_depth_m,
        "lat": ref["lat"].tolist(),
        "lon": ref["lon"].tolist(),
    }
    grid_path = out_dir / "grid.json"
    if grid_path.exists():
        old = {k: v for k, v in json.loads(grid_path.read_text()).items() if k != "years"}
        if old != grid and not args.force:
            raise SystemExit(f"{grid_path} was built with a different configuration. "
                             "Use a new processed_dir (recommended) or --force.")
    log.info("Common grid: %d levels (%.1f-%.0f m), %d x %d points, features %s",
             len(grid["depth"]), grid["depth"][0], grid["depth"][-1], len(grid["lat"]), len(grid["lon"]),
             grid["features"])

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_year, cfg, y, raw_dir, out_dir, ref): y for y in years}
            for fut in as_completed(futures):
                fut.result()
    else:
        for y in years:
            process_year(cfg, y, raw_dir, out_dir, ref)

    done = sorted(int(p.name[5:]) for p in out_dir.glob("year_*") if (p / "_DONE").exists())
    grid["years"] = done
    grid_path.write_text(json.dumps(grid))
    log.info("Done. Processed years: %s", done)


if __name__ == "__main__":
    main()