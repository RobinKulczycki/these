#!/usr/bin/env python
"""
Download the Black Sea reanalyses (BGC + physics) from Copernicus Marine, one NetCDF per (group, stream, year),
or per month with --daily. Credentials: run `copernicusmarine login` once; run on a node with internet access.

    python data_download.py --list | --dry-run
    python data_download.py [--groups bgc_bio phy_tem] [--start-year 2000 --end-year 2005]
    python data_download.py --daily [--days-per-week 2]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DATASET_GROUPS, Config, DatasetGroup, apply_overrides, kept_weekdays, raw_root

log = logging.getLogger("download")


def _cm():
    try:
        import copernicusmarine
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the official client first:  pip install 'copernicusmarine>=2.0'") from exc
    return copernicusmarine


# --------------------------------------------------------------------------------------- catalogue helpers
def _product_datasets(product_id: str) -> list:
    catalogue = _cm().describe(product_id=product_id)
    return [ds for product in catalogue.products for ds in product.datasets]


def _dataset_variables(dataset) -> List[str]:
    names = set()
    try:
        for version in dataset.versions:
            for part in version.parts:
                for service in part.services:
                    for var in service.variables:
                        names.add(var.short_name)
    except AttributeError:  # catalogue schema changed; listing is only informative
        pass
    return sorted(names)


def resolve_datasets(groups: Dict[str, DatasetGroup], include_interim: bool,
                     temporal_resolution: str = "monthly") -> Dict[str, List[Tuple[str, object]]]:
    """Map each group to [(stream, catalogue dataset), ...] with "my" first, then "myint"."""
    cache: Dict[str, list] = {}
    resolved = {}
    for name, group in groups.items():
        if group.product_id not in cache:
            cache[group.product_id] = _product_datasets(group.product_id)
        per_stream = {}
        for ds in cache[group.product_id]:
            match = re.match(group.regex(temporal_resolution), ds.dataset_id)
            if not match:
                continue
            stream = match.group(1)
            if stream == "myint" and not include_interim:
                continue
            if stream in per_stream:
                log.warning("%s: several '%s' datasets match (%s, %s); keeping the first",
                            name, stream, per_stream[stream].dataset_id, ds.dataset_id)
                continue
            per_stream[stream] = ds
        if "my" not in per_stream:
            raise RuntimeError(
                f"No multi-year dataset matches group '{name}' ({group.regex(temporal_resolution)}) in {group.product_id}. "
                "Inspect `copernicusmarine describe -p <product_id> -r dataset_id` and fix config.DATASET_GROUPS."
            )
        resolved[name] = [(s, per_stream[s]) for s in ("my", "myint") if s in per_stream]
    return resolved


def time_coverage(dataset_id: str, variable: str) -> Tuple[np.datetime64, np.datetime64]:
    """First/last time stamp, read lazily from the ARCO store (no data is downloaded)."""
    ds = _cm().open_dataset(dataset_id=dataset_id, variables=[variable])
    times = ds["time"].values
    ds.close()
    return np.datetime64(times.min(), "s"), np.datetime64(times.max(), "s")


def _year(t: np.datetime64) -> int:
    return int(t.astype("datetime64[Y]").astype(int) + 1970)


def _periods(year: int, daily: bool) -> List[Tuple[str, np.datetime64, np.datetime64]]:
    """(file label, first second, last second) of each file to download for one year."""
    if not daily:
        return [(f"{year}", np.datetime64(f"{year}-01-01T00:00:00"), np.datetime64(f"{year}-12-31T23:59:59"))]
    out = []
    for m in range(1, 13):
        first = np.datetime64(f"{year}-{m:02d}", "M")
        out.append((f"{year}-{m:02d}", first.astype("datetime64[s]"), (first + 1).astype("datetime64[s]") - 1))
    return out


# --------------------------------------------------------------------------------------- download
def _compressing_engine() -> Optional[str]:
    """A NetCDF writer that can compress (netCDF4 or h5netcdf), or None (then files are written uncompressed)."""
    for engine, modules in (("netcdf4", ("netCDF4",)), ("h5netcdf", ("h5netcdf", "h5py"))):
        try:
            for module in modules:
                __import__(module)
            return engine
        except ImportError:
            continue
    log.warning("Neither netCDF4 nor h5netcdf is installed: daily files are written uncompressed")
    return None


def _keep_days(src: Path, dst: Path, weekdays: List[int]) -> None:
    """Write the chosen weekdays of a daily file to dst, compressed, then delete src. Weekdays are
    (days since 1970-01-01) % 7, as in preprocess.py, so both pick exactly the same days."""
    import xarray as xr

    tmp = src.with_name(src.stem + ".kept.nc")
    with xr.open_dataset(src) as ds:
        days = ds["time"].values.astype("datetime64[D]").astype(np.int64)
        sub = ds.isel(time=np.nonzero(np.isin(days % 7, weekdays))[0]).load()  # reads only the kept days
    for v in sub.variables:  # drop the source chunking/encoding; it may not fit the shorter time axis
        sub[v].encoding = {}
    engine = _compressing_engine()
    enc = {v: {"zlib": True, "complevel": 1} for v in sub.data_vars} if engine else None
    sub.to_netcdf(tmp, engine=engine, encoding=enc)
    os.replace(tmp, dst)  # atomic: a file under its final name is always complete
    src.unlink()


def download_file(cfg: Config, dataset_id: str, group: DatasetGroup, start: str, end: str,
                  out_path: Path, dry_run: bool, retries: int, keep_weekdays: Optional[List[int]] = None) -> float:
    """Download one file into <dir>/_incomplete and move it into place only when complete. Returns MB.
    With keep_weekdays (daily files), only those days are kept and the file is compressed."""
    cm = _cm()
    tmp_dir = out_path.parent / "_incomplete"
    if not dry_run:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / out_path.name
    kwargs = dict(
        dataset_id=dataset_id,
        variables=list(group.variables),
        minimum_longitude=cfg.lon_min,
        maximum_longitude=cfg.lon_max,
        minimum_latitude=cfg.lat_min,
        maximum_latitude=cfg.lat_max,
        start_datetime=start,
        end_datetime=end,
        coordinates_selection_method="inside",
        output_directory=str(tmp_dir),
        output_filename=out_path.name,
        dry_run=dry_run,
    )
    if group.is_3d:
        # A margin below max_depth_m so physics can still be interpolated onto the deepest BGC level.
        kwargs.update(minimum_depth=0.0,
                      maximum_depth=5000.0 if cfg.max_depth_m is None else float(cfg.max_depth_m) + 100.0)

    for attempt in range(1, retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            response = cm.subset(**kwargs)
            if dry_run:
                size = float(getattr(response, "file_size", 0.0) or 0.0)
                if keep_weekdays is not None:  # uncompressed estimate of the kept days only
                    size *= len(keep_weekdays) / 7.0
                log.info("[dry-run] %-45s ~%8.1f MB", out_path.name, size)
                return size
            produced = Path(getattr(response, "file_path", tmp_path))
            if keep_weekdays is not None:
                _keep_days(produced, out_path, keep_weekdays)
            else:
                os.replace(produced, out_path)
            size = out_path.stat().st_size / 1e6
            log.info("saved %-45s %8.1f MB", out_path.name, size)
            return size
        except Exception as exc:  # network hiccups, server-side timeouts, ...
            if attempt == retries:
                raise
            wait = 30 * attempt
            log.warning("%s failed (attempt %d/%d): %s -- retrying in %ds", out_path.name, attempt, retries, exc, wait)
            time.sleep(wait)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", default=None, help=f"subset of {list(DATASET_GROUPS)}")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--daily", action="store_true",
                        help="daily means into raw_dir_daily (same as --set temporal_resolution=daily)")
    parser.add_argument("--days-per-week", type=int, default=7,
                        help="daily only: keep this many evenly spread days of every week (default 7 = all)")
    parser.add_argument("--list", action="store_true", help="print resolved dataset IDs and variables, then exit")
    parser.add_argument("--dry-run", action="store_true", help="estimate file sizes without downloading")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)  # harmless "pool is full" noise
    logging.getLogger("copernicusmarine").propagate = False  # avoid every message being printed twice
    cfg = apply_overrides(Config(), args.set)
    if args.daily:
        cfg.temporal_resolution = "daily"
    daily = cfg.temporal_resolution == "daily"
    log.info("%s means -> %s", cfg.temporal_resolution.capitalize(), raw_root(cfg))

    unknown = set(args.groups or []) - set(DATASET_GROUPS)
    if unknown:
        raise SystemExit(f"Unknown groups: {sorted(unknown)}")
    groups = {k: g for k, g in DATASET_GROUPS.items() if not args.groups or k in args.groups}
    resolved = resolve_datasets(groups, cfg.include_interim, cfg.temporal_resolution)

    if args.list:
        for name, streams in resolved.items():
            for stream, ds in streams:
                print(f"{name:13s} {stream:6s} {ds.dataset_id}\n{'':21s}variables: {', '.join(_dataset_variables(ds)) or '?'}")
        return

    for name, streams in resolved.items():
        for _, ds in streams:
            available = set(_dataset_variables(ds))
            missing = set(groups[name].variables) - available
            if available and missing:
                raise SystemExit(f"{ds.dataset_id} has no variable(s) {sorted(missing)}; available: {sorted(available)}")

    raw_dir = Path(raw_root(cfg))
    keep = None
    if daily:
        keep = kept_weekdays(cfg, args.days_per_week)
        record = raw_dir / "download_weekdays.json"
        if record.exists() and json.loads(record.read_text())["kept_weekdays"] != keep and not args.overwrite:
            raise SystemExit(f"{raw_dir} holds days {json.loads(record.read_text())['kept_weekdays']} (weekday codes); "
                             f"--days-per-week {args.days_per_week} would keep {keep}. Use another raw_dir_daily, "
                             "the same --days-per-week, or --overwrite to download everything again.")
        if not args.dry_run:
            raw_dir.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"kept_weekdays": keep, "days_per_week": args.days_per_week,
                                          "week_day_offset": cfg.week_day_offset}))
        log.info("Daily files keep weekday codes %s ((days since 1970-01-01) %% 7), compressed", keep)
    total_mb = 0.0
    for name, streams in resolved.items():
        group = groups[name]
        out_dir = raw_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stream, ds in streams:
            t0, t1 = time_coverage(ds.dataset_id, group.variables[0])
            log.info("%s [%s] %s covers %s .. %s", name, stream, ds.dataset_id, t0, t1)
            for year in range(_year(t0), _year(t1) + 1):
                if (args.start_year and year < args.start_year) or (args.end_year and year > args.end_year):
                    continue
                # Monthly means: one file per year. Daily means: one file per month (a year would be ~30x larger).
                for period, start, end in _periods(year, daily):
                    start, end = max(start, t0), min(end, t1)
                    if start > end:  # outside this dataset's coverage
                        continue
                    out_path = out_dir / f"{name}_{stream}_{period}.nc"
                    if out_path.exists() and not (args.overwrite or args.dry_run):
                        continue
                    total_mb += download_file(cfg, ds.dataset_id, group, str(start), str(end),
                                              out_path, args.dry_run, args.retries, keep)
    for name in resolved:  # remove the staging folders once they are empty
        try:
            (raw_dir / name / "_incomplete").rmdir()
        except OSError:
            pass
    log.info("%s %.1f GB", "Estimated total:" if args.dry_run else "Downloaded this session:", total_mb / 1e3)


if __name__ == "__main__":
    main()