#!/usr/bin/env python3
"""Standalone Pipeline B likelihood-weighted particle filter reproduction.

This script extracts only the Pipeline B likelihood-PF path from the public
`rogii-dual-pipeline-blend.ipynb` notebook. It does not train LightGBM/CatBoost
and does not use target lookup. It reads test horizontal/typewell files plus
`sample_submission.csv`, emits per-row likelihood-PF candidate columns, and
writes a `likpf_scale_5` candidate submission.

Typical Kaggle usage:
    python pipeline_b_likpf_only.py \
      --data-path /kaggle/input/rogii-wellbore-geology-prediction \
      --output-dir /kaggle/working

Local smoke test with fewer wells and fewer particles/seeds:
    python pipeline_b_likpf_only.py --data-path /path/to/data --max-wells 1 --seeds 4 --particles 64
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import importlib
import importlib.util

import numpy as np
import pandas as pd
from numba import njit

if importlib.util.find_spec("joblib") is not None:
    _joblib = importlib.import_module("joblib")
    Parallel = _joblib.Parallel
    delayed = _joblib.delayed
else:  # pragma: no cover - serial fallback for minimal environments
    Parallel = None
    delayed = None

if importlib.util.find_spec("scipy.signal") is not None:
    savgol_filter = importlib.import_module("scipy.signal").savgol_filter
else:  # pragma: no cover - smoothing is optional
    savgol_filter = None


DEFAULT_SCALES = (3.0, 5.0, 8.0, 12.0)
DEFAULT_PARTICLES = 500
DEFAULT_SEEDS = 128


@njit(cache=True)
def _interp1(grid: np.ndarray, v: float, vmin: float, step: float) -> float:
    """Linear interpolation into a regular typewell-GR grid."""
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True, nogil=True)
def _pf_lik_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gg: np.ndarray,
    vmin: float,
    step: float,
    gs: float,
    ls: float,
    ir: float,
    N: int,
    n_seeds: int,
    seed_base: int,
    MOM: float,
    VN: float,
    PN: float,
    RP: float,
    RR: float,
    RESAMP: float,
    init_spr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the likelihood-weighted PF for all seeds.

    Copied/adapted from Pipeline B's `_pf_lik_allseeds` numba kernel. Each seed
    produces one TVT path and one accumulated GR log-likelihood. Seed paths are
    combined later by `lik_pf` using soft likelihood weights at multiple scales.
    """
    n = len(md_v)
    preds = np.empty((n_seeds, n))
    liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step

    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N)
        rate = np.empty(N)
        w = np.ones(N) / N

        for j in range(N):
            pos[j] = ls + init_spr * np.random.randn()
            rate[j] = ir + 0.01 * np.random.randn()

        log_lik = 0.0
        prev_md = md_v[0] - 1.0

        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0

            for j in range(N):
                rate[j] = MOM * rate[j] + VN * np.random.randn()
                pos[j] += rate[j] * dm + PN * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.0:
                    tvt_j = vmin - 100.0
                if tvt_j > tmax + 100.0:
                    tvt_j = tmax + 100.0
                pos[j] = tvt_j + z_v[i]

            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                dd = d * d
                if dd > 600.0:
                    dd = 600.0
                lk = np.exp(-0.5 * dd)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += w[j] * lk
                w[j] = w[j] * lk

            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += np.log(avg_lk)

            ws = 0.0
            for j in range(N):
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N

            neff_inv = 0.0
            for j in range(N):
                neff_inv += w[j] * w[j]
            neff = 1.0 / neff_inv

            if neff < RESAMP * N:
                cum = np.empty(N)
                c = 0.0
                for j in range(N):
                    c += w[j]
                    cum[j] = c
                u0 = np.random.uniform(0.0, 1.0 / N)
                newpos = np.empty(N)
                newrate = np.empty(N)
                ci = 0
                for j in range(N):
                    u = u0 + j / N
                    while ci < N - 1 and cum[ci] < u:
                        ci += 1
                    newpos[j] = pos[ci] + RP * np.random.randn()
                    newrate[j] = rate[ci] + RR * np.random.randn()
                for j in range(N):
                    pos[j] = newpos[j]
                    rate[j] = newrate[j]
                    w[j] = 1.0 / N

            est = 0.0
            for j in range(N):
                est += w[j] * (pos[j] - z_v[i])
            preds[s, i] = est
            prev_md = md_v[i]

        liks[s] = log_lik

    return preds, liks


def _find_data(explicit: str | None = None) -> Path:
    """Find the ROGII competition data directory."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            os.environ.get("ROGII_DATA", ""),
            "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
            "/kaggle/input/rogii-wellbore-geology-prediction",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "train").exists() and (path / "test").exists() and (path / "sample_submission.csv").exists():
            return path
    for train_dir in glob.glob("/kaggle/input/**/train", recursive=True):
        path = Path(train_dir).parent
        if (path / "test").exists() and (path / "sample_submission.csv").exists():
            return path
    raise FileNotFoundError(
        "Could not find ROGII data. Pass --data or set ROGII_DATA to the directory "
        "containing train/, test/, and sample_submission.csv."
    )


def _grid(tw_tvt: np.ndarray, tw_gr: np.ndarray, step: float = 0.2) -> Tuple[np.ndarray, float, float]:
    """Create the regular typewell GR interpolation grid used by the PF kernel."""
    tmin = float(np.nanmin(tw_tvt))
    tmax = float(np.nanmax(tw_tvt))
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), tmin, float(step)


def parse_sample_submission(sample: pd.DataFrame) -> pd.DataFrame:
    """Parse `id` into `well` and zero-based horizontal-well `row_idx`."""
    required = {"id"}
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"sample_submission is missing columns: {sorted(missing)}")
    out = sample.copy()
    parts = out["id"].astype(str).str.extract(r"^(?P<well>[^_]+)_(?P<row_idx>\d+)$")
    if parts.isna().any().any():
        bad = out.loc[parts.isna().any(axis=1), "id"].head().tolist()
        raise ValueError(f"Could not parse sample id(s), examples: {bad}")
    out["well"] = parts["well"]
    out["row_idx"] = parts["row_idx"].astype(int)
    return out


def load_well(data_dir: Path, wid: str, split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one horizontal well and its typewell."""
    base = data_dir / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw


def lik_pf(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = DEFAULT_PARTICLES,
    n_seeds: int = DEFAULT_SEEDS,
    scales: Iterable[float] = DEFAULT_SCALES,
    init_spr: float = 4.5,
    seed_base: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, float | np.ndarray]]:
    """Run Pipeline B's likelihood-weighted PF for one well.

    Returns a mapping of `pf_scale_X`/`pf_mean` arrays for evaluation rows only,
    the original horizontal-well row indices for those arrays, and quality stats.
    """
    required_hw = {"TVT_input", "MD", "Z", "GR"}
    required_tw = {"TVT", "GR"}
    missing_hw = required_hw - set(hw.columns)
    missing_tw = required_tw - set(tw.columns)
    if missing_hw or missing_tw:
        raise ValueError(f"Missing columns: horizontal={sorted(missing_hw)}, typewell={sorted(missing_tw)}")

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr_series = tw_s["GR"].astype(float)
    tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(dtype=float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return {}, np.array([], dtype=int), {}
    if len(kn) == 0:
        raise ValueError("Cannot run likelihood-PF without at least one known TVT_input row")

    last = kn.iloc[-1]
    ls = float(last["TVT_input"]) + float(last["Z"])

    tw_at_k = np.interp(kn["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)
    kgr = kn["GR"].astype(float).fillna(0.0).to_numpy(dtype=float)
    gs = float(np.clip(np.nanstd(kgr - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(dtype=float))
    dz = np.diff(tail["Z"].to_numpy(dtype=float))
    dm = np.diff(tail["MD"].to_numpy(dtype=float))
    valid_dm = dm > 0
    ir = float(np.median((dt + dz)[valid_dm] / dm[valid_dm])) if valid_dm.sum() >= 3 else 0.0

    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_interp = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    gr_v = gr_interp.to_numpy(dtype=float)[ev.index]

    preds, liks = _pf_lik_allseeds(
        ev["MD"].to_numpy(dtype=float),
        ev["Z"].to_numpy(dtype=float),
        gr_v,
        gg,
        gmin,
        gst,
        gs,
        ls,
        ir,
        int(n_particles),
        int(n_seeds),
        int(seed_base),
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        init_spr,
    )

    ln = liks - liks.max()
    out: Dict[str, np.ndarray] = {}
    for scale in scales:
        weights = np.exp(ln / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * preds).sum(axis=0).astype(np.float32)
    out["pf_mean"] = preds.mean(axis=0).astype(np.float32)

    quality = {
        "pf_best_ll_per_row": float(liks.max()) / max(len(ev), 1),
        "pf_ll_spread": float(liks.std()),
        "pf_point_std_mean": float(preds.std(axis=0).mean()),
        "pf_gr_sig": gs,
        "n_eval": int(len(ev)),
        "n_known": int(len(kn)),
    }
    return out, ev.index.to_numpy(dtype=int), quality


def _well_likpf_rows(
    data_dir: Path,
    wid: str,
    split: str,
    n_particles: int,
    n_seeds: int,
    scales: Tuple[float, ...],
    seed_base: int,
) -> Tuple[pd.DataFrame, Dict[str, float | int | str]]:
    """Build likelihood-PF candidate rows for one well."""
    t0 = time.time()
    hw, tw = load_well(data_dir, wid, split)
    out, idx, quality = lik_pf(
        hw,
        tw,
        n_particles=n_particles,
        n_seeds=n_seeds,
        scales=scales,
        seed_base=seed_base,
    )
    if len(idx) == 0:
        return pd.DataFrame({"id": []}), {"well": wid, "seconds": time.time() - t0, "n_eval": 0}

    rows: Dict[str, object] = {"id": [f"{wid}_{int(i)}" for i in idx], "well": wid, "row_idx": idx}
    for src_name, values in out.items():
        dst_name = "likpf_" + src_name.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
        rows[dst_name] = values.astype(np.float32)
    elapsed = time.time() - t0
    stats: Dict[str, float | int | str] = {"well": wid, "seconds": elapsed, "n_eval": int(len(idx))}
    stats.update(quality)
    return pd.DataFrame(rows), stats


def build_likpf(
    data_dir: Path,
    well_ids: Iterable[str],
    split: str = "test",
    n_particles: int = DEFAULT_PARTICLES,
    n_seeds: int = DEFAULT_SEEDS,
    scales: Iterable[float] = DEFAULT_SCALES,
    n_jobs: int = 1,
    seed_base: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate likelihood-PF candidates for all requested wells."""
    well_list = list(dict.fromkeys(well_ids))
    scale_tuple = tuple(float(s) for s in scales)
    tasks = [
        (data_dir, wid, split, n_particles, n_seeds, scale_tuple, seed_base + 1000 * i)
        for i, wid in enumerate(well_list)
    ]
    if n_jobs != 1 and Parallel is not None:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_well_likpf_rows)(*task) for task in tasks
        )
    else:
        results = [_well_likpf_rows(*task) for task in tasks]
    frames = [frame for frame, _ in results if len(frame)]
    stats = pd.DataFrame([stat for _, stat in results])
    if not frames:
        return pd.DataFrame(), stats
    return pd.concat(frames, ignore_index=True), stats


def smooth_per_well(
    sample_pred: pd.DataFrame,
    value_col: str = "tvt",
    window: int = 61,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Apply simple per-well Savitzky-Golay smoothing if scipy is available."""
    out = sample_pred.copy()
    if savgol_filter is None:
        out["smoothing_applied"] = False
        return out
    out["smoothing_applied"] = False
    for _, group in out.groupby("well", sort=False):
        pos = group.index.to_numpy()
        values = group[value_col].to_numpy(dtype=float)
        n = len(values)
        win = min(int(window), n)
        if win % 2 == 0:
            win -= 1
        if win >= polyorder + 2:
            out.loc[pos, value_col] = savgol_filter(values, win, int(polyorder))
            out.loc[pos, "smoothing_applied"] = True
    return out


def _prediction_ranges(df: pd.DataFrame, pred_col: str = "tvt") -> pd.DataFrame:
    """Summarize prediction ranges per well for the required checks."""
    return (
        df.groupby("well", sort=False)[pred_col]
        .agg(rows="size", min="min", max="max", mean="mean", std="std")
        .reset_index()
    )


def run(args: argparse.Namespace) -> None:
    data_dir = _find_data(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sample_raw_full = pd.read_csv(data_dir / "sample_submission.csv")
    sample_full = parse_sample_submission(sample_raw_full)
    wells = sample_full["well"].drop_duplicates().tolist()
    if args.max_wells is not None:
        if args.max_wells <= 0:
            raise ValueError("--max-wells must be a positive integer")
        wells = wells[: args.max_wells]
        smoke_mask = sample_full["well"].isin(wells)
        sample = sample_full.loc[smoke_mask].reset_index(drop=True)
        sample_raw = sample_raw_full.loc[smoke_mask].reset_index(drop=True)
        print(f"SMOKE MODE: limiting run to first {len(wells)} sample well(s): {wells}")
    else:
        sample = sample_full
        sample_raw = sample_raw_full
    print(f"data_dir={data_dir}")
    print(f"sample rows={len(sample)} | wells={len(wells)} | particles={args.particles} | seeds={args.seeds}")

    candidates_eval, stats = build_likpf(
        data_dir=data_dir,
        well_ids=wells,
        split="test",
        n_particles=args.particles,
        n_seeds=args.seeds,
        scales=DEFAULT_SCALES,
        n_jobs=args.n_jobs,
        seed_base=args.seed_base,
    )
    if candidates_eval.empty:
        raise RuntimeError("No likelihood-PF candidates were generated")

    candidates = sample[["id", "well", "row_idx"]].merge(candidates_eval, on=["id", "well", "row_idx"], how="left")

    candidate_cols = ["likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12", "likpf_mean"]
    missing_cols = [c for c in candidate_cols if c not in candidates.columns]
    if missing_cols:
        raise RuntimeError(f"Missing expected candidate columns: {missing_cols}")

    # Defensive fallback only for IDs outside the missing-TVT evaluation rows. This uses only
    # test-time TVT_input when present, otherwise last known TVT_input; it never reads train TVT.
    fallback_counts = {c: int(candidates[c].isna().sum()) for c in candidate_cols}
    if any(fallback_counts.values()):
        print(f"candidate gaps before fallback: {fallback_counts}")
        for wid, idx in candidates.groupby("well").groups.items():
            hw, _ = load_well(data_dir, wid, "test")
            known = hw["TVT_input"].dropna()
            if len(known) == 0:
                raise RuntimeError(f"Missing candidates for {wid} and no TVT_input fallback is available")
            last_known = float(known.iloc[-1])
            for row_i in idx:
                ridx = int(candidates.at[row_i, "row_idx"])
                if 0 <= ridx < len(hw) and pd.notna(hw.at[ridx, "TVT_input"]):
                    fallback_value = float(hw.at[ridx, "TVT_input"])
                else:
                    fallback_value = last_known
                for col in candidate_cols:
                    if pd.isna(candidates.at[row_i, col]):
                        candidates.at[row_i, col] = fallback_value

    if candidates[candidate_cols].isna().any().any():
        missing = candidates[candidate_cols].isna().sum().to_dict()
        raise AssertionError(f"Missing predictions after fallback: {missing}")

    candidates.to_csv(out_dir / "likpf_candidates.csv", index=False)

    submission = sample[["id", "well", "row_idx"]].copy()
    submission["tvt"] = candidates["likpf_scale_5"].astype(float).to_numpy()
    if not args.no_smooth:
        submission = smooth_per_well(submission, "tvt", window=args.smooth_window, polyorder=args.smooth_poly)
    else:
        submission["smoothing_applied"] = False

    # Required checks.
    if submission["id"].tolist() != sample_raw["id"].astype(str).tolist():
        raise AssertionError("submission IDs/order do not match sample_submission")
    if submission["tvt"].isna().any():
        raise AssertionError("submission contains missing predictions")
    if not np.isfinite(submission["tvt"].to_numpy(dtype=float)).all():
        raise AssertionError("submission contains non-finite predictions")

    ranges = _prediction_ranges(submission, "tvt")
    ranges.to_csv(out_dir / "likpf_prediction_ranges.csv", index=False)
    stats.to_csv(out_dir / "likpf_runtime_stats.csv", index=False)

    final_submission = submission[["id", "tvt"]].copy()
    final_submission.to_csv(out_dir / "submission_likpf_scale5.csv", index=False)

    elapsed = time.time() - t0
    rows_per_second = len(candidates_eval) / max(float(stats["seconds"].sum()) if "seconds" in stats else elapsed, 1e-9)
    estimate = len(sample) / max(rows_per_second, 1e-9)

    print("checks: submission IDs match sample_submission: OK")
    print("checks: no missing/non-finite predictions: OK")
    print("per-well prediction ranges:")
    print(ranges.to_string(index=False))
    print("runtime stats:")
    print(stats.to_string(index=False))
    print(f"runtime: elapsed={elapsed:.2f}s | observed_rows_per_sec={rows_per_second:.2f}")
    print(f"runtime estimate for {len(sample)} rows at observed speed: {estimate:.2f}s")
    print(f"wrote {out_dir / 'likpf_candidates.csv'}")
    print(f"wrote {out_dir / 'submission_likpf_scale5.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Pipeline B likelihood-PF candidate generator")
    parser.add_argument(
        "--data-path",
        "--data",
        dest="data_path",
        default=None,
        help="ROGII data directory containing train/, test/, sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        default=".",
        help="Directory for submission_likpf_scale5.csv and likpf_candidates.csv",
    )
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES, help="Particles per PF seed")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of PF random seeds")
    parser.add_argument("--seed-base", type=int, default=0, help="Base random seed offset")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel wells; use 1 for deterministic/simple local runs")
    parser.add_argument("--max-wells", type=int, default=None, help="Smoke-test mode: run only the first N sample wells")
    parser.add_argument("--no-smooth", action="store_true", help="Disable per-well Savitzky-Golay smoothing")
    parser.add_argument("--smooth-window", type=int, default=61, help="Savitzky-Golay window length cap")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay polynomial order")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
