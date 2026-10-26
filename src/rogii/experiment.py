"""Fixed grouped experiment; no official-test labels or learned spatial priors."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

from .features import INPUT_COLUMNS, aligned_submission, build_features, target_mask


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_fingerprint(h):
    values = pd.util.hash_pandas_object(h[INPUT_COLUMNS], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def split_wells(wells, seed=204):
    ordered = np.array(sorted(wells), dtype=str)
    if len(ordered) < 10:
        raise ValueError("Need at least ten eligible wells for a three-role experiment")
    ordered = np.random.default_rng(seed).permutation(ordered)
    n = max(1, len(ordered) // 10)
    result = {"test": sorted(ordered[:n].tolist()),
              "validation": sorted(ordered[n:2*n].tolist()),
              "train": sorted(ordered[2*n:].tolist())}
    groups = [set(v) for v in result.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Split overlap")
    return result


def audit_data(data):
    """Check all available wells before splitting; only metadata uses official test."""
    data = Path(data)
    official = sorted(p.name.split("__")[0] for p in (data / "test").glob("*__horizontal_well.csv"))
    if not official:
        raise ValueError("No official test wells found")
    official_fingerprints = set()
    inputs = {}
    for wid in official:
        p = data / "test" / f"{wid}__horizontal_well.csv"
        h = pd.read_csv(p, usecols=INPUT_COLUMNS)
        target_mask(h)
        official_fingerprints.add(input_fingerprint(h))
        inputs[f"test/{p.name}"] = digest(p)
    eligible, seen, duplicates, excluded, records = [], set(), [], [], []
    train_paths = sorted((data / "train").glob("*__horizontal_well.csv"))
    for p in train_paths:
        wid = p.name.split("__")[0]
        h = pd.read_csv(p, usecols=INPUT_COLUMNS + ["TVT"])
        mask = target_mask(h)
        if not h.MD.is_monotonic_increasing or h.MD.duplicated().any():
            raise ValueError(f"Invalid/duplicate MD in {wid}")
        if not np.isfinite(h.TVT).all():
            raise ValueError(f"Missing/nonfinite target in {wid}")
        if not np.allclose(h.loc[~mask, "TVT_input"], h.loc[~mask, "TVT"], atol=1e-6, rtol=0):
            raise ValueError(f"Known prefix TVT mismatch in {wid}")
        fp = input_fingerprint(h)
        if wid in official or fp in official_fingerprints:
            excluded.append(wid)
        elif fp in seen:
            duplicates.append(wid)
        else:
            eligible.append(wid)
            seen.add(fp)
        records.append({"well": wid, "rows": len(h), "target_rows": int(mask.sum()),
                        "input_fingerprint": fp})
        inputs[f"train/{p.name}"] = digest(p)
    # Hash reference files and sample order too; no credentials or archives are read.
    for split in ("train", "test"):
        for p in sorted((data / split).glob("*__typewell.csv")):
            inputs[f"{split}/{p.name}"] = digest(p)
    inputs["sample_submission.csv"] = digest(data / "sample_submission.csv")
    return {"official_test_wells": official, "eligible_wells": eligible,
            "excluded_overlap_wells": excluded, "duplicate_input_wells": duplicates,
            "train_well_count": len(train_paths), "well_records": records,
            "input_sha256": inputs}


def load_role(data, wells, split, cap=0):
    xs, ys, groups, ids = [], [], [], []
    for wid in wells:
        h = pd.read_csv(Path(data) / split / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(Path(data) / split / f"{wid}__typewell.csv", usecols=["TVT", "GR"])
        x, row_ids = build_features(h, tw, wid)
        take = (np.unique(np.linspace(0, len(x)-1, cap, dtype=int))
                if cap and len(x) > cap else np.arange(len(x)))
        xs.append(x.iloc[take])
        ids.extend(np.array(row_ids)[take].tolist())
        groups.extend([wid] * len(take))
        if split == "train":
            # Labels are extracted separately after feature construction.
            ys.extend(h.loc[h.TVT_input.isna(), "TVT"].to_numpy()[take].tolist())
    return (pd.concat(xs, ignore_index=True), np.asarray(ys),
            np.asarray(groups), ids)


def fit_ensemble(x, y, groups, seed=204, rounds=200):
    """Three fold models; equal average, not Ridge stacking. Fixed iteration count."""
    models, fold_log, oof = [], [], np.full(len(y), np.nan)
    residual = y - x.last_known_tvt.to_numpy()
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=3).split(x, y, groups)):
        train_wells = set(groups[tr]); valid_wells = set(groups[va])
        if train_wells & valid_wells:
            raise ValueError("Inner fold overlap")
        model = LGBMRegressor(n_estimators=rounds, learning_rate=0.05,
                              num_leaves=31, max_depth=-1, min_child_samples=40,
                              reg_lambda=1.0, n_jobs=4, random_state=seed + fold,
                              deterministic=True, force_col_wise=True, verbosity=-1)
        model.fit(x.iloc[tr], residual[tr])
        oof[va] = model.predict(x.iloc[va]) + x.last_known_tvt.to_numpy()[va]
        models.append(model)
        fold_log.append({"fold": fold, "train_wells": sorted(train_wells),
                         "validation_wells": sorted(valid_wells),
                         "rmse": float(np.sqrt(np.mean((y[va] - oof[va])**2)))})
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete out-of-fold predictions")
    return models, fold_log, oof


def predict_ensemble(models, x):
    return np.mean([m.predict(x) for m in models], axis=0) + x.last_known_tvt.to_numpy()


def smooth_predictions(values, groups):
    result = np.asarray(values, dtype=float).copy()
    for wid in np.unique(groups):
        idx = np.flatnonzero(groups == wid)
        window = min(11, len(idx) if len(idx) % 2 else len(idx)-1)
        if window >= 5:
            result[idx] = savgol_filter(result[idx], window, 3)
    return result


def metrics(y, pred, groups):
    sq = (np.asarray(y) - pred)**2
    per = [float(np.sqrt(sq[groups == g].mean())) for g in np.unique(groups)]
    return {"row_rmse_ft": float(np.sqrt(sq.mean())),
            "mean_well_rmse_ft": float(np.mean(per)),
            "median_well_rmse_ft": float(np.median(per)),
            "rows": len(y), "wells": len(per)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-wells", type=int, default=0, help="0: all eligible wells; >=10: smoke subset")
    p.add_argument("--train-rows-per-well", type=int, default=200, help="Deterministic coverage of each training suffix; 0: all")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=204)
    a = p.parse_args()
    if a.output_dir.exists():
        raise ValueError("Use a fresh output directory to preserve frozen results")
    if a.rounds < 1 or a.train_rows_per_well < 0 or (a.max_wells and a.max_wells < 10):
        p.error("Require positive rounds, nonnegative row cap, max-wells 0 or >=10")
    a.output_dir.mkdir(parents=True)
    start = time.monotonic()
    audit = audit_data(a.data_dir)
    wells = audit["eligible_wells"][:a.max_wells or None]
    split = split_wells(wells, a.seed)
    manifest = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "split": split, "audit": audit,
                "protocol": "Fixed config; 80/10/10 wells; 3 inner folds; full outer suffixes; no tuning",
                "environment": {"python": platform.python_version(), **{
                    k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "joblib")}},
                "source_sha256": {f.name: digest(f) for f in Path(__file__).parent.glob("*.py")}}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    x, y, g, ids = load_role(a.data_dir, split["train"], "train", a.train_rows_per_well)
    print(f"Training: {len(y):,} rows, {len(set(g))} wells, {x.shape[1]} features", flush=True)
    models, folds, oof = fit_ensemble(x, y, g, a.seed, a.rounds)
    joblib.dump({"models": models, "feature_names": x.columns.tolist(), "config": manifest["config"]}, a.output_dir / "model.joblib")
    (a.output_dir / "folds.json").write_text(json.dumps(folds, indent=2))
    scores = {"inner_oof_training_sample": metrics(y, oof, g)}
    train_mean = float(np.mean(y))
    for role in ("validation", "test"):
        xx, yy, gg, row_ids = load_role(a.data_dir, split[role], "train")
        raw = predict_ensemble(models, xx)
        pred = smooth_predictions(raw, gg)
        variants = {"training_mean": np.full(len(yy), train_mean),
                    "last_known": xx.last_known_tvt.to_numpy(),
                    "linear_extrapolation": xx.linear_tvt_baseline.to_numpy(),
                    "lightgbm_fold_average": raw, "lightgbm_fold_average_sg11": pred}
        scores[role] = {name: metrics(yy, values, gg) for name, values in variants.items()}
        # One row per well: enough to inspect heterogeneity without redistributing labels.
        per_well = [{"well": wid, "rows": int(np.sum(gg == wid)), **{
            name: float(np.sqrt(np.mean((yy[gg == wid] - values[gg == wid])**2)))
            for name, values in variants.items()}} for wid in np.unique(gg)]
        pd.DataFrame(per_well).to_csv(a.output_dir / f"{role}_well_metrics.csv", index=False)
        print(role, json.dumps(scores[role]), flush=True)
    tx, _, tg, tids = load_role(a.data_dir, audit["official_test_wells"], "test")
    sample = pd.read_csv(a.data_dir / "sample_submission.csv")
    submission = aligned_submission(sample, tids, smooth_predictions(predict_ensemble(models, tx), tg))
    submission.to_csv(a.output_dir / "submission.csv", index=False)
    scores["runtime_seconds"] = round(time.monotonic() - start, 2)
    scores["official_submission_rows"] = len(submission)
    scores["performance_scope"] = "Internal held-out wells; official-test labels never scored. Training rows are capped as recorded."
    (a.output_dir / "metrics.json").write_text(json.dumps(scores, indent=2))
    print(f"Finished in {scores['runtime_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
