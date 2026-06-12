#!/usr/bin/env python3
"""FLE3N-style ROGII training pipeline.

This script implements the model-training techniques used by the referenced
public FLE3N ROGII v5 notebook in a reusable, Kaggle-ready Python module:

* per-well horizontal/typewell loading;
* gamma-ray dynamic-time-warping (DTW) alignment to typewell TVT;
* propagated known ``TVT_input`` context and TVT-gradient features;
* typewell summary, nearest-GR, rolling-GR, z-score, and spatial features;
* GroupKFold-by-well LightGBM training with early stopping;
* fold-averaged test inference plus optional Savitzky-Golay smoothing.

Typical Kaggle usage:
    python fle3n_rogii_v5_train.py \
      --data-path /kaggle/input/rogii-wellbore-geology-prediction \
      --output-dir /kaggle/working \
      --submission-file submission.csv

Local smoke test:
    python fle3n_rogii_v5_train.py --data-path /path/to/data --output-dir /tmp/rogii --max-wells 2
"""
from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


DEFAULT_ROLL_WINDOWS = (5, 10, 20, 50, 100)
DEFAULT_EXCLUDE_COLUMNS = {
    "well_id",
    "row_index",
    "TVT",
    "TVT_input",
    "is_train",
    "Geology",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
}


class OptionalDependencyError(RuntimeError):
    """Raised when an optional modeling dependency is required but unavailable."""


def _optional_module(name: str):
    """Import an optional module without wrapping imports in try/except."""
    parts = name.split(".")
    probe = ""
    for part in parts:
        probe = part if not probe else f"{probe}.{part}"
        if importlib.util.find_spec(probe) is None:
            return None
    return importlib.import_module(name)


def _find_data(explicit: str | None = None) -> Path:
    """Find the ROGII competition directory containing train/test/sample files."""
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
        "Could not find ROGII data. Pass --data-path or set ROGII_DATA to the directory "
        "containing train/, test/, and sample_submission.csv."
    )


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error compatible with older sklearn versions."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_sample_submission(sample: pd.DataFrame) -> pd.DataFrame:
    """Parse Kaggle sample-submission ids into well id and horizontal row index."""
    out = sample.copy()
    parts = out["id"].astype(str).str.extract(r"^(?P<well_id>[^_]+)_(?P<row_index>\d+)$")
    if parts.isna().any().any():
        bad = out.loc[parts.isna().any(axis=1), "id"].head().tolist()
        raise ValueError(f"Could not parse sample id(s), examples: {bad}")
    out["well_id"] = parts["well_id"]
    out["row_index"] = parts["row_index"].astype(int)
    return out


def load_well(split_dir: Path, well_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one horizontal well and its matching typewell."""
    horizontal = pd.read_csv(split_dir / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(split_dir / f"{well_id}__typewell.csv")
    horizontal["well_id"] = well_id
    typewell["well_id"] = well_id
    horizontal["row_index"] = np.arange(len(horizontal), dtype=np.int64)
    return horizontal, typewell


def load_all_wells(split_dir: Path, max_wells: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all wells from a split directory, optionally truncating for smoke tests."""
    horizontal_files = sorted(split_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        horizontal_files = horizontal_files[:max_wells]
    if not horizontal_files:
        raise FileNotFoundError(f"No horizontal-well CSV files found in {split_dir}")

    horizontal_frames: list[pd.DataFrame] = []
    typewell_frames: list[pd.DataFrame] = []
    for path in horizontal_files:
        well_id = path.name.replace("__horizontal_well.csv", "")
        horizontal, typewell = load_well(split_dir, well_id)
        horizontal_frames.append(horizontal)
        typewell_frames.append(typewell)
    return pd.concat(horizontal_frames, ignore_index=True), pd.concat(typewell_frames, ignore_index=True)


def _fill_numeric(values: pd.Series) -> np.ndarray:
    """Fill a numeric series with median/zero fallback and return float values."""
    numeric = pd.to_numeric(values, errors="coerce")
    fallback = numeric.median()
    if pd.isna(fallback):
        fallback = 0.0
    return numeric.fillna(float(fallback)).to_numpy(dtype=np.float64)


def dtw_align_tvt(horizontal_gr: np.ndarray, typewell_gr: np.ndarray, typewell_tvt: np.ndarray) -> np.ndarray:
    """Align horizontal GR to typewell GR with DTW and map matched indices to typewell TVT.

    If ``dtaidistance`` is unavailable, this returns NaNs so downstream models can
    still use the rest of the FLE3N-style feature set.
    """
    dtaidtw = _optional_module("dtaidistance.dtw")
    if dtaidtw is None or len(horizontal_gr) == 0 or len(typewell_gr) == 0:
        return np.full(len(horizontal_gr), np.nan, dtype=np.float64)

    horizontal_norm = (horizontal_gr - horizontal_gr.mean()) / (horizontal_gr.std() + 1e-8)
    typewell_norm = (typewell_gr - typewell_gr.mean()) / (typewell_gr.std() + 1e-8)
    path = dtaidtw.warping_path(horizontal_norm.astype(np.float64), typewell_norm.astype(np.float64))
    path_arr = np.asarray(path, dtype=np.int64)
    aligned = np.full(len(horizontal_gr), np.nan, dtype=np.float64)
    for i in range(len(horizontal_gr)):
        matched = path_arr[path_arr[:, 0] == i, 1]
        if len(matched):
            aligned[i] = typewell_tvt[min(int(np.median(matched)), len(typewell_tvt) - 1)]
    missing = np.isnan(aligned)
    if missing.all():
        return aligned
    if missing.any():
        idx = np.arange(len(aligned))
        aligned[missing] = np.interp(idx[missing], idx[~missing], aligned[~missing])
    return aligned


def _nearest_typewell_features(horizontal_gr: np.ndarray, typewell_gr: np.ndarray, typewell_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-GR and top-3-nearest-GR typewell TVT proxies."""
    if len(typewell_gr) == 0:
        nan_values = np.full(len(horizontal_gr), np.nan, dtype=np.float64)
        return nan_values, nan_values.copy()
    gr_diffs = np.abs(horizontal_gr[:, None] - typewell_gr[None, :])
    nearest_idx = np.argmin(gr_diffs, axis=1)
    k = min(3, gr_diffs.shape[1])
    topk_idx = np.argpartition(gr_diffs, kth=k - 1, axis=1)[:, :k]
    return typewell_tvt[nearest_idx], typewell_tvt[topk_idx].mean(axis=1)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the same-length Pearson correlation used as a well-level GR match feature."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a_norm = (a[:n] - a[:n].mean()) / (a[:n].std() + 1e-8)
    b_norm = (b[:n] - b[:n].mean()) / (b[:n].std() + 1e-8)
    corr = float(np.corrcoef(a_norm, b_norm)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _known_tvt_context(known: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build forward/backward/interpolated TVT_input context plus TVT gradient."""
    fwd = np.full(len(known), np.nan, dtype=np.float64)
    last = np.nan
    for i, value in enumerate(known):
        if not np.isnan(value):
            last = value
        fwd[i] = last

    bwd = np.full(len(known), np.nan, dtype=np.float64)
    nxt = np.nan
    for i in range(len(known) - 1, -1, -1):
        if not np.isnan(known[i]):
            nxt = known[i]
        bwd[i] = nxt

    interp = np.nanmean(np.vstack([fwd, bwd]), axis=0)
    interp = np.where(np.isnan(interp), 0.0, interp)

    known_mask = ~np.isnan(known)
    if known_mask.sum() > 1:
        known_idx = np.where(known_mask)[0]
        known_values = known[known_mask]
        grad_values = np.gradient(known_values, known_idx)
        grad = np.interp(np.arange(len(known)), known_idx, grad_values)
    else:
        grad = np.zeros(len(known), dtype=np.float64)
    return fwd, bwd, interp, grad


def build_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    roll_windows: Iterable[int] = DEFAULT_ROLL_WINDOWS,
) -> pd.DataFrame:
    """Build FLE3N-style per-row features from horizontal/typewell data."""
    records: list[pd.DataFrame] = []
    for well_id in horizontal_df["well_id"].drop_duplicates():
        horizontal = (
            horizontal_df.loc[horizontal_df["well_id"] == well_id]
            .sort_values("MD")
            .reset_index(drop=True)
            .copy()
        )
        typewell = typewell_df.loc[typewell_df["well_id"] == well_id].sort_values("TVT").copy()

        horizontal_gr = _fill_numeric(horizontal["GR"])
        typewell_gr = _fill_numeric(typewell["GR"])
        typewell_tvt = _fill_numeric(typewell["TVT"])

        horizontal["dtw_tvt_pred"] = dtw_align_tvt(horizontal_gr, typewell_gr, typewell_tvt)

        known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=np.float64)
        tvt_fwd, tvt_bwd, tvt_interp, tvt_grad = _known_tvt_context(known)
        horizontal["tvt_fwd"] = tvt_fwd
        horizontal["tvt_bwd"] = tvt_bwd
        horizontal["tvt_interp"] = tvt_interp
        horizontal["tvt_grad"] = tvt_grad

        horizontal["tw_gr_mean"] = float(np.mean(typewell_gr))
        horizontal["tw_gr_std"] = float(np.std(typewell_gr))
        horizontal["tw_tvt_mean"] = float(np.mean(typewell_tvt))
        horizontal["tw_tvt_range"] = float(np.max(typewell_tvt) - np.min(typewell_tvt))
        horizontal["tw_tvt_min"] = float(np.min(typewell_tvt))
        horizontal["tw_tvt_max"] = float(np.max(typewell_tvt))
        horizontal["gr_vs_tw"] = horizontal_gr - float(np.mean(typewell_gr))

        nearest_tvt, top3_tvt = _nearest_typewell_features(horizontal_gr, typewell_gr, typewell_tvt)
        horizontal["tw_nearest_tvt"] = nearest_tvt
        horizontal["tw_top3_tvt_mean"] = top3_tvt
        horizontal["xcorr"] = _safe_corr(horizontal_gr, typewell_gr)
        records.append(horizontal)

    features = pd.concat(records, ignore_index=True).sort_values(["well_id", "MD"]).reset_index(drop=True)

    gr_by_well = features.groupby("well_id")["GR"]
    for window in roll_windows:
        features[f"GR_mean_{window}"] = gr_by_well.transform(
            lambda values: values.rolling(window, min_periods=1, center=True).mean()
        )
        features[f"GR_std_{window}"] = gr_by_well.transform(
            lambda values: values.rolling(window, min_periods=1, center=True).std().fillna(0.0)
        )
        features[f"GR_range_{window}"] = gr_by_well.transform(
            lambda values: values.rolling(window, min_periods=1, center=True).max()
            - values.rolling(window, min_periods=1, center=True).min()
        )

    features["GR_grad"] = gr_by_well.transform(lambda values: np.gradient(values.fillna(values.median()).to_numpy()))
    features["GR_zscore"] = gr_by_well.transform(lambda values: (values - values.mean()) / (values.std() + 1e-8))
    features["md_from_heel"] = features.groupby("well_id")["MD"].transform(lambda values: values - values.min())
    features["md_rel"] = features.groupby("well_id")["MD"].transform(
        lambda values: (values - values.min()) / (values.max() - values.min() + 1e-8)
    )
    features["dZ"] = features.groupby("well_id")["Z"].transform(lambda values: values.diff().fillna(0.0))
    return features


def select_feature_columns(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> list[str]:
    """Select numeric columns shared by train and prediction rows, excluding leakage columns."""
    numeric_kinds = {"f", "i", "u", "b"}
    return [
        column
        for column in train_df.columns
        if column not in DEFAULT_EXCLUDE_COLUMNS
        and column in predict_df.columns
        and train_df[column].dtype.kind in numeric_kinds
        and predict_df[column].dtype.kind in numeric_kinds
    ]


def lgb_params(seed: int, n_estimators: int, learning_rate: float) -> dict[str, object]:
    """LightGBM parameters mirroring the FLE3N v5 training setup."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 255,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "n_estimators": n_estimators,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": seed,
    }


def train_lightgbm_groupkfold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
) -> tuple[list[object], pd.DataFrame, pd.DataFrame]:
    """Train fold LightGBM models with GroupKFold by well."""
    lgb = _optional_module("lightgbm")
    if lgb is None:
        raise OptionalDependencyError("lightgbm is required for training. Install it or run this script on Kaggle.")

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(train_df["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    groups = train_df["well_id"].astype(str).to_numpy()
    if np.isnan(y).any():
        raise ValueError("Training target TVT contains missing values on rows with known TVT_input")

    unique_groups = np.unique(groups)
    actual_folds = min(int(n_folds), len(unique_groups))
    if actual_folds < 2:
        raise ValueError("Need at least two wells for GroupKFold training")

    models: list[object] = []
    fold_rows: list[dict[str, float | int]] = []
    oof = np.full(len(train_df), np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=actual_folds)
    params = lgb_params(seed, n_estimators, learning_rate)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        val_pred = model.predict(X[val_idx])
        oof[val_idx] = val_pred
        score = rmse(y[val_idx], val_pred)
        fold_rows.append(
            {
                "fold": fold,
                "rmse": score,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(val_idx)),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            }
        )
        models.append(model)
        print(f"fold {fold}/{actual_folds}: rmse={score:.5f} n_valid={len(val_idx):,}")

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores.loc[len(fold_scores)] = {
        "fold": 0,
        "rmse": rmse(y, oof),
        "n_train": int(len(train_df)),
        "n_valid": int(len(train_df)),
        "best_iteration": 0,
    }

    importances = np.mean([model.feature_importances_ for model in models], axis=0)
    feature_importance = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    return models, fold_scores, feature_importance


def predict_fold_average(models: list[object], predict_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Average predictions from all fold models."""
    X_test = predict_df[feature_cols].to_numpy(dtype=np.float32)
    return np.mean([model.predict(X_test) for model in models], axis=0)


def smooth_per_well(df: pd.DataFrame, value_col: str, window: int, polyorder: int) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to each well's prediction sequence when available."""
    scipy_signal = _optional_module("scipy.signal")
    out = df.copy()
    out["smoothing_applied"] = False
    if scipy_signal is None or window <= polyorder + 1:
        return out

    for _, group in out.groupby("well_id", sort=False):
        idx = group.index.to_numpy()
        values = group[value_col].to_numpy(dtype=np.float64)
        win = min(int(window), len(values))
        if win % 2 == 0:
            win -= 1
        if win > polyorder:
            out.loc[idx, value_col] = scipy_signal.savgol_filter(values, window_length=win, polyorder=int(polyorder))
            out.loc[idx, "smoothing_applied"] = True
    return out


def build_submission(sample: pd.DataFrame, predictions: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Merge predictions into sample-submission order and validate completeness."""
    pred_cols = ["id", "TVT_pred"]
    submission = sample[["id"]].merge(predictions[pred_cols].rename(columns={"TVT_pred": target_col}), on="id", how="left")
    if submission[target_col].isna().any():
        missing = int(submission[target_col].isna().sum())
        fill_value = float(predictions["TVT_pred"].median())
        print(f"warning: filling {missing} unmatched sample rows with median prediction {fill_value:.5f}")
        submission[target_col] = submission[target_col].fillna(fill_value)
    if not np.isfinite(submission[target_col].to_numpy(dtype=float)).all():
        raise AssertionError("submission contains non-finite predictions")
    if submission["id"].tolist() != sample["id"].astype(str).tolist():
        raise AssertionError("submission IDs/order do not match sample_submission")
    return submission


def run(args: argparse.Namespace) -> None:
    """Run the full FLE3N-style training and inference pipeline."""
    start = time.time()
    set_seed(args.seed)
    data_dir = _find_data(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"data_dir={data_dir}")
    print("loading wells...")
    train_hz, train_tw = load_all_wells(data_dir / "train", max_wells=args.max_wells)
    test_hz, test_tw = load_all_wells(data_dir / "test", max_wells=args.max_wells)
    print(f"train wells={train_hz['well_id'].nunique()} rows={len(train_hz):,}")
    print(f"test wells={test_hz['well_id'].nunique()} rows={len(test_hz):,}")

    print("building train features...")
    train_features = build_features(train_hz, train_tw, roll_windows=args.roll_windows)
    print("building test features...")
    test_features = build_features(test_hz, test_tw, roll_windows=args.roll_windows)

    train_rows = train_features.loc[train_features["TVT_input"].notna()].copy()
    predict_rows = test_features.loc[test_features["TVT_input"].isna()].copy()
    if args.max_wells is not None:
        keep_wells = set(test_hz["well_id"].drop_duplicates())
        predict_rows = predict_rows.loc[predict_rows["well_id"].isin(keep_wells)].copy()
    if train_rows.empty or predict_rows.empty:
        raise ValueError("Need non-empty training rows and prediction rows")

    feature_cols = select_feature_columns(train_rows, predict_rows)
    if not feature_cols:
        raise ValueError("No usable shared numeric feature columns were selected")
    train_rows.loc[:, feature_cols] = train_rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    predict_rows.loc[:, feature_cols] = predict_rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"selected {len(feature_cols)} features")

    if "dtw_tvt_pred" in feature_cols and train_rows["dtw_tvt_pred"].notna().any():
        mask = train_rows["dtw_tvt_pred"].notna()
        print(f"DTW-only train RMSE={rmse(train_rows.loc[mask, 'TVT'].to_numpy(), train_rows.loc[mask, 'dtw_tvt_pred'].to_numpy()):.5f}")

    models, fold_scores, feature_importance = train_lightgbm_groupkfold(
        train_rows,
        feature_cols,
        n_folds=args.n_folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print("fold scores:")
    print(fold_scores.to_string(index=False))
    print("top feature importances:")
    print(feature_importance.head(25).to_string(index=False))

    predict_rows = predict_rows.copy()
    predict_rows["TVT_pred_raw"] = predict_fold_average(models, predict_rows, feature_cols)
    predict_rows["TVT_pred"] = predict_rows["TVT_pred_raw"]
    if not args.no_smooth:
        smoothed = smooth_per_well(predict_rows, "TVT_pred", args.smooth_window, args.smooth_poly)
        predict_rows["TVT_pred"] = smoothed["TVT_pred"]
        predict_rows["smoothing_applied"] = smoothed["smoothing_applied"]
    else:
        predict_rows["smoothing_applied"] = False
    predict_rows["id"] = predict_rows["well_id"].astype(str) + "_" + predict_rows["row_index"].astype(int).astype(str)

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if args.max_wells is not None:
        parsed_sample = parse_sample_submission(sample)
        sample = sample.loc[parsed_sample["well_id"].isin(set(test_hz["well_id"].drop_duplicates()))].reset_index(drop=True)
    target_col = next((column for column in sample.columns if column != "id"), "tvt")
    submission = build_submission(sample, predict_rows, target_col)

    submission_path = output_dir / args.submission_file
    legacy_submission_path = output_dir / "submission_fle3n_lgbm.csv"
    predictions_path = output_dir / "fle3n_predictions.csv"
    scores_path = output_dir / "fle3n_fold_scores.csv"
    importance_path = output_dir / "fle3n_feature_importance.csv"
    features_path = output_dir / "fle3n_feature_columns.txt"

    submission.to_csv(submission_path, index=False)
    if legacy_submission_path != submission_path:
        submission.to_csv(legacy_submission_path, index=False)
    predict_rows.to_csv(predictions_path, index=False)
    fold_scores.to_csv(scores_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    features_path.write_text("\n".join(feature_cols) + "\n")

    print("checks: submission IDs match sample_submission: OK")
    print("checks: no missing/non-finite predictions: OK")
    print(f"wrote {submission_path}")
    if legacy_submission_path != submission_path:
        print(f"wrote {legacy_submission_path}")
    print(f"wrote {predictions_path}")
    print(f"wrote {scores_path}")
    print(f"wrote {importance_path}")
    print(f"runtime: {time.time() - start:.2f}s")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a FLE3N-style LightGBM model for ROGII TVT prediction")
    parser.add_argument("--data-path", "--data", dest="data_path", default=None, help="ROGII data directory")
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-folds", type=int, default=5, help="GroupKFold folds by well")
    parser.add_argument("--n-estimators", type=int, default=5000, help="LightGBM boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="LightGBM early stopping rounds")
    parser.add_argument("--roll-windows", type=int, nargs="+", default=list(DEFAULT_ROLL_WINDOWS), help="GR rolling windows")
    parser.add_argument("--smooth-window", type=int, default=11, help="Savitzky-Golay smoothing window cap")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay polynomial order")
    parser.add_argument("--no-smooth", action="store_true", help="Disable per-well prediction smoothing")
    parser.add_argument("--max-wells", type=int, default=None, help="Smoke-test mode: first N train/test wells")
    parser.add_argument(
        "--submission-file",
        default="submission.csv",
        help="Submission filename to write inside --output-dir; Kaggle notebook-version submits expect submission.csv",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
