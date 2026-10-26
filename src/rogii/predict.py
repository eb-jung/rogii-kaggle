"""Load trusted local model artifacts and predict from test-only supplied inputs."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .experiment import predict_ensemble, smooth_predictions
from .features import INPUT_COLUMNS, aligned_submission, build_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Refusing to overwrite existing predictions")
    model = joblib.load(a.model)  # Only load artifacts you trust; joblib is executable.
    sample = pd.read_csv(a.data_dir / "sample_submission.csv")
    well_ids = sorted(set(sample.id.str.rsplit("_", n=1).str[0]))
    frames, ids, groups = [], [], []
    for wid in well_ids:
        # No train directory, horizontal labels, or formation columns are opened.
        h = pd.read_csv(a.data_dir / "test" / f"{wid}__horizontal_well.csv", usecols=INPUT_COLUMNS)
        tw = pd.read_csv(a.data_dir / "test" / f"{wid}__typewell.csv", usecols=["TVT", "GR"])
        frame, row_ids = build_features(h, tw, wid)
        frames.append(frame)
        ids.extend(row_ids)
        groups.extend([wid] * len(frame))
    x = pd.concat(frames, ignore_index=True)
    if x.columns.tolist() != model["feature_names"]:
        raise ValueError("Feature schema differs from training")
    pred = smooth_predictions(predict_ensemble(model["models"], x), np.array(groups))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    aligned_submission(sample, ids, pred).to_csv(a.output, index=False)
    print(f"Wrote {len(ids):,} predictions in sample order to {a.output}")


if __name__ == "__main__":
    main()
