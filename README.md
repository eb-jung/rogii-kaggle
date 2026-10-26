# Wellbore Geology Prediction — Leakage-Aware Ensemble Regression

Predict the missing true vertical thickness (TVT) trajectory beyond a horizontal
well's prediction-start point, using measured geometry, gamma-ray logs, a supplied
TVT prefix, and a reference typewell. Depths and RMSE are in **feet**.

This version recovers the valid suffix-prediction design from the local baseline
and consolidates selected FLE3N-style features into an auditable pipeline. It does
not use released-test target lookup, restricted formation surfaces, pretrained
public artifacts, or Ridge stacking.

## Data and inference setting

Source: [ROGII Wellbore Geology Prediction on Kaggle](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).
Obtain data through your own authorized competition access. Raw files and
credentials are not distributed here.

```text
DATA_DIR/
  train/<well>__horizontal_well.csv
  train/<well>__typewell.csv
  test/<well>__horizontal_well.csv
  test/<well>__typewell.csv
  sample_submission.csv
```

| Input | Availability and use |
|---|---|
| Horizontal `MD, X, Y, Z, GR` | Supplied for the whole trajectory; usable for offline inference |
| Horizontal `TVT_input` | Known contiguous prefix; missing suffix defines prediction rows |
| Reference typewell `TVT, GR` | Supplied throughout; valid reference, not horizontal target labels |
| Horizontal `TVT` | Supervised labels only; never read by feature calculations |
| Formation surfaces / `Geology` | Not used; unavailable in the same form in test |

Centered GR windows and full-trajectory geometry assume offline inference with
the complete supplied log. This is not a causal streaming/drilling system.

## Features and estimator

- Prefix TVT anchor, recent TVT/MD slope, and linear extrapolation.
- Geometry/GR differences from the last known point and suffix position.
- Per-well centered GR mean/std/range, gradient, z-score, and vertical increments.
- Typewell summaries and nearest-GR TVT proxy. Gamma-ray matches are ambiguous;
  this proxy is not a unique geological alignment or a DTW path.
- Three LightGBM models fit on grouped training folds, equally averaged at
  inference. Regression target: `TVT - last_known_tvt`.
- Fixed per-well 11-point cubic Savitzky-Golay smoothing. Raw and smoothed scores
  are both reported; no meaningful improvement is attributed to smoothing.

No CatBoost or Ridge estimator is used in this selected version. Historical
implementations are compared in the [research audit](docs/research_audit.md).

## Validation and leakage prevention

All 773 released training wells were checked. Three overlap with official test
wells and are excluded. No additional exact full-input duplicate wells were
found. The remaining **770 wells** are deterministically split (seed 204):

| Role | Wells | Training/evaluation rows |
|---|---:|---:|
| Training | 616 | 123,200 sampled suffix rows, up to 200 per well |
| Validation | 77 | 372,424 complete suffix rows |
| Internal final test | 77 | 365,592 complete suffix rows |

Three inner GroupKFold models use only the training role. Each model sees
two-thirds of those training wells. Hyperparameters, iteration count, row cap,
and smoothing were fixed before the fresh run. There is no early stopping,
hyperparameter search, or refit on outer held-out wells.

An explicit input whitelist prevents horizontal target/formation columns from
entering features. Valid prefix information is preserved. No cross-well learned
preprocessing or label-derived spatial priors are fitted before splits. Original
CSV indices are retained; submission IDs must match exactly one-to-one. Missing,
duplicate, unexpected, or nonfinite predictions raise errors.

## Verified results

Fresh local execution; full outer suffixes and identical rows across methods:

| Method | Validation row RMSE | Internal-test row RMSE | Internal-test mean well RMSE |
|---|---:|---:|---:|
| Training-sample mean | 608.441 | 649.525 | 570.618 |
| Last supplied TVT | 15.012 | 14.489 | 13.071 |
| Recent-prefix linear extrapolation | 88.975 | 80.004 | 57.264 |
| LightGBM fold average | 13.790 | 13.557 | 11.854 |
| LightGBM fold average + fixed SG11 | 13.789 | **13.557** | **11.854** |

The predeclared smoothed estimator reduces internal-test pooled RMSE by **6.43%**
relative to the strong last-known baseline. This is an observed difference on
this partition, not a significance claim. Linear continuation is poor over long
suffixes. Smoothing changes RMSE by only about 0.0003 ft.

[Metrics](reports/metrics.json), [per-well results](reports/test_well_metrics.csv),
[fold memberships](reports/folds.json), and the [run manifest](reports/run_manifest.json)
record exact counts, configuration, input/source hashes, and package versions.
Runtime was approximately 27 seconds on the available local CPU setup; this is
not a standardized speed benchmark.

**No official Kaggle score is claimed.** The old target-lookup submission matches
14,151 released train labels exactly. This is recorded in the
[overlap audit](reports/overlap_audit.json), not counted as predictive validation.

## Reproduce

Tested with Python 3.14.0 and pinned versions in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v

python -m rogii.experiment \
  --data-dir /absolute/path/to/data/raw \
  --output-dir artifacts/full_fixed_204 \
  --train-rows-per-well 200 --rounds 200 --seed 204
```

Use a fresh output directory; frozen results are never overwritten. The command
writes a model, manifest, fold log, aggregate/per-well metrics, and ordered
`submission.csv`. Model binaries and predictions are gitignored.

For a bounded smoke run use `--max-wells 20 --rounds 5` with a new output path.
Such a sorted subset run must not be presented as the full result.

Saved-model inference opens only `test/` inputs and sample IDs:

```bash
python -m rogii.predict \
  --model artifacts/full_fixed_204/model.joblib \
  --data-dir /absolute/path/to/data/raw \
  --output artifacts/reproduced_submission.csv
```

Only deserialize model files you trust: joblib loading executes code.

On macOS, LightGBM needs an OpenMP runtime. This existing environment initially
failed because `libomp.dylib` was missing from its loader path. The run used an
already-installed scikit-learn runtime through process-local `DYLD_LIBRARY_PATH`;
no original environment/package files changed. For a normal fresh setup, install
an appropriate runtime (commonly `brew install libomp`) if that loader error
occurs. See [validation notes](reports/validation.md) for the exact executed setup.

## Code map

```text
src/rogii/features.py     Input boundary, context/GR features, strict ID alignment
src/rogii/experiment.py   Data audit, disjoint roles, grouped models, baselines
src/rogii/predict.py      Saved-model inference using test inputs only
tests/test_integrity.py  Target mutation, context, splits, duplicates, determinism
docs/research_audit.md   Comparison, leakage findings, attribution, resume audit
reports/                 Verified metrics/manifests; no raw data or model binaries
archive/github_legacy/   Original tracked experiments; not validated entry points
```

## Limits and next steps

This is bounded CPU validation with sampled training rows, not a full-data
competition stack. Nearby wells may share geology, so well separation does not
establish geographic independence. Historical experiments informed the design;
this partition is a fresh computational holdout, not a pristine research dataset.
One internal-test partition does not establish robustness across fields or time.
No row-independent confidence intervals or profitability claims are made.

Pooled RMSE weights longer wells more; inspect per-well outcomes too. Prefix
slopes can extrapolate badly, and nearest-GR matching is non-identifiable. DTW,
particle filters, CatBoost, and Ridge have not been freshly validated here.
Future model selection should use validation only and reserve a new evaluation
dataset for new generalization claims after this final test has been inspected.

## Attribution

Recovered from local `notebooks/02_baseline.ipynb` and selected existing FLE3N-style
features. Historical notebooks explicitly credit public Kaggle work by
lightningv08, needless090, ravaghi, sarpal465, fleongg/FLE3N, and others.
See [source and contribution details](docs/research_audit.md#attribution).
Recovery, evaluation repair, tests, and packaging used Codex assistance.
Repository ownership/file possession does not prove sole authorship. No new
license is asserted over third-party work or competition data.
