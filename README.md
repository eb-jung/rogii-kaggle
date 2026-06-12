# rogii-kaggle

## FLE3N-style LightGBM training

`fle3n_rogii_v5_train.py` is a Kaggle-ready training pipeline that mirrors the
main model-training techniques from the referenced public `fle3n-rogii-v5`
notebook:

- per-well horizontal/typewell loading with stable sample row ids;
- gamma-ray dynamic-time-warping (DTW) alignment to typewell TVT when
  `dtaidistance` is available;
- known-interval `TVT_input` context features (`tvt_fwd`, `tvt_bwd`,
  `tvt_interp`, `tvt_grad`);
- typewell summary features plus nearest-GR and top-3-nearest-GR TVT proxies;
- centered rolling GR statistics, GR gradients/z-scores, and spatial MD/Z
  features;
- LightGBM regression trained with `GroupKFold` by well, early stopping, and
  fold-averaged inference;
- optional per-well Savitzky-Golay smoothing before writing the submission.

Run the full training pipeline on Kaggle with:

```bash
python fle3n_rogii_v5_train.py \
  --data-path /kaggle/input/rogii-wellbore-geology-prediction \
  --output-dir /kaggle/working
```

If the competition data is mounted under Kaggle's `/kaggle/input/competitions`
prefix, the script will auto-detect it. The main Kaggle-ready output is
`submission.csv`; the legacy alias `submission_fle3n_lgbm.csv` and
diagnostics are also written as
`fle3n_predictions.csv`, `fle3n_fold_scores.csv`,
`fle3n_feature_importance.csv`, and `fle3n_feature_columns.txt`.

For a lightweight smoke test that only loads the first one or two train/test
wells, add `--max-wells` and reduce the number of boosting rounds:

```bash
python fle3n_rogii_v5_train.py \
  --data-path /kaggle/input/rogii-wellbore-geology-prediction \
  --output-dir /kaggle/working \
  --max-wells 2 \
  --n-estimators 200
```

### Notebook-format run and Kaggle submission

`fle3n_rogii_v5_train.ipynb` is the notebook-format version of the same
pipeline. Its final cell calls the training pipeline with Kaggle paths and
writes `/kaggle/working/submission.csv`, which is the filename expected by the
Kaggle notebook-output submission command.

After the notebook has finished on Kaggle and you know the notebook slug and
version number, submit that version's `submission.csv` output with:

```bash
kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k e5jung/<NOTEBOOK> \
  -v <VERSION> \
  -m "FLE3N LightGBM submission"
```

Or use the wrapper in this repository after authenticating the Kaggle CLI. For
new Kaggle access tokens, export the `KGAT_...` value as `KAGGLE_API_TOKEN`
without committing or printing it:

```bash
export KAGGLE_API_TOKEN="<your KGAT token>"
./submit_kaggle_notebook_output.sh <NOTEBOOK> <VERSION> "FLE3N LightGBM submission"
```

For a step-by-step handover that another local Codex agent can follow to push,
run, and submit the Kaggle notebook, see `HANDOVER_KAGGLE_SUBMISSION.md`.

## Pipeline B likelihood-PF only

Run the standalone Pipeline B likelihood-weighted particle-filter candidate generator on Kaggle with:

```bash
python pipeline_b_likpf_only.py \
  --data-path /kaggle/input/rogii-wellbore-geology-prediction \
  --output-dir /kaggle/working
```

For a lightweight smoke test that only processes the first one or two sample wells but still writes `likpf_candidates.csv` and `submission_likpf_scale5.csv`, add `--max-wells`:

```bash
python pipeline_b_likpf_only.py \
  --data-path /kaggle/input/rogii-wellbore-geology-prediction \
  --output-dir /kaggle/working \
  --max-wells 1
```
