# rogii-kaggle
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
