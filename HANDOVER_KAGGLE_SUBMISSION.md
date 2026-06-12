# Handover: submit the FLE3N ROGII notebook output to Kaggle

This file is for a follow-up agent running in a local Codex chat or another
environment with normal outbound network access. The current hosted environment
could not complete the upload because GitHub, PyPI, and Kaggle endpoints were
blocked by the proxy.

## Goal

Submit the Kaggle notebook-version output file named `submission.csv` to the
`rogii-wellbore-geology-prediction` competition using this command pattern:

```bash
kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k e5jung/<NOTEBOOK> \
  -v <VERSION> \
  -m "FLE3N LightGBM submission"
```

## Inputs the local agent needs

- A working internet connection to GitHub, PyPI, and Kaggle.
- Kaggle CLI installed and authenticated.
- A Kaggle `KGAT_...` access token supplied out-of-band by the user.
  - Do **not** commit it, echo it into logs, or write it into this repo.
  - Prefer setting it only for the current shell: `export KAGGLE_API_TOKEN='KGAT_...'`.
- The Kaggle username/account is expected to be `e5jung`.
- Either:
  - an existing Kaggle notebook slug plus completed version number, or
  - permission to create/update the notebook slug `fle3n-rogii-v5-train`.

## Local setup checklist

```bash
git clone https://github.com/eb-jung/rogii-kaggle.git
cd rogii-kaggle
git pull --ff-only origin main || git pull --ff-only origin master

python -m pip install --upgrade kaggle
kaggle --version

export KAGGLE_API_TOKEN='KGAT_REPLACE_WITH_USER_TOKEN'
kaggle competitions list -s rogii
```

If authentication works, the competition list command should return the ROGII
competition or at least exit without an auth error.

## If the Kaggle notebook already exists and has a completed version

Run the repository wrapper from the repo root:

```bash
./submit_kaggle_notebook_output.sh <NOTEBOOK> <VERSION> "FLE3N LightGBM submission"
```

For example, if the completed notebook is `e5jung/fle3n-rogii-v5-train` and its
completed version is `1`, run:

```bash
./submit_kaggle_notebook_output.sh fle3n-rogii-v5-train 1 "FLE3N LightGBM submission"
```

Then verify the submission appears in Kaggle:

```bash
kaggle competitions submissions -c rogii-wellbore-geology-prediction
```

## If the Kaggle notebook needs to be created or rerun first

Create a temporary Kaggle kernel upload folder. Keep generated metadata out of
the repo unless the user explicitly asks to track it.

```bash
KAGGLE_USER=e5jung
KERNEL_SLUG=fle3n-rogii-v5-train
KERNEL_DIR="$(mktemp -d)"

cp fle3n_rogii_v5_train.ipynb "${KERNEL_DIR}/fle3n_rogii_v5_train.ipynb"
cat > "${KERNEL_DIR}/kernel-metadata.json" <<EOF
{
  "id": "${KAGGLE_USER}/${KERNEL_SLUG}",
  "title": "FLE3N ROGII v5 train",
  "code_file": "fle3n_rogii_v5_train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "false",
  "dataset_sources": [],
  "competition_sources": ["rogii-wellbore-geology-prediction"],
  "kernel_sources": [],
  "model_sources": []
}
EOF

kaggle kernels push -p "${KERNEL_DIR}"
```

Wait for the Kaggle run to complete, then inspect status/output files:

```bash
kaggle kernels status e5jung/fle3n-rogii-v5-train
kaggle kernels files e5jung/fle3n-rogii-v5-train
```

Confirm the output list includes `submission.csv`. Determine the completed
version number from the Kaggle UI or CLI output, then submit that notebook
version:

```bash
./submit_kaggle_notebook_output.sh fle3n-rogii-v5-train <VERSION> "FLE3N LightGBM submission"
kaggle competitions submissions -c rogii-wellbore-geology-prediction
```

## Troubleshooting

- `401` or auth errors: re-export `KAGGLE_API_TOKEN` in the same shell and rerun
  `kaggle competitions list -s rogii`.
- `403` on competition access: confirm the `e5jung` Kaggle account has accepted
  the ROGII competition rules.
- `submission.csv` missing from notebook output: open the Kaggle notebook logs;
  the final cell in `fle3n_rogii_v5_train.ipynb` should run `run(args)` with
  `submission_file="submission.csv"`.
- Notebook run fails on dependencies: Kaggle usually includes `numpy`, `pandas`,
  `scikit-learn`, and `scipy`; install or enable any missing optional packages
  in the notebook only if the logs require it.
- Local hosted Codex cannot do this if GitHub/PyPI/Kaggle are blocked; run these
  steps from a local shell or a network-enabled Codex session.
