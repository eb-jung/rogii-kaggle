# Executed validation and remaining limits

## Fresh experiment

- Python 3.14.0; package versions are pinned in `../requirements.txt`.
- Initial import failed because LightGBM could not locate `libomp.dylib`.
- Resolved with the scikit-learn OpenMP runtime already installed in the original
  virtual environment, exposed only for each command through `DYLD_LIBRARY_PATH`.
  No packages were installed into the original environment and no source research
  files were changed. A Homebrew fetch probe found no requested Sonoma bottle;
  no Homebrew package was installed.
- Fixed seed 204; 200 rows per training well; 200 iterations; three grouped models.
- 773 train wells audited; 3 official-overlap wells excluded; 770 eligible;
  no additional exact full-input duplicate wells detected.
- 616 training / 77 validation / 77 internal-test wells, mutually disjoint.
- 123,200 sampled training rows; all 372,424 validation and 365,592 test suffix rows.
- Source hashes in `run_manifest.json` match the executed feature/experiment code.
  The machine-specific data path is redacted only in the public report copy.

Equivalent executed commands from the repository root, with `ROGII_VENV` pointing
to the supplied folder's existing `.venv` and `ROGII_DATA` to its `data/raw`:

```bash
DYLD_LIBRARY_PATH="$ROGII_VENV/lib/python3.14/site-packages/sklearn/.dylibs" \
  PYTHONPATH=src "$ROGII_VENV/bin/python" -m unittest discover -s tests -v

DYLD_LIBRARY_PATH="$ROGII_VENV/lib/python3.14/site-packages/sklearn/.dylibs" \
  PYTHONPATH=src "$ROGII_VENV/bin/python" -m rogii.experiment \
  --data-dir "$ROGII_DATA" --output-dir artifacts/full_fixed_204 \
  --train-rows-per-well 200 --rounds 200
```

Runtime 27.29 seconds for the fresh experiment on the available local setup.
The primary internal-test row RMSE is 13.557168259936638 ft versus
14.489390883943543 ft for last-known TVT, a 6.4338289405937% reduction.
The model improves per-well RMSE on 58/77 test wells and 48/77 validation wells;
it does not improve every well. No significance test is claimed.

Historical HGB RMSE 17.928051 and invalid known-row OOF RMSE 7.6154 are not
comparable to this fresh split. Neither is used to claim a measured improvement.

## Tests and independent checks

**10 unittest tests passed**, including actual small LightGBM fits:

1. Hidden target and formation/Geology mutation leaves features unchanged.
2. Target columns can be absent entirely.
3. Legitimate prefix-context changes are retained in features.
4. Non-prefix known TVT is rejected.
5. Duplicate/non-increasing measured depth is rejected.
6. Submission order is preserved; missing/duplicate/nonfinite outputs fail.
7. Outer splits are disjoint and invariant to input-list ordering.
8. Inner folds are disjoint; repeated small model fits/predictions match exactly.
9. Smoothing does not cross well boundaries.
10. Official test IDs, renamed exact official-input duplicates, and duplicate
    training sequences are excluded before splitting.

Saved production-model reload reproduced all 14,151 official-test predictions
exactly. A separate inference invocation was also run against a test-only directory
with no `train/` directory to establish the inference path's independence from
released training targets. The official submission was generated, not uploaded.

All ten original tracked files, including README, were verified byte-for-byte
against the base commit after moving to `archive/github_legacy/`.
Credential-pattern scanning found only the explicit `KGAT_REPLACE_WITH_USER_TOKEN`
placeholder in the historical handover document, not an actual matching token.
No raw data, model binaries, or prediction CSVs are staged for publication.

## Limits

Only small-model deterministic refits were repeated; the full training experiment
was run once. Saved-model inference was checked separately. Environment lock
records Python packages, not a fully portable OS/native-library image.
Well separation does not remove spatial dependence. Historical development used
this released dataset, and prior notebook outputs were available during audit.
The final test is now inspected; future model selection must not optimize it.

No fresh CatBoost, Ridge stack, DTW, particle-filter, or external-artifact result
was produced. Reconstructing such a stack properly would require fold-specific
spatial features and independent outer evaluation. It is intentionally deferred.
