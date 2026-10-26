# Recovery and research-validity audit

## Scope and provenance

Reviewed recursively: local `Kaggle/` source notebooks, both kernel upload folders,
the local `rogii-kaggle/` checkout, three saved submission CSVs, all training/test
well CSVs, and the problem-setting slides. The local committed HEAD and GitHub
`main` both resolve to `0613cf02577526aafc4b9c7078a2a5d779609b06`.
The original checkout has an uncommitted handover-document edit; it was not touched
or silently incorporated. The clean branch was cloned from the committed state.
Original local files remain unchanged. Previously tracked experiments were moved
byte-for-byte into `archive/github_legacy/`; their historical Git provenance remains.
`reports/recovery_inventory.json` records SHA-256 hashes of recovered source files.

Notebook cell numbers below are zero-based. Source paths identify the original
local `Kaggle/` hierarchy. Public-source notebook copies are not newly redistributed;
only already-tracked historical versions remain in the archive. Attribution is not
evidence of a license to redistribute additional copies.

## Implementation comparison

| Implementation | Actually implemented | Evaluation / reproducibility | Portfolio decision |
|---|---|---|---|
| `notebooks/02_baseline.ipynb` | HistGradientBoosting, typewell summaries, prefix anchor/slope, geometric deltas, mean baseline | GroupShuffleSplit by well; missing-input suffix labels only; official-overlap wells excluded; saved 17.928051 RMSE on 154 wells, 774,130 rows. This is a historical saved output, not a fresh result. | Strongest recovered validation design; source of the repaired context/suffix protocol. |
| `notebooks/04_public_best_lb_7_776.ipynb` and matching upload copy | LightGBM, CatBoost, Ridge over OOF columns; PF/beam alignment; spatial formation priors; heuristic blend | No saved execution outputs. External `koolbox`/pretrained artifacts needed for exact reproduction. Formation priors are built before folds; inference substitutes train-copy formation reconstruction on overlap wells. | Most elaborate architecture, not the strongest defensible evaluation. No performance retained. |
| `notebooks/04_public_best_ridge_sp45_projection.ipynb` and upload copy | Above plus robust polynomial projection of TVT+Z | Same issues; projection comments claim CV improvements without supporting saved execution evidence here. | Reference only; do not claim the filename's public score as this user's result. |
| GitHub `fle3n_rogii_v5_train.py` and notebook | LightGBM GroupKFold, DTW when available, GR rolling statistics, TVT context, smoothing | Selects known-input rows as supervised validation targets. At those rows fwd/bwd features include the row's own TVT_input, which equals TVT in the released files. | Reuse appropriate per-well features, replace supervised-row selection and evaluation. |
| GitHub `leakage_version.ipynb` | Earlier FLE3N-style LightGBM | Same known-row shortcut; saved 7.6154 average fold RMSE invalid for suffix prediction. Recreates row IDs after suffix filtering, risking incorrect submission alignment. | Quarantined historical artifact; no score retained. |
| GitHub `best_submission.ipynb`; local `03_best_score_submission.ipynb` | Exact target lookup plus unused fallback on visible rows | Lookup from train TVT by well and original row index; saved coverage 14,151/14,151. | Data-overlap audit evidence only; never ML generalization. |
| GitHub `rogii-dual-pipeline-blend.ipynb` | Two large PF/GBM stacks, LightGBM/CatBoost/Ridge, projection, final blend, guarded overlap reconstruction | No saved execution outputs. Spatial priors precede folds; second-level OOF evaluation is not fully nested; overlap reconstruction remains in inference. Some plotted result numbers are hard-coded. | Archive, not default portfolio entry point. |
| GitHub `pipeline_b_likpf_only.py` / notebook | Standalone likelihood-weighted particle filter with multiple seeds/scales, known-prefix calibration, SG smoothing | Uses legitimate prefix context and typewell input; no target lookup in inspected standalone path. No held-out baseline comparison established; numba absent locally. | Plausible future benchmark candidate; not evidence for the resume's GBM/Ridge claims. |
| New `src/rogii/` | Recovered context features + FLE3N-style GR summaries and typewell proxy; three-fold LightGBM average | Explicit feature whitelist, suffix-only supervision, separate train/validation/test wells, duplicate/overlap checks, fixed hyperparameters, full outer suffix evaluation | Selected practical portfolio version. Smaller architecture, inspectable validation. |

## Feature availability and fixes

The supplied task slides state: horizontal TVT is known only before prediction
start; typewell TVT and GR are supplied throughout; XYZ/MD/GR are available after
prediction start. All depth/trajectory quantities are in feet.

1. **Released-data overlap is real, but not generalization.** The three visible
   test well IDs also occur in train. The lookup notebook explicitly reads train
   `TVT` for requested IDs. We do not assess competition-rule permissibility here.
   Neither a zero lookup error nor a train-formation reconstruction is retained
   as predictive validation. Official test wells and exact deployable-input
   duplicates are excluded before internal splits.
2. **Preserve the known prefix, supervise the missing suffix.** Historical
   `fle3n_rogii_v5_train.py:212-240` propagates known context; `run():476` then
   selects known rows for supervision. The problem is that combination, not the
   existence of prefix context. New `features.py` retains only prefix-derived
   anchor/slope, returns only naturally masked suffix rows, and never reads
   horizontal target/formation columns. Mutating unavailable labels cannot change
   features (tested). Future known TVT after the first missing row is rejected.
3. **No cross-well fitting before splits.** In reference SP45 cell 8 and dual
   notebook cells 9/47/55, spatial imputers use all train wells. Excluding the
   current well still permits other outer-validation wells' restricted formation
   information into features. Such features need fold-specific construction;
   they are omitted from this focused version. Per-well GR transforms and
   supplied typewell summaries require no fitting on other wells and are valid
   under the declared offline inference setting.
4. **OOF stacking distinction.** Reference SP45 cells 14/16/19 and dual cell 51
   do construct base OOF columns and group-wise Ridge folds. But meta-training
   OOF rows can have been generated by base models trained on the meta-validation
   wells. Their meta-OOF score is not fully nested out-of-sample evaluation.
   Pretrained `Trainer` objects cannot be independently audited without their
   actual artifacts/implementation. We therefore do not claim verified Ridge
   stacking. New code uses an equal fold average, evaluated on separate outer
   wells whose labels never reach any fitted estimator.
5. **Alignment is strict.** Original CSV row indices are retained, sample IDs
   must match exactly one-to-one, and missing/duplicate/nonfinite predictions
   raise errors. There is no silent median fallback. Smoothing runs separately
   within each well with a fixed 11-point cubic Savitzky-Golay window.

## Fixed validation protocol

Declared before the fresh result: seed 204; deterministic shuffle of eligible
sorted well IDs; 10% internal test, 10% validation, remaining training. Three
inner GroupKFold models with 200 trees, learning rate .05, 31 leaves, and L2=1.
No early stopping or hyperparameter search. Each training well contributes up to
200 evenly spaced suffix rows to keep CPU work bounded. Both outer roles use
every suffix row. All compared methods are scored on identical target rows.

The supervised target is TVT minus the last supplied prefix TVT; the anchor is
added back at inference. Baselines are the training-sample mean, prefix last
value, and recent-prefix linear extrapolation. Primary deployed estimator is the
predeclared fold average plus SG11; raw average is reported as a diagnostic.
No choice is made between variants based on internal-test outcomes. The model
is not refitted on validation/test wells before submission generation.

Saved historical validation influenced the design choice, so the new partition
is a fresh computational holdout, not an untouched historical research dataset.
Nearby wells may share geology; well separation does not establish geographic
or prospective generalization. Row RMSE weights longer wells more heavily;
mean/median per-well RMSE are reported too. No independent-row confidence interval
is used. Official-test labels and leaderboard scores play no role in tuning.

## Resume claim audit

| Original claim | Evidence | Current resume decision |
|---|---|---|
| LightGBM implemented and used | Present in historical scripts; fresh selected execution recorded in reports | Retain after fresh execution succeeds. |
| CatBoost implemented and used | Training/loading code exists in public-adapted notebooks; no saved execution outputs or model artifacts recovered | Do not claim verified use in selected project. |
| OOF Ridge stacking | Base-OOF and Ridge code exists; nesting/provenance not established | Remove from selected-project bullet. |
| Correct group-aware CV | Historical grouping alone did not fix feature/stacking leakage | Retain only for repaired implementation and separate outer wells. |
| Domain-specific post-processing | SG smoothing, damping, projection code exist; selected version uses SG11 | Describe precisely; no unsupported performance gain. |
| Original intellectual contribution | Local notebooks acknowledge public sources; authorship cannot be inferred from uploads or ownership | Say adapted/recovered; claim only personally defensible contributions. |

## Attribution

The selected code consolidates the recovered local baseline's context/suffix
design and the existing repository's FLE3N-style GR-feature approach. Standard
LightGBM, grouped resampling, residual targets, and Savitzky-Golay filtering are
not presented as new algorithms. This audit/repackaging was performed with Codex
assistance; file possession does not establish sole authorship of earlier code.

Historical local notebooks explicitly identify public references including:

- lightningv08, `lb-7-776-rogii` (as named in the recovered notebook; its score is not ours)
- [needless090 ridge-sp](https://www.kaggle.com/code/needless090/lb-7-878-rogii-ridge-sp)
- [ravaghi reference/artifacts](https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-ridge)
- [sarpal465 SP45](https://www.kaggle.com/code/sarpal465/rogii-ridge-sp45)
- fleongg/FLE3N, aidensong123, romantamrazov, qamarmath as credited in the archived notebooks

This is a provenance record from local source, not a fresh verification of those
external pages, licenses, or public leaderboard scores. No new public notebook
code, raw competition data, external model artifacts, or credentials are uploaded.
