# Review and publication

Prepared branch: `research/leakage-aware-portfolio`.
Base: `0613cf02577526aafc4b9c7078a2a5d779609b06` (GitHub main at review time).
The checkout's origin is `https://github.com/eb-jung/rogii-kaggle.git`.
No push, merge, PR creation, Kaggle submission, or visibility change was performed.

## Review locally

From this checkout:

```bash
git status --short --branch
git diff --stat main...HEAD
git diff main...HEAD -- README.md src tests docs reports/validation.md
```

Read `docs/research_audit.md` for version/cell evidence and the resume claim audit.
Read `reports/metrics.json` and `reports/test_well_metrics.csv` for the verified
result. Local model/submission files are in ignored `artifacts/full_fixed_204/`.
The original research directory was preserved, including its existing dirty edit.

## Publish only after explicit user approval

These are proposed future commands, not actions already taken:

```bash
git fetch origin
git diff --stat origin/main...HEAD
git push -u origin research/leakage-aware-portfolio
```

Then open a pull request targeting `main`, review attribution/data exclusions,
and merge only with separate authorization. If `origin/main` advanced, review
the differences before rebasing or merging. Do not force-push or change visibility.

Suggested PR title: `Recover leakage-aware wellbore regression and verified evaluation`.

Suggested resume bullet, contingent on personally reviewing and understanding
the implementation:

> Adapted and validated a **LightGBM ensemble** for wellbore trajectory prediction,
> preserving known geological context and isolating wells across evaluation splits;
> reduced RMSE **6.4%** versus a last-known-value baseline on 77 held-out wells.

Do not retain CatBoost, OOF Ridge stacking, public leaderboard score, or sole
algorithmic originality claims for the selected version. The archived source has
those model names but does not establish their valid executed performance here.
