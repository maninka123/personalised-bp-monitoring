# Release Notes: Research Paper Package

## What is included

- New `paper/` folder with a journal-style manuscript draft.
- Paper-ready figures describing:
  - framework pipeline,
  - dataset complementarity,
  - Dryad sleep-aware profiles,
  - awake versus sleep SBP,
  - morning surge distribution,
  - example 24-hour BP curves,
  - Kaggle feature importance,
  - new-patient workflow with separate ML support validation,
  - results dashboard.
- Paper result tables as CSV files.
- Script to rebuild paper assets from generated analysis outputs.
- Streamlit and npm assistant fixes:
  - custom questions are no longer submitted as blank,
  - assistant answers render markdown formatting more cleanly.

## Main analysis results

- Dryad raw ABPM rows: 1,623.
- Valid ABPM rows after zero filtering: 1,090.
- Participants with valid ABPM: 30.
- Insufficient sleep BP data: participants `007`, `009`, `014`.
- Dipping categories:
  - normal dipper: 17,
  - non-dipper: 7,
  - extreme dipper: 3,
  - insufficient sleep BP data: 3.
- Kaggle best model results:
  - BP-Load: Random Forest, AUROC 0.980, F1 0.986, balanced accuracy 0.976.
  - Circadian-Rythm: Random Forest, AUROC 0.991, F1 0.955, balanced accuracy 0.935.
  - Morning-Surge: Logistic Regression, AUROC 0.941, F1 0.697, balanced accuracy 0.874.
  - Pulse-Pressure: Logistic Regression, AUROC 0.732, F1 0.828, balanced accuracy 0.737.

## Interpretation boundary

The new-patient profile is assigned using transparent rule-based BP features. Kaggle ML is separate support validation only; it does not make the final clinical decision.
