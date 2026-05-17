# Paper Package

This folder contains a free-format medical-journal style manuscript draft for the project:

**A Sleep-Aware Blood Pressure Profiling Framework for Personalised Hypertension Monitoring**

## Files

- `manuscript.md` - main paper draft.
- `figures/` - paper-ready PNG figures.
- `tables/` - CSV result tables used in the paper.
- `figure_captions.md` - figure caption list.
- `scripts/build_paper_assets.py` - rebuilds paper figures/tables from `outputs/`.

## Rebuild

From the repository root:

```powershell
python sleep_aware_bp_framework.py --dryad-dir "..\24-hour physiological monitoring" --kaggle-arff "..\Kaggle dataset\y4dh3b3tfx-1\ABPM-dataset.arff" --output-dir outputs
python scripts\create_new_patient_framework_figure.py
python paper\scripts\build_paper_assets.py
```

## Clinical Boundary

The paper frames the framework as clinician-review support. It does not claim automatic diagnosis, medication adjustment, admission reduction, or prospective clinical validation.

Before journal submission, the clinical wording should be reviewed by an independent hypertension expert, ideally a consultant cardiologist, consultant physician, nephrologist, or ABPM-experienced clinician in Sri Lanka.
