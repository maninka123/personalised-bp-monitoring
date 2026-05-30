# ABPM-TSL research folder

This folder expands the limited-input ABPM teacher-student method into a paper-ready research package.

Files:

- `research_article.md` - manuscript-style article with key points, method, prototype results, limitations, and references.
- `scripts/run_abpm_tsl_pipeline.py` - reproducible synthetic-data, NN training, ablation, and figure-generation pipeline.
- `data/synthetic_limited_inputs.csv` - generated synthetic limited-input dataset.
- `results/` - teacher, baseline, student, missingness, input-group, and regression result tables.
- `models/student_abpm_tsl_torchscript.pt` - trained TorchScript student model used by the app when available.
- `prototype_results.csv` - outputs from the app-facing prototype estimator presets.
- `figures/` - SVG figures referenced by the article.

Important boundary:

The limited-input predictions are for method development and proof-of-concept testing. They are not clinical diagnostic claims and must be validated on real paired clinic/home BP plus ABPM data before clinical use.
