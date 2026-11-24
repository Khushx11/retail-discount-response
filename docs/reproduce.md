# Reproduce the study

Run the commands from the repository root. The original run used Python 3.13.9 on Apple Silicon and two threads per fit. Install Python dependencies in a virtual environment. Your platform may require an OpenMP runtime for LightGBM. [Recorded environment](results/expanded_environment.json)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s outputs/COMP5064_Demo_and_Results/src -p test_timing.py
```

The repository bundles the four VISUELLE 2.0 input CSVs under `datasets/visuelle2/`. Derived feature and prediction tables, saved weights and bundled dependencies are stored locally. [Dataset files and citation](../datasets/visuelle2/README.md)

```sh
python prepare_features.py datasets/visuelle2
python work/final_report/robust/study.py
python work/final_report/robust/availability.py
python work/final_report/robust/robustness.py
python work/final_report/robust/support.py
python work/final_report/robust/summarize.py
python work/final_report/robust/verify.py
python work/final_report/robust/figures.py
python work/final_report/robust/demo.py --discount 0.3
```

The scripts write generated evidence to their original locations under `outputs/` and `work/`. Start full retraining in a fresh checkout to avoid reusing cached refit results. Use `robustness.py --resume` to reuse rolling folds whose summaries and prediction rows agree. A partially saved fold is recomputed. Separate bootstrap-refit caches are reused when present. The archive supplied by the author contains the feature table, saved predictions and checkpoints for verification and demo without retraining. Extract it to a separate directory and run its scripts from the extracted root.

[Verification record](results/verification.json) contains the archived results from reloading the checkpoints and reconstructing metrics. The record also includes the dense-grid monotonicity check and scores from 40 saved refit models. Run the verification script against outputs produced in the same run and environment.

## Verify the source CSVs

From the repository root, verify the bundled files before rebuilding features. If you obtained a copy from Humatics Lab, set `VISUELLE_DIR` to its extracted folder.

```sh
VISUELLE_DIR=datasets/visuelle2
manifest="$PWD/docs/visuelle2.sha256"
(cd "$VISUELLE_DIR" && shasum -a 256 -c "$manifest")
python prepare_features.py "$VISUELLE_DIR"
```

The four files in this manifest reproduced the archived 1,175,350-row feature table exactly, including row order and all 34 columns. The rebuilt data contains 578,053 training rows, 191,305 validation rows and 405,992 historical test rows. [Input checksums](visuelle2.sha256)

## Data and publication rights

Publisher permission covers the four bundled source CSVs. The full release is available through the publisher’s download form. Direct questions about further redistribution to Humatics Lab. [Publisher’s dataset page](https://humaticslab.github.io/forecasting/visuelle)

The portfolio includes aggregate model results and redacted coursework materials. [Adelaide University Academic Integrity Policy](https://adelaide.edu.au/about/policies/academic-integrity-policy/)
