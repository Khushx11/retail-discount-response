# Reproduce the study

Run the commands from the repository root. The original run used Python 3.13.9 on Apple Silicon and two threads per fit. Install Python dependencies in a virtual environment. Your platform may require an OpenMP runtime for LightGBM. [Recorded environment](results/expanded_environment.json)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s outputs/COMP5064_Demo_and_Results/src -p test_timing.py
```

Obtain the dataset from the original publisher and follow its access terms. Pass the directory containing `sales.csv`, `price_discount_series.csv`, `restocks.csv` and `vis2_gtrends_data.csv` to the feature builder. Dataset records, derived row-level data, saved weights, local assessment documents and bundled dependencies are kept outside the repository. [VISUELLE 2.0 access and citation](https://humaticslab.github.io/forecasting/visuelle)

```sh
python prepare_features.py /path/to/VISUELLE2
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

Download VISUELLE 2.0 from [Humatics Lab](https://humaticslab.github.io/forecasting/visuelle) and extract it locally. The study uses four CSVs from the release. From the repository root, set `VISUELLE_DIR` to the extracted folder and check the files before rebuilding features.

```sh
VISUELLE_DIR=/path/to/visuelle2
manifest="$PWD/docs/visuelle2.sha256"
(cd "$VISUELLE_DIR" && shasum -a 256 -c "$manifest")
python prepare_features.py "$VISUELLE_DIR"
```

The four files in this manifest reproduced the archived 1,175,350-row feature table exactly, including row order and all 34 columns. The rebuilt data contains 578,053 training rows, 191,305 validation rows and 405,992 historical test rows. [Input checksums](visuelle2.sha256)

## Data and publication rights

The publisher provides VISUELLE 2.0 through a download form. The benchmark code repository carries a CC BY-NC-SA 4.0 licence file. The publisher’s dataset download terms could not be verified from the supplied material. Dataset redistribution permission was not established from the project page or the archive, so use a copy obtained under the publisher’s terms. The portfolio includes aggregate metrics and plots only. [Publisher’s dataset page](https://humaticslab.github.io/forecasting/visuelle) · [Benchmark repository licence file](https://github.com/HumaticsLAB/visuelle2.0-code/blob/main/LICENSE.txt)

The portfolio uses aggregate evidence. Check course-specific permission before redistributing any original assessment materials. [Adelaide University Academic Integrity Policy](https://adelaide.edu.au/about/policies/academic-integrity-policy/)
