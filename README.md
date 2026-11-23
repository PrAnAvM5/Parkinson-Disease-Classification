# Parkinson's Disease classification (AML Assignment 2)

Advanced Machine Learning, Assignment 2. We compare an SVM (linear + RBF), an
MLP and a Fuzzy Inference System (FIS) for predicting `Diagnosis` (1 = PD,
0 = non-PD), choose each model's threshold by maximising F2, and evaluate the
three final models once on a held-out test set.

**Deadline:** Sun 6 Dec 2026, 20:00 · **Internal target:** full draft Sat 28 Nov

## Set-up (once)

```bash
git clone <repo-url>
cd Parkinson-Disease-Classification
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install             # keeps notebook outputs out of commits
```

## Run order

Run the notebooks top to bottom, in this order (or `bash run_all.sh`):

| # | Notebook | Owner | Reads | Writes |
|---|---|---|---|---|
| 01 | `01_data_understanding` | Charlotte | `data/raw/` | `data/processed/split.csv`, `results/figures/eda/` |
| 02 | `02_data_preparation` | Charlotte | train set | decisions in `src/preprocessing.py`, `results/figures/eda/` |
| 03a | `03a_model_svm` | Pranav | train set | `results/tuning/svm_configs.csv`, `results/scores/svm_oof.csv`, `results/models/svm.joblib`, `results/figures/models/svm_*.png` |
| 03b | `03b_model_mlp` | Pranav | train set, `svm_oof.csv` (for comparison) | `results/tuning/mlp_configs.csv`, `results/scores/mlp_oof.csv`, `results/models/mlp.joblib`, `results/figures/models/mlp_*.png` |
| 03c | `03c_model_fis` | Renske | train set | `results/scores/fis_oof.csv`, `results/models/fis.json` |
| 04 | `04_evaluation` | Hime, Marc | all `*_oof.csv`, models, **test set (once)** | `results/thresholds.json`, `results/test_metrics.csv` |
| 05 | `05_decision_support` | Hime, Marc | `results/test_metrics.csv` | `results/figures/decision/` |

03a, 03b and 03c can run in any order, except that 03b reads `svm_oof.csv` for its SVM comparison (run 03a first).

## Folders

- `data/raw/`: original CSV and attribute description. **Never edit.**
- `data/processed/`: `split.csv`, generated once by notebook 01.
- `src/`: shared functions imported by the notebooks (the toolbox):
  - `data.py`: loading, quality checks, the train/test split, column groups (`load_train()`, `load_test()`)
  - `preprocessing.py`: engineered features and the preprocessing pipeline (`add_engineered_features()`, `build_preprocessor()`)
  - `validation.py`: shared CV folds, OOF scores, F2 threshold, metrics (`get_cv()`, `oof_scores()`, `save_oof()`, `best_f2_threshold()`, `classification_metrics()`)
  - `fis.py`: the fuzzy inference system
- `notebooks/`: one notebook per step, one owner each (the workbench).
- `results/`: everything the notebooks produce (the hand-offs).

## Agreed file formats

| File | Format |
|---|---|
| `data/processed/split.csv` | `PatientID, split` where split is `train` or `test` |
| `load_train()` / `load_test()` | return `X` (DataFrame, PatientID index), `y` (Series); `load_test()` only works once `results/thresholds.json` exists |
| `build_preprocessor(use_engineered, use_bins)` | unfitted `ColumnTransformer`; SVM and MLP use `use_engineered=True, use_bins=True` (adds 10 one-hot decile bins for UPDRS, MoCA, FunctionalAssessment) |
| Models (`*.joblib`) | whole pipeline (preprocessing + model): call `add_engineered_features(X)` before scoring |
| `results/scores/*_oof.csv` | `PatientID, y_true, score` (higher score = more likely PD; SVM scores are decision-function values, MLP and FIS are probabilities in [0, 1]) |
| FIS | `FIS().fit(X_train)` and `.score(X)` returning values in [0, 1] |
| `results/thresholds.json` | `{"svm": t, "mlp": t, "fis": t}` |
| `results/test_metrics.csv` | `model, TP, FN, FP, TN, recall, precision, specificity, F2` |

## Rules

1. The test set is opened **only** in `04_evaluation`, once, after thresholds are frozen.
2. Preprocessing (scaling, encoding, bins) is fitted inside each CV fold (via `Pipeline`), never on all data. All models use the same folds from `get_cv()`.
3. At most 20 configurations for the SVM family (both kernels together) and 20 for the MLP.
4. FIS data-dependent settings come from training data only.
5. No changes to any model or threshold after seeing test results.
6. Keep code for methods we dropped, as long as it runs.

## Working with Git

- `git pull origin main` before you start.
- Work on your own branch (`git checkout -b <name>/<topic>`) and edit only your own files.
- Commit, push, open a pull request to `main`; a teammate has a quick look.
- Mention changes to `src/` in the group chat, since everyone imports it.
