"""Feature engineering and the preprocessing pipeline. Used by 02, 03a, 03b.

Owner: Charlotte
Should provide: add_engineered_features(), build_preprocessor().
See README for the agreed formats.
"""

import inspect
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from src.data import CONTINUOUS_COLS, BINARY_COLS, CATEGORICAL_COLS, ORDINAL_COLS

MOTOR_SYMPTOM_COLS = ["Tremor", "Rigidity", "Bradykinesia", "PosturalInstability"]
NON_MOTOR_SYMPTOM_COLS = ["SpeechProblems", "SleepDisorders", "Constipation"]

# Clinical scales with threshold-like relationships to Diagnosis (EDA quintile plots, notebook 01)
BIN_COLS = ["UPDRS", "MoCA", "FunctionalAssessment"]
N_BINS = 10
# Same quantile definition on every scikit-learn version (the parameter exists from 1.7 on)
_KBINS_KW = ({"quantile_method": "averaged_inverted_cdf"}
             if "quantile_method" in inspect.signature(KBinsDiscretizer).parameters else {})

def add_engineered_features(df):
    """Add row-wise engineered features."""
    df = df.copy()

    # Computed for EDA inspection only — excluded from model pipeline
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]

    df["MotorSymptomCount"] = df[MOTOR_SYMPTOM_COLS].sum(axis=1)
    df["SymptomCount"] = df[NON_MOTOR_SYMPTOM_COLS].sum(axis=1) 

    return df


def build_preprocessor(use_engineered=False, use_bins=False, n_bins=N_BINS):
    """
    Returns an UNFITTED ColumnTransformer.
    Fit it only inside cross-validation folds to prevent data leakage!

    use_engineered: add SymptomCount and MotorSymptomCount to the scaled block.
    use_bins: add quantile bins of BIN_COLS as one-hot columns (the scaled originals stay in).
              Bin edges are learned from the data the pipeline is fitted on (training folds
              only) and the step ignores the labels, so it cannot leak validation/test information.
    """
    continuous = CONTINUOUS_COLS.copy()

    if use_engineered:
        continuous += ["SymptomCount", "MotorSymptomCount"]

    transformers = [
        ("scale", StandardScaler(), continuous),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("pass", "passthrough", BINARY_COLS + ORDINAL_COLS),
    ]
    if use_bins:
        transformers.append(
            ("bins", KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile",
                                      **_KBINS_KW), BIN_COLS))

    return ColumnTransformer(transformers=transformers)