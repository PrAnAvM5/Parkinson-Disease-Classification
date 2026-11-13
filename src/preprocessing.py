"""Feature engineering and the preprocessing pipeline. Used by 02, 03a, 03b.

Owner: Charlotte
Should provide: add_engineered_features(), build_preprocessor().
See README for the agreed formats.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data import CONTINUOUS_COLS, BINARY_COLS, CATEGORICAL_COLS, ORDINAL_COLS

def add_engineered_features(df):
    """Add row-wise engineered features."""
    df = df.copy()
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]

    SYMPTOM_COLS = [
        "Tremor", "Rigidity", "Bradykinesia",
        "PosturalInstability", "SpeechProblems",
        "SleepDisorders", "Constipation"
    ]
    df["SymptomCount"] = df[SYMPTOM_COLS].sum(axis=1)

    return df


def build_preprocessor(use_engineered = False):
    """
    Returns an UNFITTED ColumnTransformer.
    Fit it only inside cross-validation folds to prevent data leakage!
    """
    continuous = CONTINUOUS_COLS.copy()

    if use_engineered:
        continuous += ["PulsePressure", "SymptomCount"]

    return ColumnTransformer(transformers=[
        ("scale", StandardScaler(), continuous),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("pass", "passthrough", BINARY_COLS + ORDINAL_COLS),
    ])