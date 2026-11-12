"""Loading, quality checks and the stratified train/test split. Used by every notebook.

Owner: Charlotte
Should provide: load_raw(), check_quality(), make_split(), load_train(), load_test(); shared settings (paths, seed, column groups).
See README for the agreed formats.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import warnings

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
DATA_RAW = os.path.join(PARENT_DIR, "data", "raw", "parkinsons_disease_data.csv")

SPLIT_PATH = os.path.join(PARENT_DIR, "data", "processed", "split.csv")
SEED = 42

CONTINUOUS_COLS = [
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity",
    "DietQuality", "SleepQuality", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "UPDRS", "MoCA", "FunctionalAssessment"
]

BINARY_COLS = [
    "Gender", "Smoking", "FamilyHistoryParkinsons",
    "TraumaticBrainInjury", "Hypertension", "Diabetes",
    "Depression", "Stroke", "Tremor", "Rigidity",
    "Bradykinesia", "PosturalInstability", "SpeechProblems",
    "SleepDisorders", "Constipation"
]

CATEGORICAL_COLS = ["Ethnicity"]   # → OneHotEncoder
ORDINAL_COLS     = ["EducationLevel"]  # → treat as numeric / OrdinalEncoder
TARGET_COL = "Diagnosis"

def load_raw():
    """Load the raw dataset, set index, drop non-features."""
    df = pd.read_csv(DATA_RAW)
    df = df.set_index('PatientID')
    df = df.drop(columns=["DoctorInCharge"])
    return df


def check_quality(df):
    """Print a data quality report."""
    print("\n--- SHAPE ---")
    print(df.shape)

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")

    print("\n--- DUPLICATE ROWS ---")
    n_dupes = df.duplicated().sum()
    print(f"{n_dupes} duplicate rows found")

    print("\n--- CLASS BALANCE ---")
    print(df["Diagnosis"].value_counts())
    print(df["Diagnosis"].value_counts(normalize=True).round(3))

    print("\n--- INCONSISTENCY: DiastolicBP >= SystolicBP ---")
    bp_issues = df[df["DiastolicBP"] >= df["SystolicBP"]]
    print(f"{len(bp_issues)} observations flagged")

    print("\n--- INCONSISTENCY: LDL + HDL > CholesterolTotal ---")
    chol_issues = df[
        (df["CholesterolLDL"] + df["CholesterolHDL"]) > df["CholesterolTotal"]
    ]
    print(f"{len(chol_issues)} observations flagged")

    print("\n--- DTYPES ---")
    print(df.dtypes)

    print("\n--- BASIC STATS ---")
    print(df.describe().T)

    return {"bp_issues": bp_issues, "chol_issues": chol_issues}


def make_split(df):
    """
    Create a stratified 80/20 train/test split and save to split.csv.
    Run ONCE and commit — never re-run after this!
    """
    if os.path.exists(SPLIT_PATH):
        print(f"Split already exists at {SPLIT_PATH}. Skipping to prevent overwrite!")
        return

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["Diagnosis"], random_state=SEED
    )
    df_split = df.copy()
    df_split["split"] = "test"
    df_split.loc[train.index, "split"] = "train"
    df_split[["split"]].to_csv(SPLIT_PATH)
    print(f"Split saved to {SPLIT_PATH}")
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")


def _load_split(split_name):
    """Internal helper: load rows for a given split."""
    df = load_raw()
    split = pd.read_csv(SPLIT_PATH, index_col="PatientID")
    df = df.join(split)
    subset = df[df["split"] == split_name].drop(columns=["split"])
    X = subset.drop(columns=["Diagnosis"])
    y = subset["Diagnosis"]
    return X, y

def load_train():
    """Return (X_train, y_train) — safe to use anytime."""
    return _load_split("train")

def load_test():
    """Return (X_test, y_test) — ONLY call this after thresholds are frozen!"""
    warnings.warn("load_test() called! Only use this after all models and thresholds are finalized!", UserWarning, stacklevel=2)
    return _load_split("test")