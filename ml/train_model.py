import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SMARTEDGE - RANDOM FOREST TRAINING
# Leakage-free future-request prediction
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    r"D:\smart edge\dataset\processed\ml_dataset.csv"
)

MODEL_DIR = (
    r"D:\smart edge\model"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "smartedge_random_forest.pkl"
)

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "predictions.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "hour",
    "day_of_week",
    "past_frequency",
    "recency_seconds",
    "content_size"
]

TARGET = "future_request"


# ============================================================
# 3. SETTINGS
# ============================================================

TRAIN_RATIO = 0.80

RANDOM_STATE = 42

N_ESTIMATORS = 200

MAX_DEPTH = 16

MIN_SAMPLES_LEAF = 2


# ============================================================
# 4. START
# ============================================================

print("=" * 70)
print("SMARTEDGE RANDOM FOREST TRAINING")
print("=" * 70)

print("\nTraining configuration:")
print(f"Training ratio      : {TRAIN_RATIO}")
print(f"Number of trees     : {N_ESTIMATORS}")
print(f"Maximum tree depth  : {MAX_DEPTH}")
print(f"Minimum leaf size   : {MIN_SAMPLES_LEAF}")
print(f"Random state        : {RANDOM_STATE}")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING ML DATASET")
print("=" * 70)

print(f"Dataset: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nML dataset not found:\n{INPUT_FILE}\n\n"
        "Run prepare_ml_dataset.py first."
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

print("\nChecking dataset columns...")

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print(f" - {column}")

    raise ValueError(
        "The ML dataset does not contain "
        "the required columns."
    )

print("All required columns found.")


# ============================================================
# 7. CONVERT TIMESTAMP
# ============================================================

if "timestamp" in df.columns:

    print("\nConverting timestamp...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )


# ============================================================
# 8. CLEAN NUMERICAL FEATURES
# ============================================================

print("\nCleaning numerical features...")

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


# ============================================================
# 9. REMOVE INVALID VALUES
# ============================================================

before_cleaning = len(df)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURES + [TARGET]
)

after_cleaning = len(df)

print(
    f"Removed invalid rows: "
    f"{before_cleaning - after_cleaning:,}"
)

print(
    f"Remaining rows: "
    f"{after_cleaning:,}"
)


# ============================================================
# 10. MAKE SURE TARGET IS 0 / 1
# ============================================================

df[TARGET] = (
    df[TARGET]
    .astype(int)
)

invalid_target_values = (
    ~df[TARGET].isin([0, 1])
)

if invalid_target_values.any():

    print(
        "\nRemoving invalid target values..."
    )

    df = df[
        ~invalid_target_values
    ]


# ============================================================
# 11. SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting data chronologically...")

if "timestamp" in df.columns:

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )


# ============================================================
# 12. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_counts = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

total = len(df)

for label, count in target_counts.items():

    percentage = (
        count / total * 100
    )

    if label == 0:
        name = "No Future Request"
    else:
        name = "Future Request"

    print(
        f"{name:20s}: "
        f"{count:10,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 13. PREPARE X AND Y
# ============================================================

print("\nPreparing features and target...")

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


print(
    f"Feature matrix: {X.shape}"
)

print(
    f"Target vector : {y.shape}"
)

print("\nFeatures:")

for feature in FEATURES:
    print(f" - {feature}")

print(
    f"\nTarget: {TARGET}"
)


# ============================================================
# 14. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)

split_index = int(
    len(df) * TRAIN_RATIO
)

X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_train = y.iloc[
    :split_index
].copy()

y_test = y.iloc[
    split_index:
].copy()


print(
    f"Training samples: {len(X_train):,}"
)

print(
    f"Testing samples : {len(X_test):,}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Training uses earlier requests."
)

print(
    "Testing uses later requests."
)


# ============================================================
# 15. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print(
    "Training model..."
)

model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Training completed successfully."
)


# ============================================================
# 16. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


print(
    f"Predictions generated: "
    f"{len(y_pred):,}"
)


# ============================================================
# 17. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


try:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

except ValueError:

    roc_auc = 0.0


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Future Request",
            "Future Request"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")
print("-" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


for _, row in feature_importance.iterrows():

    print(
        f"{row['feature']:20s}: "
        f"{row['importance']:.4f}"
    )


# ============================================================
# 21. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print(
    f"\nFeature importance saved to:"
)

print(
    FEATURE_IMPORTANCE_FILE
)


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

prediction_results = X_test.copy()

prediction_results["actual"] = (
    y_test.values
)

prediction_results["predicted"] = (
    y_pred
)

prediction_results["prediction_probability"] = (
    y_probability
)


prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    "\nPredictions saved to:"
)

print(
    PREDICTIONS_FILE
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "Model saved successfully:"
)

print(
    MODEL_FILE
)


# ============================================================
# 24. VERIFY SAVED MODEL
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED MODEL")
print("=" * 70)

loaded_model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)

print(
    f"Model type: "
    f"{type(loaded_model).__name__}"
)

print(
    f"Number of features: "
    f"{loaded_model.n_features_in_}"
)

print(
    "Feature names:"
)

for feature in loaded_model.feature_names_in_:

    print(
        f" - {feature}"
    )


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"Training samples : {len(X_train):,}"
)

print(
    f"Testing samples  : {len(X_test):,}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    "\nModel:"
)

print(
    MODEL_FILE
)

print("=" * 70)