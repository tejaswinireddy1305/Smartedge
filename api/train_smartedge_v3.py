import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PATHS
# ============================================================

DATASET = r"D:\smart edge\dataset\processed\final_ml_dataset.csv"

MODEL_DIR = r"D:\smart edge\model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "smartedge_v3_random_forest.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMARTEDGE 3.0 MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print("Rows:", len(df))


# ============================================================
# CREATE CACHE VALUE TARGET
# ============================================================

def create_cache_value(x):

    if x == 0:
        return 0

    elif x <= 4:
        return 1

    elif x <= 10:
        return 2

    elif x <= 19:
        return 3

    else:
        return 4


df["cache_value"] = df[
    "future_requests"
].apply(create_cache_value)


# ============================================================
# SHOW DISTRIBUTION
# ============================================================

print("\nCache Value Distribution")

print(
    df["cache_value"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "hour",
    "day_of_week",
    "past_frequency",
    "recency_seconds",
    "content_size"
]

TARGET = "cache_value"


X = df[FEATURES]

y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=16,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# EVALUATION
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "NO_REUSE",
            "LOW_REUSE",
            "MEDIUM_REUSE",
            "HIGH_REUSE",
            "VERY_HIGH_REUSE"
        ]
    )
)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\nFeature Importance")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(

    os.path.join(
        MODEL_DIR,
        "smartedge_v3_feature_importance.csv"
    ),

    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


print("\nModel saved:")
print(MODEL_PATH)


print("\n" + "=" * 70)
print("SMARTEDGE 3.0 TRAINING COMPLETE")
print("=" * 70)