"""
AI MediDetect - Improved Machine Learning Training Script

Trains and evaluates symptom-based disease prediction models.
This is an educational decision-support prototype, not a clinical system.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "symptoms.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "disease_model.pkl"
ENCODER_PATH = MODEL_DIR / "feature_encoder.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
REPORT_PATH = MODEL_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.csv"


# -------------------------------------------------------------------
# Dataset preparation
# -------------------------------------------------------------------

def load_and_validate_dataset():
    """Load symptoms.csv and make the input safe for model training."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Create data/symptoms.csv before running training."
        )

    df = pd.read_csv(DATA_PATH)

    if "disease" not in df.columns:
        raise ValueError(
            "data/symptoms.csv must contain a column named 'disease'."
        )

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    if "disease" not in df.columns:
        raise ValueError(
            "The disease column was not found after column cleaning."
        )

    df["disease"] = df["disease"].astype(str).str.strip()

    df = df.dropna(subset=["disease"])
    df = df[df["disease"] != ""].copy()

    symptom_columns = [
        column for column in df.columns if column != "disease"
    ]

    if not symptom_columns:
        raise ValueError(
            "No symptom feature columns found in symptoms.csv."
        )

    X = df[symptom_columns].copy()

    # Convert symptom values safely into 0 or 1.
    for column in symptom_columns:
        X[column] = (
            pd.to_numeric(X[column], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype(int)
        )

    y = df["disease"].copy()

    class_counts = y.value_counts()
    too_small_classes = class_counts[class_counts < 2]

    if not too_small_classes.empty:
        details = ", ".join(
            f"{disease}: {count}"
            for disease, count in too_small_classes.items()
        )
        raise ValueError(
            "Each disease needs at least 2 dataset rows for stratified "
            f"train/test splitting. Problem classes: {details}"
        )

    return X, y, symptom_columns, class_counts


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------

def determine_cv_folds(class_counts):
    """
    Use up to 5 folds. The smallest class decides the maximum possible
    number of stratified folds.
    """
    minimum_class_count = int(class_counts.min())
    return min(5, minimum_class_count)


def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder,
    cv_folds,
):
    """Train, cross-validate, test, and return model metrics."""

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    weighted_f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1_values, support = (
        precision_recall_fscore_support(
            y_test,
            test_predictions,
            labels=np.arange(len(label_encoder.classes_)),
            zero_division=0,
        )
    )

    cross_validation_scores = []

    if cv_folds >= 2:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )

        cross_validation_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=splitter,
            scoring="f1_weighted",
            n_jobs=-1,
        ).tolist()

    report = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=np.arange(len(label_encoder.classes_)),
    )

    class_metrics = []

    for index, disease_name in enumerate(label_encoder.classes_):
        class_metrics.append(
            {
                "disease": str(disease_name),
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1_score": round(float(f1_values[index]), 4),
                "support": int(support[index]),
            }
        )

    metrics = {
        "model_name": model_name,
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "cross_validation_f1_scores": [
            round(float(score), 4)
            for score in cross_validation_scores
        ],
        "mean_cross_validation_f1": (
            round(float(np.mean(cross_validation_scores)), 4)
            if cross_validation_scores
            else None
        ),
        "class_metrics": class_metrics,
    }

    return metrics, report, cm


# -------------------------------------------------------------------
# Save files
# -------------------------------------------------------------------

def save_evaluation_files(metrics, report, confusion_matrix_data, label_encoder):
    """Save metrics for project documentation and viva presentation."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    confusion_df = pd.DataFrame(
        confusion_matrix_data,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    confusion_df.index.name = "Actual Disease"
    confusion_df.columns.name = "Predicted Disease"
    confusion_df.to_csv(CONFUSION_MATRIX_PATH)


def print_metrics(metrics):
    """Print concise and useful output in the terminal."""

    print(f"\nSelected Model: {metrics['model_name']}")
    print(f"Training Accuracy: {metrics['train_accuracy'] * 100:.2f}%")
    print(f"Testing Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Weighted F1-score: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Macro F1-score: {metrics['macro_f1'] * 100:.2f}%")

    if metrics["mean_cross_validation_f1"] is not None:
        print(
            "Mean Cross-validation Weighted F1-score: "
            f"{metrics['mean_cross_validation_f1'] * 100:.2f}%"
        )


# -------------------------------------------------------------------
# Main training flow
# -------------------------------------------------------------------

def train_model():
    print("=" * 65)
    print("AI MEDIDETECT - IMPROVED MODEL TRAINING")
    print("=" * 65)

    print("\n[1/7] Loading and validating the dataset...")
    X, y, symptom_columns, class_counts = load_and_validate_dataset()

    print(f"Dataset rows: {len(X)}")
    print(f"Symptoms/features: {len(symptom_columns)}")
    print(f"Disease classes: {y.nunique()}")

    print("\nDisease distribution:")
    for disease, count in class_counts.items():
        print(f"  - {disease}: {count}")

    print("\n[2/7] Encoding disease labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\n[3/7] Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    cv_folds = determine_cv_folds(class_counts)

    if cv_folds < 2:
        print(
            "\nWarning: Cross-validation skipped because the dataset "
            "does not have enough examples in each disease class."
        )
    else:
        print(f"\n[4/7] Using {cv_folds}-fold stratified cross-validation...")

    candidate_models = {
        "Bernoulli Naive Bayes": BernoulliNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    all_results = {}

    print("\n[5/7] Training and evaluating candidate models...")

    for model_name, candidate_model in candidate_models.items():
        print(f"\nEvaluating: {model_name}")

        metrics, report, cm = evaluate_model(
            candidate_model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
            label_encoder,
            cv_folds,
        )

        all_results[model_name] = {
            "model": candidate_model,
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm,
        }

        print_metrics(metrics)

    best_model_name = max(
        all_results,
        key=lambda name: (
            all_results[name]["metrics"]["weighted_f1"],
            all_results[name]["metrics"]["test_accuracy"],
        ),
    )

    best_result = all_results[best_model_name]
    best_model = best_result["model"]
    best_metrics = best_result["metrics"]

    print("\n[6/7] Selecting the best model...")
    print(f"Best model selected: {best_model_name}")

    print("\nRefitting selected model on the complete dataset...")
    best_model.fit(X, y_encoded)

    print("\n[7/7] Saving model, encoder, and evaluation files...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(best_model, file)

    with ENCODER_PATH.open("wb") as file:
        pickle.dump(
            {
                "label_encoder": label_encoder,
                "symptom_columns": symptom_columns,
            },
            file,
        )

    save_evaluation_files(
        best_metrics,
        best_result["report"],
        best_result["confusion_matrix"],
        label_encoder,
    )

    print("\nSaved files:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {ENCODER_PATH}")
    print(f"  - {METRICS_PATH}")
    print(f"  - {REPORT_PATH}")
    print(f"  - {CONFUSION_MATRIX_PATH}")

    print("\n" + "=" * 65)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nImportant interpretation:")
    print(
        "- Do not use training accuracy as proof of real-world accuracy."
    )
    print(
        "- Report test accuracy, weighted F1-score, and cross-validation F1."
    )
    print(
        "- If your dataset has repeated or artificially generated rows, "
        "high scores may not reflect clinical reliability."
    )

    return best_model, label_encoder, best_metrics


if __name__ == "__main__":
    train_model()