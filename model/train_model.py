"""
AI MediDetect - Disease Model Training (Improved)

Educational project only. It is not a diagnostic or prescription system.

20 samples per disease = 140 total samples for better generalization.
"""

import pickle
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

MODEL_PATH = BASE_DIR / "disease_model_improved.pkl"
ENCODER_PATH = BASE_DIR / "feature_encoder_improved.pkl"


ALL_SYMPTOMS = [
    "fever",
    "headache",
    "cough",
    "fatigue",
    "nausea",
    "vomiting",
    "sore_throat",
    "body_pain",
    "chest_pain",
    "shortness_of_breath",
    "dizziness",
    "joint_pain",
    "runny_nose",
    "chills",
    "loss_of_appetite",
    "muscle_pain",
    "weakness",
    "sneezing",
    "wheezing",
]


SYMPTOM_ALIASES = {
    "high fever": "fever",
    "high_fever": "fever",
    "shortness of breath": "shortness_of_breath",
    "sore throat": "sore_throat",
    "body pain": "body_pain",
    "chest pain": "chest_pain",
    "joint pain": "joint_pain",
    "runny nose": "runny_nose",
    "loss of appetite": "loss_of_appetite",
    "muscle pain": "muscle_pain",
}


TRAINING_DATA = [
    # --- COMMON COLD (20 samples) ---
    {"symptoms": ["fever", "cough", "runny_nose", "sore_throat", "fatigue"], "disease": "common cold"},
    {"symptoms": ["cough", "sore_throat", "runny_nose", "headache"], "disease": "common cold"},
    {"symptoms": ["runny_nose", "sneezing", "sore_throat"], "disease": "common cold"},
    {"symptoms": ["fever", "cough", "fatigue", "body_pain"], "disease": "common cold"},
    {"symptoms": ["runny_nose", "cough", "fatigue", "sore_throat"], "disease": "common cold"},
    {"symptoms": ["sore_throat", "cough", "sneezing", "headache"], "disease": "common cold"},
    {"symptoms": ["fever", "runny_nose", "sore_throat", "weakness"], "disease": "common cold"},
    {"symptoms": ["cough", "runny_nose", "headache", "fatigue"], "disease": "common cold"},
    {"symptoms": ["sneezing", "runny_nose", "sore_throat", "cough"], "disease": "common cold"},
    {"symptoms": ["fever", "sore_throat", "cough", "chills"], "disease": "common cold"},
    {"symptoms": ["runny_nose", "headache", "fatigue", "sneezing"], "disease": "common cold"},
    {"symptoms": ["cough", "sore_throat", "fatigue"], "disease": "common cold"},
    {"symptoms": ["fever", "cough", "runny_nose", "weakness"], "disease": "common cold"},
    {"symptoms": ["sore_throat", "runny_nose", "sneezing"], "disease": "common cold"},
    {"symptoms": ["headache", "cough", "sore_throat", "fatigue"], "disease": "common cold"},
    {"symptoms": ["fever", "fatigue", "cough", "body_pain"], "disease": "common cold"},
    {"symptoms": ["runny_nose", "cough", "headache"], "disease": "common cold"},
    {"symptoms": ["sore_throat", "sneezing", "runny_nose", "cough"], "disease": "common cold"},
    {"symptoms": ["cough", "fatigue", "runny_nose", "chills"], "disease": "common cold"},
    {"symptoms": ["fever", "sore_throat", "headache", "fatigue"], "disease": "common cold"},

    # --- VIRAL FEVER (20 samples) ---
    {"symptoms": ["fever", "body_pain", "headache", "chills", "fatigue"], "disease": "viral fever"},
    {"symptoms": ["fever", "nausea", "vomiting", "body_pain"], "disease": "viral fever"},
    {"symptoms": ["fever", "muscle_pain", "joint_pain", "fatigue"], "disease": "viral fever"},
    {"symptoms": ["fever", "headache", "weakness", "chills"], "disease": "viral fever"},
    {"symptoms": ["high fever", "body_pain", "headache", "nausea"], "disease": "viral fever"},
    {"symptoms": ["fever", "chills", "muscle_pain", "fatigue"], "disease": "viral fever"},
    {"symptoms": ["fever", "vomiting", "body_pain", "weakness"], "disease": "viral fever"},
    {"symptoms": ["fever", "headache", "nausea", "fatigue"], "disease": "viral fever"},
    {"symptoms": ["fever", "joint_pain", "muscle_pain", "chills"], "disease": "viral fever"},
    {"symptoms": ["fever", "body_pain", "loss_of_appetite", "weakness"], "disease": "viral fever"},
    {"symptoms": ["fever", "headache", "chills", "dizziness"], "disease": "viral fever"},
    {"symptoms": ["high fever", "fatigue", "body_pain", "vomiting"], "disease": "viral fever"},
    {"symptoms": ["fever", "weakness", "muscle_pain", "headache"], "disease": "viral fever"},
    {"symptoms": ["fever", "nausea", "headache", "chills"], "disease": "viral fever"},
    {"symptoms": ["fever", "body_pain", "fatigue", "loss_of_appetite"], "disease": "viral fever"},
    {"symptoms": ["fever", "vomiting", "weakness", "joint_pain"], "disease": "viral fever"},
    {"symptoms": ["fever", "headache", "muscle_pain", "nausea"], "disease": "viral fever"},
    {"symptoms": ["fever", "chills", "headache", "body_pain"], "disease": "viral fever"},
    {"symptoms": ["fever", "fatigue", "body_pain", "dizziness"], "disease": "viral fever"},
    {"symptoms": ["fever", "nausea", "chills", "fatigue"], "disease": "viral fever"},

    # --- PNEUMONIA (20 samples) ---
    {"symptoms": ["chest_pain", "cough", "fever", "shortness_of_breath", "fatigue"], "disease": "pneumonia"},
    {"symptoms": ["cough", "shortness_of_breath", "chest_pain", "fever"], "disease": "pneumonia"},
    {"symptoms": ["fever", "cough", "body_pain", "shortness_of_breath"], "disease": "pneumonia"},
    {"symptoms": ["fatigue", "shortness_of_breath", "cough", "chest_pain"], "disease": "pneumonia"},
    {"symptoms": ["cough", "fever", "chest_pain", "weakness"], "disease": "pneumonia"},
    {"symptoms": ["shortness_of_breath", "cough", "fever", "body_pain"], "disease": "pneumonia"},
    {"symptoms": ["chest_pain", "cough", "fatigue", "fever"], "disease": "pneumonia"},
    {"symptoms": ["cough", "shortness_of_breath", "fever", "headache"], "disease": "pneumonia"},
    {"symptoms": ["fever", "chest_pain", "cough", "nausea"], "disease": "pneumonia"},
    {"symptoms": ["cough", "body_pain", "fever", "shortness_of_breath"], "disease": "pneumonia"},
    {"symptoms": ["shortness_of_breath", "chest_pain", "fatigue", "cough"], "disease": "pneumonia"},
    {"symptoms": ["fever", "cough", "weakness", "chest_pain"], "disease": "pneumonia"},
    {"symptoms": ["cough", "fever", "shortness_of_breath", "chills"], "disease": "pneumonia"},
    {"symptoms": ["chest_pain", "fatigue", "cough", "shortness_of_breath"], "disease": "pneumonia"},
    {"symptoms": ["fever", "cough", "body_pain", "weakness"], "disease": "pneumonia"},
    {"symptoms": ["shortness_of_breath", "cough", "chest_pain", "fatigue"], "disease": "pneumonia"},
    {"symptoms": ["cough", "fever", "headache", "chest_pain"], "disease": "pneumonia"},
    {"symptoms": ["fever", "shortness_of_breath", "cough", "body_pain"], "disease": "pneumonia"},
    {"symptoms": ["cough", "chest_pain", "fever", "fatigue"], "disease": "pneumonia"},
    {"symptoms": ["shortness_of_breath", "fever", "cough", "nausea"], "disease": "pneumonia"},

    # --- ASTHMA (20 samples) ---
    {"symptoms": ["shortness_of_breath", "chest_pain", "cough"], "disease": "asthma"},
    {"symptoms": ["shortness_of_breath", "wheezing", "fatigue"], "disease": "asthma"},
    {"symptoms": ["cough", "chest_pain", "shortness_of_breath"], "disease": "asthma"},
    {"symptoms": ["wheezing", "shortness_of_breath", "cough"], "disease": "asthma"},
    {"symptoms": ["chest_pain", "wheezing", "shortness_of_breath"], "disease": "asthma"},
    {"symptoms": ["cough", "wheezing", "fatigue", "chest_pain"], "disease": "asthma"},
    {"symptoms": ["shortness_of_breath", "fatigue", "cough"], "disease": "asthma"},
    {"symptoms": ["wheezing", "cough", "shortness_of_breath"], "disease": "asthma"},
    {"symptoms": ["chest_pain", "cough", "wheezing"], "disease": "asthma"},
    {"symptoms": ["shortness_of_breath", "chest_pain", "fatigue"], "disease": "asthma"},
    {"symptoms": ["cough", "shortness_of_breath", "wheezing"], "disease": "asthma"},
    {"symptoms": ["wheezing", "chest_pain", "fatigue"], "disease": "asthma"},
    {"symptoms": ["shortness_of_breath", "cough", "fatigue"], "disease": "asthma"},
    {"symptoms": ["chest_pain", "shortness_of_breath", "cough"], "disease": "asthma"},
    {"symptoms": ["cough", "fatigue", "wheezing"], "disease": "asthma"},
    {"symptoms": ["shortness_of_breath", "wheezing", "chest_pain"], "disease": "asthma"},
    {"symptoms": ["wheezing", "fatigue", "cough"], "disease": "asthma"},
    {"symptoms": ["cough", "chest_pain", "fatigue"], "disease": "asthma"},
    {"symptoms": ["shortness_of_breath", "wheezing", "body_pain"], "disease": "asthma"},
    {"symptoms": ["chest_pain", "fatigue", "shortness_of_breath"], "disease": "asthma"},

    # --- HEART DISEASE (20 samples) ---
    {"symptoms": ["chest_pain", "shortness_of_breath", "fatigue", "dizziness"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "body_pain", "nausea"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "shortness_of_breath", "fatigue"], "disease": "heart disease"},
    {"symptoms": ["dizziness", "chest_pain", "shortness_of_breath"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "nausea", "dizziness"], "disease": "heart disease"},
    {"symptoms": ["shortness_of_breath", "chest_pain", "body_pain"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "fatigue", "dizziness"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "nausea", "shortness_of_breath"], "disease": "heart disease"},
    {"symptoms": ["dizziness", "shortness_of_breath", "chest_pain"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "weakness", "fatigue"], "disease": "heart disease"},
    {"symptoms": ["shortness_of_breath", "fatigue", "dizziness"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "body_pain", "dizziness"], "disease": "heart disease"},
    {"symptoms": ["nausea", "chest_pain", "fatigue"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "shortness_of_breath", "nausea"], "disease": "heart disease"},
    {"symptoms": ["dizziness", "fatigue", "chest_pain"], "disease": "heart disease"},
    {"symptoms": ["shortness_of_breath", "chest_pain", "dizziness"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "fatigue", "body_pain"], "disease": "heart disease"},
    {"symptoms": ["nausea", "dizziness", "chest_pain"], "disease": "heart disease"},
    {"symptoms": ["chest_pain", "weakness", "shortness_of_breath"], "disease": "heart disease"},
    {"symptoms": ["fatigue", "chest_pain", "dizziness"], "disease": "heart disease"},

    # --- DIABETES (20 samples) ---
    {"symptoms": ["fatigue", "nausea", "loss_of_appetite", "dizziness"], "disease": "diabetes"},
    {"symptoms": ["fatigue", "dizziness", "headache"], "disease": "diabetes"},
    {"symptoms": ["nausea", "fatigue", "body_pain"], "disease": "diabetes"},
    {"symptoms": ["loss_of_appetite", "fatigue", "weakness"], "disease": "diabetes"},
    {"symptoms": ["fatigue", "nausea", "dizziness"], "disease": "diabetes"},
    {"symptoms": ["dizziness", "headache", "fatigue"], "disease": "diabetes"},
    {"symptoms": ["weakness", "fatigue", "nausea"], "disease": "diabetes"},
    {"symptoms": ["loss_of_appetite", "nausea", "fatigue"], "disease": "diabetes"},
    {"symptoms": ["fatigue", "body_pain", "headache"], "disease": "diabetes"},
    {"symptoms": ["dizziness", "fatigue", "nausea"], "disease": "diabetes"},
    {"symptoms": ["headache", "fatigue", "dizziness"], "disease": "diabetes"},
    {"symptoms": ["nausea", "loss_of_appetite", "weakness"], "disease": "diabetes"},
    {"symptoms": ["fatigue", "weakness", "loss_of_appetite"], "disease": "diabetes"},
    {"symptoms": ["body_pain", "fatigue", "nausea"], "disease": "diabetes"},
    {"symptoms": ["dizziness", "weakness", "fatigue"], "disease": "diabetes"},
    {"symptoms": ["loss_of_appetite", "dizziness", "headache"], "disease": "diabetes"},
    {"symptoms": ["fatigue", "headache", "body_pain"], "disease": "diabetes"},
    {"symptoms": ["nausea", "fatigue", "dizziness"], "disease": "diabetes"},
    {"symptoms": ["weakness", "headache", "fatigue"], "disease": "diabetes"},
    {"symptoms": ["fatigue", "dizziness", "loss_of_appetite"], "disease": "diabetes"},

    # --- ALLERGIC RHINITIS (20 samples) ---
    {"symptoms": ["runny_nose", "sore_throat", "cough", "headache"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "headache", "fatigue"], "disease": "allergic rhinitis"},
    {"symptoms": ["sore_throat", "cough", "runny_nose"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "sneezing", "headache"], "disease": "allergic rhinitis"},
    {"symptoms": ["cough", "runny_nose", "headache"], "disease": "allergic rhinitis"},
    {"symptoms": ["sneezing", "runny_nose", "sore_throat"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "fatigue", "cough"], "disease": "allergic rhinitis"},
    {"symptoms": ["headache", "sneezing", "runny_nose"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "sore_throat", "sneezing"], "disease": "allergic rhinitis"},
    {"symptoms": ["cough", "sneezing", "runny_nose"], "disease": "allergic rhinitis"},
    {"symptoms": ["sore_throat", "headache", "runny_nose"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "cough", "sneezing"], "disease": "allergic rhinitis"},
    {"symptoms": ["headache", "runny_nose", "fatigue"], "disease": "allergic rhinitis"},
    {"symptoms": ["sneezing", "sore_throat", "runny_nose"], "disease": "allergic rhinitis"},
    {"symptoms": ["cough", "headache", "runny_nose"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "sneezing", "fatigue"], "disease": "allergic rhinitis"},
    {"symptoms": ["sore_throat", "runny_nose", "headache"], "disease": "allergic rhinitis"},
    {"symptoms": ["runny_nose", "fatigue", "headache"], "disease": "allergic rhinitis"},
    {"symptoms": ["cough", "runny_nose", "sore_throat"], "disease": "allergic rhinitis"},
    {"symptoms": ["sneezing", "headache", "runny_nose"], "disease": "allergic rhinitis"},
]


DISEASE_HOSPITAL_MAP = {
    "common cold": {
        "priority": "general",
        "specialties": ["general practice", "clinic", "urgent care"],
        "urgency": "low",
        "advice": "Visit a general clinic or see a doctor if symptoms persist or worsen.",
    },
    "viral fever": {
        "priority": "general",
        "specialties": ["general hospital", "fever clinic", "clinic"],
        "urgency": "low",
        "advice": "Consider a clinic or general hospital assessment, especially if fever persists.",
    },
    "pneumonia": {
        "priority": "high",
        "specialties": ["chest hospital", "pulmonology", "general hospital"],
        "urgency": "high",
        "advice": "Seek prompt medical assessment, particularly for breathing difficulty or chest pain.",
    },
    "asthma": {
        "priority": "medium",
        "specialties": ["respiratory clinic", "pulmonology", "general hospital"],
        "urgency": "medium",
        "advice": "Consult a respiratory specialist or clinician for assessment.",
    },
    "heart disease": {
        "priority": "critical",
        "specialties": ["cardiology", "cardiac hospital", "emergency"],
        "urgency": "critical",
        "advice": "For chest pain or breathing difficulty, seek emergency medical care immediately.",
    },
    "diabetes": {
        "priority": "medium",
        "specialties": ["endocrinology", "general hospital", "clinic"],
        "urgency": "medium",
        "advice": "Consult a clinician for appropriate blood-sugar testing and management.",
    },
    "allergic rhinitis": {
        "priority": "low",
        "specialties": ["clinic", "general practice", "ENT"],
        "urgency": "low",
        "advice": "Visit a clinic or ENT specialist if symptoms are persistent.",
    },
}


def normalize_symptom(symptom):
    symptom = str(symptom).strip().lower()
    return SYMPTOM_ALIASES.get(symptom, symptom)


def build_training_data():
    rows = []
    labels = []

    for record in TRAINING_DATA:
        normalized_symptoms = {
            normalize_symptom(symptom)
            for symptom in record["symptoms"]
        }

        unknown_symptoms = normalized_symptoms - set(ALL_SYMPTOMS)
        if unknown_symptoms:
            print(
                f"Warning: Ignoring unknown symptoms for "
                f"{record['disease']}: {sorted(unknown_symptoms)}"
            )

        row = {
            symptom: int(symptom in normalized_symptoms)
            for symptom in ALL_SYMPTOMS
        }

        rows.append(row)
        labels.append(record["disease"])

    return pd.DataFrame(rows, columns=ALL_SYMPTOMS), np.array(labels)


def train_model():
    print("=" * 70)
    print("AI MediDetect - Model Training (20 samples per disease)")
    print("=" * 70)

    X, y = build_training_data()

    print(f"\n📊 Dataset:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(ALL_SYMPTOMS)}")
    print(f"   Diseases: {len(np.unique(y))}")
    print(f"   Classes: {sorted(np.unique(y))}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n📈 Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")

    # Tuned hyperparameters for ~140 samples
    model = RandomForestClassifier(
        n_estimators=100,          # ↓ from 300 (smaller dataset)
        max_depth=6,               # ↓ from 10 (prevent overfitting)
        min_samples_split=3,       # ↑ from 2 (require more samples per split)
        min_samples_leaf=2,        # ↑ from 1 (smooth leaves)
        class_weight="balanced",
        random_state=42,
    )

    print(f"\n🔧 Model hyperparameters:")
    print(f"   n_estimators: 100")
    print(f"   max_depth: 6")
    print(f"   min_samples_split: 3")
    print(f"   min_samples_leaf: 2")

    model.fit(X_train, y_train)

    # Evaluate on training set
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"\n✅ Performance:")
    print(f"   Training accuracy: {train_accuracy:.2%}")
    print(f"   Validation accuracy: {val_accuracy:.2%}")
    print(f"   Overfitting gap: {(train_accuracy - val_accuracy):.2%}")

    print(f"\n📋 Validation classification report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print(f"\n🔀 Confusion matrix (validation):")
    cm = confusion_matrix(y_val, val_predictions)
    print(cm)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    encoder_data = {
        "label_encoder": label_encoder,
        "symptom_columns": ALL_SYMPTOMS,
        "disease_hospital_map": DISEASE_HOSPITAL_MAP,
        "metadata": {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "model_type": "RandomForestClassifier",
            "training_accuracy": float(train_accuracy),
            "validation_accuracy": float(val_accuracy),
            "total_samples": len(X),
            "samples_per_disease": 20,
        },
    }

    with ENCODER_PATH.open("wb") as file:
        pickle.dump(encoder_data, file)

    print(f"\n💾 Saved:")
    print(f"   Model: {MODEL_PATH.name}")
    print(f"   Encoder: {ENCODER_PATH.name}")

    print(f"\n📍 Model details:")
    print(f"   Feature names: {list(model.feature_names_in_)[:5]}... ({len(model.feature_names_in_)} total)")
    print(f"   Classes: {label_encoder.classes_}")

    print("\n" + "=" * 70)
    print("✨ Training complete. Ready to run app.py")
    print("=" * 70)

    return model, encoder_data


if __name__ == "__main__":
    train_model()