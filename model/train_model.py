"""
AI MediDetect - Machine Learning Model Training Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_model():
    print("="*60)
    print("AI MediDetect - Model Training")
    print("="*60)

    print("\n[1/6] Loading dataset...")
    data_path = os.path.join('data', 'symptoms.csv')
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")

    print("\n[2/6] Preparing features and labels...")
    symptom_columns = [col for col in df.columns if col != 'disease']
    X = df[symptom_columns]
    y = df['disease']
    print(f"Features: {len(symptom_columns)} symptoms")
    print(f"Classes: {y.nunique()} diseases")

    print("\n[3/6] Encoding disease labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    encoder_path = os.path.join('model', 'feature_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump({'label_encoder': label_encoder, 'symptom_columns': symptom_columns}, f)
    print(f"Label encoder saved to {encoder_path}")

    print("\n[4/6] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\n[5/6] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model training completed!")

    print("\n[6/6] Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy*100:.2f}%")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    print("\nSaving model...")
    model_path = os.path.join('model', 'disease_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

    return model, label_encoder

if __name__ == "__main__":
    model, encoder = train_model()
