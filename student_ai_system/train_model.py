"""
ML Training Script - Student Performance & Dropout Risk Prediction
Run this script once to generate the trained models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SUBJECT_COLS = ['Subject_Math', 'Subject_Science', 'Subject_English', 'Subject_Programming', 'Subject_History']
FEATURE_COLS = ['Attendance', 'Study_Hours', 'Previous_Grades', 'Assignment_Score',
                'Family_Support', 'Internet_Access', 'Financial_Issues',
                'Subject_Math', 'Subject_Science', 'Subject_English',
                'Subject_Programming', 'Subject_History', 'Extra_Curricular', 'Health_Issues']


def generate_labels(df):
    """Auto-generate labels from features for training."""
    dropout_risk = []
    performance = []

    for _, row in df.iterrows():
        avg_score = (row['Subject_Math'] + row['Subject_Science'] + row['Subject_English'] +
                     row['Subject_Programming'] + row['Subject_History']) / 5

        # Dropout risk based on multiple factors
        risk_score = 0
        if row['Attendance'] < 50: risk_score += 3
        elif row['Attendance'] < 70: risk_score += 1
        if row['Study_Hours'] < 2: risk_score += 2
        elif row['Study_Hours'] < 4: risk_score += 1
        if row['Financial_Issues'] == 1: risk_score += 2
        if row['Family_Support'] == 0: risk_score += 1
        if row['Health_Issues'] == 1: risk_score += 1
        if avg_score < 40: risk_score += 2
        elif avg_score < 60: risk_score += 1

        if risk_score >= 5:
            dropout_risk.append('High')
        elif risk_score >= 3:
            dropout_risk.append('Medium')
        else:
            dropout_risk.append('Low')

        # Performance based on grades
        perf_score = (avg_score * 0.4 + row['Previous_Grades'] * 0.3 +
                      row['Assignment_Score'] * 0.2 + row['Attendance'] * 0.1)
        if perf_score >= 75:
            performance.append('Good')
        elif perf_score >= 55:
            performance.append('Average')
        else:
            performance.append('Low')

    df['Dropout_Risk'] = dropout_risk
    df['Performance'] = performance
    return df


def generate_synthetic_data(n=2000):
    """Generate synthetic training data."""
    np.random.seed(42)
    data = {
        'Attendance': np.random.randint(20, 100, n),
        'Study_Hours': np.random.randint(0, 10, n),
        'Previous_Grades': np.random.randint(20, 100, n),
        'Assignment_Score': np.random.randint(15, 100, n),
        'Family_Support': np.random.randint(0, 2, n),
        'Internet_Access': np.random.randint(0, 2, n),
        'Financial_Issues': np.random.randint(0, 2, n),
        'Subject_Math': np.random.randint(20, 100, n),
        'Subject_Science': np.random.randint(20, 100, n),
        'Subject_English': np.random.randint(20, 100, n),
        'Subject_Programming': np.random.randint(20, 100, n),
        'Subject_History': np.random.randint(20, 100, n),
        'Extra_Curricular': np.random.randint(0, 2, n),
        'Health_Issues': np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    return generate_labels(df)


def train_models():
    print("=" * 60)
    print("  Student AI - ML Model Training")
    print("=" * 60)

    # Try loading real dataset first, else use synthetic
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'student_dataset.csv')
    if os.path.exists(dataset_path):
        real_df = pd.read_csv(dataset_path)
        real_df = generate_labels(real_df)
        synth_df = generate_synthetic_data(1800)
        df = pd.concat([real_df[FEATURE_COLS + ['Dropout_Risk', 'Performance']], synth_df], ignore_index=True)
        print(f"Loaded real dataset + synthetic data. Total: {len(df)} samples")
    else:
        df = generate_synthetic_data(2000)
        print(f"Using synthetic data. Total: {len(df)} samples")

    X = df[FEATURE_COLS]

    le_dropout = LabelEncoder()
    le_perf = LabelEncoder()
    y_dropout = le_dropout.fit_transform(df['Dropout_Risk'])
    y_perf = le_perf.fit_transform(df['Performance'])

    print(f"\nDropout classes: {le_dropout.classes_}")
    print(f"Performance classes: {le_perf.classes_}")

    # Train Dropout Risk Model
    X_train, X_test, y_train, y_test = train_test_split(X, y_dropout, test_size=0.2, random_state=42)
    dropout_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    dropout_model.fit(X_train, y_train)
    dropout_acc = accuracy_score(y_test, dropout_model.predict(X_test))
    print(f"\nDropout Risk Model Accuracy: {dropout_acc:.4f} ({dropout_acc*100:.1f}%)")

    # Train Performance Model
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_perf, test_size=0.2, random_state=42)
    perf_model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    perf_model.fit(X_train2, y_train2)
    perf_acc = accuracy_score(y_test2, perf_model.predict(X_test2))
    print(f"Performance Model Accuracy: {perf_acc:.4f} ({perf_acc*100:.1f}%)")

    # Save models and encoders
    joblib.dump(dropout_model, os.path.join(MODELS_DIR, 'dropout_model.pkl'))
    joblib.dump(perf_model, os.path.join(MODELS_DIR, 'performance_model.pkl'))
    joblib.dump(le_dropout, os.path.join(MODELS_DIR, 'dropout_encoder.pkl'))
    joblib.dump(le_perf, os.path.join(MODELS_DIR, 'performance_encoder.pkl'))

    # Save accuracy metadata
    metadata = {
        'dropout_accuracy': round(dropout_acc * 100, 1),
        'performance_accuracy': round(perf_acc * 100, 1),
        'feature_cols': FEATURE_COLS,
        'subject_cols': SUBJECT_COLS
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'metadata.pkl'))

    print("\n✓ Models saved successfully to /models/")
    print("  - dropout_model.pkl")
    print("  - performance_model.pkl")
    print("  - dropout_encoder.pkl")
    print("  - performance_encoder.pkl")
    print("  - metadata.pkl")
    print("\nTraining complete! Ready to run app.py")
    return metadata


if __name__ == '__main__':
    train_models()
