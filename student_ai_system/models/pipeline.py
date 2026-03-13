"""
ML Pipeline v2 - Advanced Prediction Engine
Features: Explainability (SHAP-style), Anomaly Detection, What-If Simulation,
          Cohort Comparison, NLP Insights, Academic Health Score
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

SUBJECT_MAP = {
    'Subject_Math': 'Mathematics', 'Subject_Science': 'Science',
    'Subject_English': 'English', 'Subject_Programming': 'Programming',
    'Subject_History': 'History'
}

FEATURE_COLS = ['Attendance', 'Study_Hours', 'Previous_Grades', 'Assignment_Score',
                'Family_Support', 'Internet_Access', 'Financial_Issues',
                'Subject_Math', 'Subject_Science', 'Subject_English',
                'Subject_Programming', 'Subject_History', 'Extra_Curricular', 'Health_Issues']

FEATURE_LABELS = {
    'Attendance': 'Attendance %', 'Study_Hours': 'Study Hours',
    'Previous_Grades': 'Previous Grades', 'Assignment_Score': 'Assignment Score',
    'Family_Support': 'Family Support', 'Internet_Access': 'Internet Access',
    'Financial_Issues': 'Financial Issues', 'Subject_Math': 'Math Score',
    'Subject_Science': 'Science Score', 'Subject_English': 'English Score',
    'Subject_Programming': 'Programming Score', 'Subject_History': 'History Score',
    'Extra_Curricular': 'Extra Curricular', 'Health_Issues': 'Health Issues'
}

RISK_WEIGHTS = {
    'Attendance': -0.28, 'Study_Hours': -0.18, 'Previous_Grades': -0.15,
    'Assignment_Score': -0.12, 'Family_Support': -0.08, 'Internet_Access': -0.04,
    'Financial_Issues': 0.10, 'Subject_Math': -0.07, 'Subject_Science': -0.05,
    'Subject_English': -0.04, 'Subject_Programming': -0.06, 'Subject_History': -0.04,
    'Extra_Curricular': -0.03, 'Health_Issues': 0.06
}

BASELINE = {
    'Attendance': 75, 'Study_Hours': 4, 'Previous_Grades': 65,
    'Assignment_Score': 65, 'Family_Support': 1, 'Internet_Access': 1,
    'Financial_Issues': 0, 'Subject_Math': 65, 'Subject_Science': 65,
    'Subject_English': 65, 'Subject_Programming': 65, 'Subject_History': 65,
    'Extra_Curricular': 0, 'Health_Issues': 0
}


class MLPipeline:
    def __init__(self):
        self.dropout_model = None
        self.perf_model = None
        self.dropout_encoder = None
        self.perf_encoder = None
        self.metadata = {}
        self.anomaly_model = None
        self._loaded = False

    def load_models(self):
        try:
            self.dropout_model = joblib.load(os.path.join(MODELS_DIR, 'dropout_model.pkl'))
            self.perf_model = joblib.load(os.path.join(MODELS_DIR, 'performance_model.pkl'))
            self.dropout_encoder = joblib.load(os.path.join(MODELS_DIR, 'dropout_encoder.pkl'))
            self.perf_encoder = joblib.load(os.path.join(MODELS_DIR, 'performance_encoder.pkl'))
            self.metadata = joblib.load(os.path.join(MODELS_DIR, 'metadata.pkl'))
            self._init_anomaly()
            self._loaded = True
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False

    def _init_anomaly(self):
        np.random.seed(42)
        synth = np.column_stack([
            np.random.randint(50, 100, 600),  # Attendance
            np.random.randint(2, 8, 600),     # Study_Hours
            np.random.randint(50, 95, 600),   # Previous_Grades
            np.random.randint(50, 95, 600),   # Assignment_Score
            np.random.randint(0, 2, 600),     # Family_Support
            np.random.randint(0, 2, 600),     # Internet_Access
            np.random.randint(0, 2, 600),     # Financial_Issues
            np.random.randint(40, 100, 600),  # Subject_Math
            np.random.randint(40, 100, 600),  # Subject_Science
            np.random.randint(40, 100, 600),  # Subject_English
            np.random.randint(40, 100, 600),  # Subject_Programming
            np.random.randint(40, 100, 600),  # Subject_History
            np.random.randint(0, 2, 600),     # Extra_Curricular
            np.random.randint(0, 2, 600),     # Health_Issues
        ])
        self.anomaly_model = IsolationForest(contamination=0.08, random_state=42)
        self.anomaly_model.fit(synth)

    def is_loaded(self):
        return self._loaded

    def _get_strong_weak(self, row):
        scores = {col: float(row.get(col, 0)) for col in SUBJECT_MAP}
        return SUBJECT_MAP[max(scores, key=scores.get)], SUBJECT_MAP[min(scores, key=scores.get)]

    def _health_score(self, row):
        subjects = [float(row.get(c, 0)) for c in SUBJECT_MAP]
        avg_s = sum(subjects) / len(subjects)
        score = (float(row.get('Attendance', 0)) * 0.25 +
                 float(row.get('Study_Hours', 0)) * 4 +
                 float(row.get('Previous_Grades', 0)) * 0.20 +
                 float(row.get('Assignment_Score', 0)) * 0.15 +
                 avg_s * 0.25 +
                 float(row.get('Family_Support', 0)) * 5 -
                 float(row.get('Financial_Issues', 0)) * 8 -
                 float(row.get('Health_Issues', 0)) * 5)
        return max(0, min(100, round(score, 1)))

    def get_shap_explanation(self, row_dict):
        if not self.dropout_model:
            return []
        importances = self.dropout_model.feature_importances_
        contributions = []
        for i, col in enumerate(FEATURE_COLS):
            val = float(row_dict.get(col, 0))
            base = float(BASELINE.get(col, 0))
            weight = RISK_WEIGHTS.get(col, 0)
            importance = float(importances[i])
            if col in ['Attendance', 'Study_Hours', 'Previous_Grades', 'Assignment_Score',
                       'Subject_Math', 'Subject_Science', 'Subject_English',
                       'Subject_Programming', 'Subject_History']:
                deviation = (val - base) / max(base, 1)
                contrib = round(deviation * importance * abs(weight) * 100, 2)
                direction = 'protective' if val >= base else 'risk'
            else:
                contrib = round(val * importance * abs(weight) * 100, 2)
                direction = 'risk' if (val == 1 and weight > 0) else 'protective'
            contributions.append({
                'feature': col, 'label': FEATURE_LABELS[col],
                'value': val, 'contribution': abs(contrib),
                'direction': direction, 'raw': contrib
            })
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        return contributions[:8]

    def _detect_anomaly(self, row_dict):
        try:
            X = np.array([[float(row_dict.get(c, 0)) for c in FEATURE_COLS]])
            return bool(self.anomaly_model.predict(X)[0] == -1)
        except:
            return False

    def predict_single(self, data: dict, include_shap=True) -> dict:
        X = pd.DataFrame([{c: data.get(c, 0) for c in FEATURE_COLS}])
        dropout_enc = self.dropout_model.predict(X)[0]
        perf_enc = self.perf_model.predict(X)[0]
        dropout_pred = self.dropout_encoder.inverse_transform([dropout_enc])[0]
        perf_pred = self.perf_encoder.inverse_transform([perf_enc])[0]
        d_proba = self.dropout_model.predict_proba(X)[0]
        p_proba = self.perf_model.predict_proba(X)[0]
        strong, weak = self._get_strong_weak(data)
        health = self._health_score(data)
        dropout_dist = {c: round(float(p)*100, 1) for c, p in zip(self.dropout_encoder.classes_, d_proba)}
        perf_dist = {c: round(float(p)*100, 1) for c, p in zip(self.perf_encoder.classes_, p_proba)}
        result = {
            'dropout_risk': dropout_pred,
            'performance': perf_pred,
            'strong_subject': strong,
            'weak_subject': weak,
            'confidence': round(float(max(d_proba)) * 100, 1),
            'health_score': health,
            'is_anomaly': self._detect_anomaly(data),
            'dropout_distribution': dropout_dist,
            'performance_distribution': perf_dist,
            'study_plan': generate_study_plan(weak, data.get('Study_Hours', 3), perf_pred),
            'nlp_insight': generate_nlp_insight(data, dropout_pred, perf_pred, weak, strong, health)
        }
        if include_shap:
            result['shap'] = self.get_shap_explanation(data)
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            rd = row.to_dict()
            for col in FEATURE_COLS:
                if col not in rd: rd[col] = 0
            X = pd.DataFrame([{c: rd.get(c, 0) for c in FEATURE_COLS}])
            d_pred = self.dropout_encoder.inverse_transform(self.dropout_model.predict(X))[0]
            p_pred = self.perf_encoder.inverse_transform(self.perf_model.predict(X))[0]
            d_proba = self.dropout_model.predict_proba(X)[0]
            strong, weak = self._get_strong_weak(rd)
            health = self._health_score(rd)
            results.append({
                'Student_ID': rd.get('Student_ID', ''),
                'Student_Name': rd.get('Student_Name', 'N/A'),
                'Attendance': rd.get('Attendance', 0),
                'Study_Hours': rd.get('Study_Hours', 0),
                'Previous_Grades': rd.get('Previous_Grades', 0),
                'Assignment_Score': rd.get('Assignment_Score', 0),
                'Subject_Math': rd.get('Subject_Math', 0),
                'Subject_Science': rd.get('Subject_Science', 0),
                'Subject_English': rd.get('Subject_English', 0),
                'Subject_Programming': rd.get('Subject_Programming', 0),
                'Subject_History': rd.get('Subject_History', 0),
                'Family_Support': rd.get('Family_Support', 0),
                'Financial_Issues': rd.get('Financial_Issues', 0),
                'Health_Issues': rd.get('Health_Issues', 0),
                'Dropout_Risk': d_pred,
                'Performance': p_pred,
                'Strong_Subject': strong,
                'Weak_Subject': weak,
                'Confidence': round(float(max(d_proba)) * 100, 1),
                'Health_Score': health,
                'Is_Anomaly': self._detect_anomaly(rd),
                'AI_Insight': generate_nlp_insight(rd, d_pred, p_pred, weak, strong, health),
                'Study_Plan': generate_study_plan(weak, rd.get('Study_Hours', 3), p_pred)
            })
        return pd.DataFrame(results)

    def get_analytics(self, df: pd.DataFrame) -> dict:
        total = len(df)
        if total == 0: return {}
        att_buckets = {'<50%': 0, '50-70%': 0, '70-85%': 0, '>85%': 0}
        if 'Attendance' in df:
            for v in df['Attendance']:
                v = float(v)
                if v < 50: att_buckets['<50%'] += 1
                elif v < 70: att_buckets['50-70%'] += 1
                elif v < 85: att_buckets['70-85%'] += 1
                else: att_buckets['>85%'] += 1
        return {
            'total': total,
            'dropout_dist': df['Dropout_Risk'].value_counts().to_dict(),
            'perf_dist': df['Performance'].value_counts().to_dict(),
            'weak_subjects': df['Weak_Subject'].value_counts().to_dict(),
            'strong_subjects': df['Strong_Subject'].value_counts().to_dict(),
            'high_risk_count': int((df['Dropout_Risk'] == 'High').sum()),
            'avg_confidence': round(df['Confidence'].mean(), 1),
            'avg_health': round(df['Health_Score'].mean(), 1) if 'Health_Score' in df else 0,
            'anomaly_count': int(df['Is_Anomaly'].sum()) if 'Is_Anomaly' in df else 0,
            'attendance_buckets': att_buckets,
            'accuracy': self.metadata.get('dropout_accuracy', 95.0)
        }

    def whatif_simulate(self, base_data: dict, changes: dict) -> dict:
        modified = {**base_data, **changes}
        orig = self.predict_single(base_data, include_shap=False)
        new = self.predict_single(modified, include_shap=False)
        return {
            'original': orig, 'modified': new,
            'health_delta': round(new['health_score'] - orig['health_score'], 1),
            'risk_changed': orig['dropout_risk'] != new['dropout_risk'],
            'perf_changed': orig['performance'] != new['performance']
        }

    def compare_cohorts(self, df1, df2, label1='Cohort A', label2='Cohort B'):
        r1 = self.predict_batch(df1)
        r2 = self.predict_batch(df2)
        a1 = self.get_analytics(r1)
        a2 = self.get_analytics(r2)
        def pct(a, k): return round(a.get(k, 0) / max(a.get('total', 1), 1) * 100, 1)
        return {
            'label1': label1, 'label2': label2,
            'cohort1': a1, 'cohort2': a2,
            'deltas': {
                'high_risk_pct': round(pct(a2, 'high_risk_count') - pct(a1, 'high_risk_count'), 1),
                'avg_health': round(a2.get('avg_health', 0) - a1.get('avg_health', 0), 1),
                'good_perf_pct': round(
                    a2['perf_dist'].get('Good', 0) / max(a2['total'], 1) * 100 -
                    a1['perf_dist'].get('Good', 0) / max(a1['total'], 1) * 100, 1)
            }
        }


def generate_nlp_insight(row, dropout_risk, performance, weak, strong, health):
    name = str(row.get('Student_Name', 'This student'))
    attendance = float(row.get('Attendance', 0))
    study = float(row.get('Study_Hours', 0))
    financial = int(row.get('Financial_Issues', 0))
    family = int(row.get('Family_Support', 0))
    health_issues = int(row.get('Health_Issues', 0))
    drivers = []
    if attendance < 55: drivers.append("critically low attendance")
    elif attendance < 70: drivers.append("below-average attendance")
    if study < 2: drivers.append("insufficient study hours")
    if financial: drivers.append("financial challenges")
    if not family: drivers.append("limited family support")
    if health_issues: drivers.append("ongoing health concerns")
    risk_sentence = {
        'High': f"{name} is at HIGH dropout risk",
        'Medium': f"{name} shows moderate dropout risk indicators",
        'Low': f"{name} demonstrates stable academic engagement"
    }[dropout_risk]
    if drivers:
        risk_sentence += f", primarily driven by {' and '.join(drivers[:2])}"
    risk_sentence += f". Academic Health Score: {health}/100."
    if dropout_risk == 'High':
        rec = f"Immediate intervention recommended — schedule counseling, address {weak.lower()}, and connect with financial aid if applicable."
    elif dropout_risk == 'Medium':
        rec = f"Encourage increased study time in {weak.lower()} and maintain regular academic check-ins."
    else:
        rec = f"Continue strong performance in {strong.lower()}; consider enrichment activities to further develop {weak.lower()}."
    return risk_sentence + " " + rec


def generate_study_plan(weak_subject, study_hours, performance):
    tips = {
        'Mathematics': ["📐 Practice 20 problems daily — focus on algebra & calculus",
                        "🔢 Use Khan Academy for concept reinforcement",
                        "📊 Dedicate 40% of study time to Math exercises"],
        'Science': ["🔬 Read one science concept with diagrams each day",
                    "🧪 Watch experiment videos to visualize theory",
                    "📖 Create concept maps linking topics"],
        'English': ["📝 Write a 200-word essay daily on random topics",
                    "📚 Read articles and summarize key points",
                    "🗣️ Practice grammar exercises for 30 mins/day"],
        'Programming': ["💻 Code one small program/function daily",
                        "🐛 Debug existing code to sharpen problem-solving",
                        "🎯 Complete coding challenges on HackerRank/LeetCode"],
        'History': ["📜 Read one historical event summary each day",
                    "🗺️ Create timelines for major events",
                    "🧠 Use flashcards for dates and key figures"]
    }
    hours = max(1, int(float(study_hours)))
    perf_advice = {
        'Good': "✅ Maintain consistency. Review weak areas weekly.",
        'Average': "⚠️ Increase daily study by 1 hour. Focus on fundamentals.",
        'Low': "🚨 Urgent: Get tutoring support. Study 5+ hours/day minimum."
    }.get(performance, "")
    plan = f"Focus: {weak_subject} | {hours}h/day schedule\n"
    plan += "\n".join(tips.get(weak_subject, ["Review your weakest subject daily."]))
    plan += f"\n{perf_advice}"
    return plan


pipeline = MLPipeline()
