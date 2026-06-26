"""
evaluate_models.py
Loads the saved XGBoost model and runs a detailed evaluation.
Run: python models/evaluate_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

BASE   = os.path.join(os.path.dirname(__file__), '..')
MODEL  = joblib.load(os.path.join(BASE, 'models', 'model.pkl'))
SCALER = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
FEATS  = joblib.load(os.path.join(BASE, 'models', 'feature_names.pkl'))
OUT    = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

# ── Load & Prepare Data ───────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'data', 'student_data.csv'))
df = df.drop(columns=['StudentID', 'FinalMarks', 'FinalGrade'])
for col in ['Gender', 'SocioeconomicStatus', 'ParentEducation']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[FEATS]
y = LabelEncoder().fit_transform(df['Result'])

X_scaled = SCALER.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

y_pred  = MODEL.predict(X_test)
y_proba = MODEL.predict_proba(X_test)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  DETAILED EVALUATION – XGBoost (Best Model)")
print("=" * 55)
print(f"  Accuracy   : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  ROC-AUC    : {roc_auc_score(y_test, y_proba):.4f}")
cv = cross_val_score(MODEL, X_scaled, y, cv=10, scoring='accuracy')
print(f"  10-Fold CV : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost Model – Detailed Evaluation", fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
            linewidths=0.5, linecolor='white', cbar=False)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score    = roc_auc_score(y_test, y_proba)
axes[1].plot(fpr, tpr, color='#2E74B5', lw=2, label=f'AUC = {auc_score:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.02])
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve"); axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
print(f"\nEvaluation plots saved → outputs/evaluation_plots.png")
