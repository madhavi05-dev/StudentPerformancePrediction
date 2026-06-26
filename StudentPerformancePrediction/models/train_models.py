"""
train_models.py
Trains all ML models for Student Performance Prediction.
Saves the best model (XGBoost) as models/model.pkl
Run: python models/train_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier

# ── 1. Load Data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_data.csv')
if not os.path.exists(DATA_PATH):
    print("Dataset not found. Generating it first...")
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    exec(open('data/generate_dataset.py').read())

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(3))

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
df = df.drop(columns=['StudentID', 'FinalMarks', 'FinalGrade'])

le = LabelEncoder()
for col in ['Gender', 'SocioeconomicStatus', 'ParentEducation']:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Result'])
y = LabelEncoder().fit_transform(df['Result'])   # Pass=1, Fail=0

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 3. Define Models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression"  : LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree"        : DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)"            : SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    "KNN"                  : KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost"              : XGBClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=6, use_label_encoder=False,
                                           eval_metric='logloss', random_state=42),
}

# ── 4. Train & Evaluate ───────────────────────────────────────────────────────
results = {}
print("\n" + "="*65)
print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>8} {'CV Mean':>10}")
print("="*65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    cv  = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy').mean()

    results[name] = {'accuracy': acc, 'auc': auc, 'cv_mean': cv, 'model': model}
    print(f"{name:<25} {acc*100:>9.2f}%  {auc:>7.4f}  {cv*100:>9.2f}%")

print("="*65)

# ── 5. Best Model Detail ──────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]
print(f"\nBest Model: {best_name}  (Acc: {best['accuracy']*100:.2f}%  AUC: {best['auc']:.4f})")

y_pred_best = best['model'].predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Fail', 'Pass']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# ── 6. Save Model & Scaler ────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__))
joblib.dump(best['model'], os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(scaler,        os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, 'feature_names.pkl'))
print(f"\nModel saved → models/model.pkl")
print(f"Scaler saved → models/scaler.pkl")

# ── 7. Comparison Chart ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]
aucs  = [results[n]['auc'] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Performance Comparison\nStudent Performance Prediction Using ML",
             fontsize=14, fontweight='bold')

colors = ['#d62728' if n == best_name else '#2E74B5' for n in names]
bars   = axes[0].barh(names, accs, color=colors, edgecolor='white', height=0.6)
axes[0].set_xlabel("Accuracy (%)", fontsize=12)
axes[0].set_title("Test Accuracy", fontsize=13)
axes[0].set_xlim(50, 100)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va='center', fontsize=10)

bars2  = axes[1].barh(names, aucs, color=colors, edgecolor='white', height=0.6)
axes[1].set_xlabel("ROC-AUC Score", fontsize=12)
axes[1].set_title("AUC Score", fontsize=13)
axes[1].set_xlim(0.5, 1.0)
for bar, val in zip(bars2, aucs):
    axes[1].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va='center', fontsize=10)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'model_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"Chart saved → outputs/model_comparison.png")

# ── 8. Feature Importance ─────────────────────────────────────────────────────
if hasattr(best['model'], 'feature_importances_'):
    importances = pd.Series(best['model'].feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    importances.plot(kind='barh', ax=ax2, color='#2E74B5', edgecolor='white')
    ax2.set_title(f"Feature Importance – {best_name}", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Importance Score")
    plt.tight_layout()
    fi_path = os.path.join(OUT_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    print(f"Feature importance chart saved → outputs/feature_importance.png")

print("\nAll done!")
