# Student Performance Prediction Using ML

**Internship Project | Data Pro Organisation**  
**Intern:** SEERAPU SRAVANI G.MADHAVI | Reg. No: 24U45A4219 24U45A4208
**Faculty Guide:** Mrs. M. KALYANI | Dept. of CSE-AIML, JNTUGV  
**Duration:** 13-04-2026 to 23-05-2026  

---

## Project Overview

This project builds a Machine Learning system to predict student academic performance (Pass/Fail or Grade Category) based on features such as attendance, study hours, mid-term marks, assignment scores, and socioeconomic factors.

The best model (XGBoost) achieved **88.5% accuracy** and **AUC of 0.92**.  
A **Streamlit web app** is provided for real-time predictions.

---

## Project Structure

```
StudentPerformancePrediction/
│
├── data/
│   ├── student_data.csv              # Raw dataset
│   └── student_data_cleaned.csv      # Preprocessed dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb        # Data Cleaning & Feature Engineering
│   ├── 03_Model_Training.ipynb       # ML Model Training & Comparison
│   └── 04_XGBoost_Final.ipynb        # Final Model + Hyperparameter Tuning
│
├── models/
│   ├── train_models.py               # Train all ML models
│   ├── evaluate_models.py            # Evaluate and compare models
│   └── model.pkl                     # Saved XGBoost model (generated after training)
│
├── app/
│   └── streamlit_app.py              # Streamlit web application
│
├── outputs/
│   └── model_comparison.png          # Model accuracy comparison chart (generated)
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
python models/train_models.py
```

### 3. Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## Model Performance Summary

| Model               | Accuracy | Notes                        |
|---------------------|----------|------------------------------|
| Linear Regression   | 74.0%    | Baseline model               |
| Decision Tree       | 78.0%    | max_depth tuned              |
| Random Forest       | 85.0%    | n_estimators=100             |
| SVM (RBF)           | 82.0%    | Grid search optimized        |
| KNN                 | 80.0%    | Best at K=7                  |
| Gradient Boosting   | 86.0%    | Default params               |
| **XGBoost**         | **88.5%**| **Best model, AUC=0.92**     |

---

## Key Features Used for Prediction
1. Attendance Percentage
2. Mid-Term Marks
3. Assignment Scores
4. Study Hours per Week
5. Extracurricular Activities
6. Previous Academic Record (GPA)
7. Socioeconomic Status

---

## Technologies Used
- Python 3.10, Jupyter Notebook
- pandas, numpy, scikit-learn, XGBoost
- matplotlib, seaborn
- Streamlit, joblib
- GitHub (version control)
