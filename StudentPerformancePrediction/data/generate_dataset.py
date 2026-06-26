"""
generate_dataset.py
Generates a realistic synthetic student performance dataset.
Run this once to create data/student_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

attendance       = np.clip(np.random.normal(75, 15, n), 30, 100)
study_hours      = np.clip(np.random.normal(5, 2.5, n), 0, 14)
midterm_marks    = np.clip(np.random.normal(55, 18, n), 0, 100)
assignment_score = np.clip(np.random.normal(60, 15, n), 0, 100)
prev_gpa         = np.clip(np.random.normal(6.5, 1.5, n), 0, 10)
extracurricular  = np.random.randint(0, 2, n)
gender           = np.random.choice(['Male', 'Female'], n)
socioeconomic    = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.5, 0.2])
internet_access  = np.random.randint(0, 2, n)
parent_education = np.random.choice(['None', 'School', 'Graduate', 'Postgraduate'], n,
                                     p=[0.1, 0.3, 0.4, 0.2])

# Compute a weighted score to determine final grade
score = (
    0.30 * midterm_marks +
    0.20 * assignment_score +
    0.20 * (attendance / 10) +
    0.15 * (study_hours * 7) +
    0.15 * (prev_gpa * 10)
)
score = np.clip(score + np.random.normal(0, 5, n), 0, 100)

def grade(s):
    if s >= 85: return 'O'
    elif s >= 75: return 'A+'
    elif s >= 65: return 'A'
    elif s >= 55: return 'B+'
    elif s >= 45: return 'B'
    else:         return 'F'

final_grade  = np.array([grade(s) for s in score])
pass_fail    = np.where(score >= 45, 'Pass', 'Fail')
final_marks  = score.round(2)

df = pd.DataFrame({
    'StudentID'          : [f'STU{1000+i}' for i in range(n)],
    'Gender'             : gender,
    'SocioeconomicStatus': socioeconomic,
    'ParentEducation'    : parent_education,
    'InternetAccess'     : internet_access,
    'Extracurricular'    : extracurricular,
    'Attendance'         : attendance.round(2),
    'StudyHours'         : study_hours.round(2),
    'MidTermMarks'       : midterm_marks.round(2),
    'AssignmentScore'    : assignment_score.round(2),
    'PrevGPA'            : prev_gpa.round(2),
    'FinalMarks'         : final_marks,
    'FinalGrade'         : final_grade,
    'Result'             : pass_fail
})

df.to_csv('data/student_data.csv', index=False)
print(f"Dataset created: {len(df)} records")
print(df['Result'].value_counts())
print(df['FinalGrade'].value_counts())
