"""
Train the Salary Prediction model and save artifacts.
Run this script once to generate model.pkl and encoders.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Load Dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv("Employee-salary-prediction.csv")

# ── Rename columns ────────────────────────────────────────────────────────────
df.rename(columns={'Education Level': 'Degree', 'Years of Experience': 'Experience'}, inplace=True)

# ── Clean Data ────────────────────────────────────────────────────────────────
df = df.drop_duplicates(keep='first')
df = df.dropna(how='any')

# ── Label Encoding ────────────────────────────────────────────────────────────
le_degree   = LabelEncoder()
le_jobtitle = LabelEncoder()
le_gender   = LabelEncoder()

df['Degree_encoder']        = le_degree.fit_transform(df['Degree'])
df['Job Title_encoder']     = le_jobtitle.fit_transform(df['Job Title'])
df['Gender_encoder']        = le_gender.fit_transform(df['Gender'])

# ── Feature Scaling ───────────────────────────────────────────────────────────
ss = StandardScaler()
df['Age_scaling']        = ss.fit_transform(df[['Age']])
df['Experience_scaling'] = ss.fit_transform(df[['Experience']])

# ── Features & Target ─────────────────────────────────────────────────────────
X = df[['Age_scaling', 'Experience_scaling', 'Degree_encoder', 'Job Title_encoder', 'Gender_encoder']]
y = df['Salary']

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train Model ───────────────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = lr.predict(X_test)
print(f"R² Score  : {r2_score(y_test, preds)*100:.2f}%")
print(f"MAE       : {mean_absolute_error(y_test, preds):,.2f}")
print(f"RMSE      : {mean_squared_error(y_test, preds)**0.5:,.2f}")

# ── Save Artifacts ────────────────────────────────────────────────────────────
artifacts = {
    "model"      : lr,
    "scaler"     : ss,
    "le_degree"  : le_degree,
    "le_jobtitle": le_jobtitle,
    "le_gender"  : le_gender,
    "age_mean"   : df['Age'].mean(),
    "age_std"    : df['Age'].std(),
    "exp_mean"   : df['Experience'].mean(),
    "exp_std"    : df['Experience'].std(),
    "job_titles" : list(le_jobtitle.classes_),
    "degrees"    : list(le_degree.classes_),
    "genders"    : list(le_gender.classes_),
}

joblib.dump(artifacts, "artifacts.pkl")
print("\n✅ artifacts.pkl saved successfully!")
