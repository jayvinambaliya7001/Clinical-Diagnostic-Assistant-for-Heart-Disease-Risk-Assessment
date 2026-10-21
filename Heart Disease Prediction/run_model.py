# Heart Disease Prediction - Full Execution Script
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for saving plots without GUI popups
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("="*60)
print("1. LOADING DATASET")
print("="*60)
df = pd.read_csv('heart.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("Missing values per column:\n", df.isnull().sum())
print("\nClass distribution:\n", df['target'].value_counts())

print("\n" + "="*60)
print("2. PREPROCESSING & FEATURE ENGINEERING")
print("="*60)
# One-hot encoding categorical features
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Scaling numerical features
standScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])

X = dataset.drop('target', axis=1)
y = dataset['target']
print(f"Processed feature matrix shape: {X.shape}")

print("\n" + "="*60)
print("3. MODEL EVALUATION & CROSS-VALIDATION")
print("="*60)

# K-Nearest Neighbors
print("\n--- K-Nearest Neighbors ---")
knn_scores = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X, y, cv=10)
    knn_scores.append(scores.mean())
best_k = np.argmax(knn_scores) + 1
print(f"Optimal K: {best_k} with 10-fold CV Accuracy: {knn_scores[best_k-1]*100:.2f}%")

knn_model = KNeighborsClassifier(n_neighbors=12)
cvs_knn = cross_val_score(knn_model, X, y, cv=10)
print(f"KNeighbors (K=12) 10-fold CV Accuracy: {cvs_knn.mean()*100:.2f}%")

# Decision Tree
print("\n--- Decision Tree Classifier ---")
dt_scores = []
for d in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(dt, X, y, cv=10)
    dt_scores.append(scores.mean())
best_depth = np.argmax(dt_scores) + 1
print(f"Optimal max_depth: {best_depth} with 10-fold CV Accuracy: {dt_scores[best_depth-1]*100:.2f}%")

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
cvs_dt = cross_val_score(dt_model, X, y, cv=10)
print(f"Decision Tree (max_depth=3) 10-fold CV Accuracy: {cvs_dt.mean()*100:.2f}%")

# Random Forest
print("\n--- Random Forest Classifier ---")
rf_scores = []
estimators = list(range(10, 101, 10))
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5)
    rf_scores.append(scores.mean())
best_rf_idx = np.argmax(rf_scores)
print(f"Optimal n_estimators: {estimators[best_rf_idx]} with 5-fold CV Accuracy: {rf_scores[best_rf_idx]*100:.2f}%")

rf_model = RandomForestClassifier(n_estimators=90, random_state=42)
cvs_rf = cross_val_score(rf_model, X, y, cv=5)
print(f"Random Forest (n_estimators=90) 5-fold CV Accuracy: {cvs_rf.mean()*100:.2f}%")

print("\n" + "="*60)
print("4. DETAILED TRAIN-TEST SPLIT EVALUATION (BEST MODEL: RANDOM FOREST)")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
best_model = RandomForestClassifier(n_estimators=90, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Holdout Test Set Accuracy: {acc*100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model and scaler for deployment
joblib.dump(best_model, 'heart_disease_rf_model.pkl')
joblib.dump(standScaler, 'scaler.pkl')
print("\nSaved trained model to 'heart_disease_rf_model.pkl'")
print("Saved standard scaler to 'scaler.pkl'")
print("\nALL STEPS EXECUTED SUCCESSFULLY!")