import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, roc_auc_score
)

# ================================
# 1. Load and Prepare Dataset
# ================================
filename = r"D:\uni\cleaned_bank_transactions.csv"
transactions = pd.read_csv(filename)

# Encode AccountID
le_acc = LabelEncoder()
transactions['AccountID_enc'] = le_acc.fit_transform(transactions['AccountID'].astype(str))

# Generate labels (APPROVED / DECLINED) with some noise
np.random.seed(42)
transactions['Status'] = transactions['TransactionAmount'].apply(
    lambda x: 'DECLINED' if x > 1000 else 'APPROVED'
)
mask = np.random.rand(len(transactions)) < 0.1
transactions.loc[mask, 'Status'] = transactions.loc[mask, 'Status'].map(
    lambda s: 'APPROVED' if s == 'DECLINED' else 'DECLINED'
)

# Features and labels
X = transactions[['AccountID_enc', 'TransactionAmount']]
y = transactions['Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Binary numeric labels for ROC
y_test_bin = (y_test == "APPROVED").astype(int)

# ================================
# 2. Initialize Models
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Linear Kernel)": SVC(kernel='linear', probability=True, random_state=42)
}

# ================================
# 3. Train, Evaluate & Collect ROC data
# ================================
plt.figure(figsize=(7, 6))

for name, model in models.items():
    # Scale features for SVM only
    if "SVM" in name:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_score = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="APPROVED")
    recall = recall_score(y_test, y_pred, pos_label="APPROVED")
    f1 = f1_score(y_test, y_pred, pos_label="APPROVED")
    auc = roc_auc_score(y_test_bin, y_score)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    print(f"\n{name} Results")
    print("-----------------------")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC      : {auc:.3f}")
    print(classification_report(y_test, y_pred))

# ================================
# 4. Final ROC Plot
# ================================
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
plt.title("ROC Curve Comparison – Credit Card Fraud Detection System")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
