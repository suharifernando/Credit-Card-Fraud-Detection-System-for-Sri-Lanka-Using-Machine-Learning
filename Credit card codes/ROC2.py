import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Load and prepare dataset ---
filename = r"D:\uni\cleaned_dataset 3...csv"
transactions = pd.read_csv(filename)

# Fix column names
transactions = transactions.rename(columns={
    'approval_number': 'TransactionID',
    'card_number': 'AccountID',
    'net_amount': 'TransactionAmount'
})

# Encode AccountID
le_acc = LabelEncoder()
transactions['AccountID_enc'] = le_acc.fit_transform(transactions['AccountID'])

# Create labels (Status) using rule + noise
np.random.seed(42)
transactions['Status'] = transactions['TransactionAmount'].apply(
    lambda x: 'DECLINED' if x > 1000 else 'APPROVED'
)
mask = np.random.rand(len(transactions)) < 0.1
transactions.loc[mask, 'Status'] = transactions.loc[mask, 'Status'].map(
    lambda s: 'APPROVED' if s == 'DECLINED' else 'DECLINED'
)

# Features and numeric labels
X = transactions[['AccountID_enc', 'TransactionAmount']]
y = (transactions['Status'] == 'APPROVED').astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Initialize Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
}

# --- Evaluate each model ---
results = []
plt.figure(figsize=(7, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": auc
    })

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# --- Plot settings ---
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.title("ROC Curve Comparison – Credit Card Fraud Detection System (Dataset-2)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print Results Summary ---
df_results = pd.DataFrame(results)
print("\nModel Performance Summary")
print(df_results)
