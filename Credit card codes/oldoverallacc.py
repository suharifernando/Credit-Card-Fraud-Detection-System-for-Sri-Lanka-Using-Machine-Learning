import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load dataset ---
filename = r"D:\uni\cleaned_bank_transactions.csv"
transactions = pd.read_csv(filename)

print("\n🔹 First few rows of dataset:")
print(transactions.head())
print("\n🔹 Columns in dataset:")
print(transactions.columns.tolist())

# --- Encode AccountID ---
le_acc = LabelEncoder()
transactions['AccountID_enc'] = le_acc.fit_transform(transactions['AccountID'].astype(str))

# --- Generate labels (APPROVED / DECLINED) with noise ---
np.random.seed(42)
transactions['Status'] = transactions['TransactionAmount'].apply(
    lambda x: 'DECLINED' if x > 1000 else 'APPROVED'
)
mask = np.random.rand(len(transactions)) < 0.1
transactions.loc[mask, 'Status'] = transactions.loc[mask, 'Status'].map(
    lambda s: 'APPROVED' if s == 'DECLINED' else 'DECLINED'
)

# --- Features + labels ---
X = transactions[['AccountID_enc', 'TransactionAmount']]
y = transactions['Status']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Train and evaluate model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="APPROVED")
recall = recall_score(y_test, y_pred, pos_label="APPROVED")
f1 = f1_score(y_test, y_pred, pos_label="APPROVED")

print("\nOverall Model Results")
print("----------------------")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-score : {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred, labels=["APPROVED", "DECLINED"])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["APPROVED", "DECLINED"],
            yticklabels=["APPROVED", "DECLINED"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

