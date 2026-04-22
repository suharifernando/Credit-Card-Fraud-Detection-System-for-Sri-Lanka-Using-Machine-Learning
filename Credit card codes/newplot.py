import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate model, return metrics and predictions."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="APPROVED")
    recall = recall_score(y_test, y_pred, pos_label="APPROVED")
    f1 = f1_score(y_test, y_pred, pos_label="APPROVED")
    
    print(f"\n{model_name} Results")
    print("-------------------")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1-score : {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "y_true": y_test,
        "y_pred": y_pred
    }

def main():
    # Load dataset
    filename = r"D:\uni\cleaned_dataset 3...csv"  
    transactions = pd.read_csv(filename)

    #  Fix column names to match dataset structure
    transactions = transactions.rename(columns={
        'approval_number': 'TransactionID',
        'card_number': 'AccountID',
        'net_amount': 'TransactionAmount'
    })

    # Encode AccountID
    le_acc = LabelEncoder()
    transactions['AccountID_enc'] = le_acc.fit_transform(transactions['AccountID'])

    #  Create labels (ground truth) with noise
    np.random.seed(42)
    transactions['Status'] = transactions['TransactionAmount'].apply(
        lambda x: 'DECLINED' if x > 1000 else 'APPROVED'
    )
    # Flip 10% labels randomly
    mask = np.random.rand(len(transactions)) < 0.1
    transactions.loc[mask, 'Status'] = transactions.loc[mask, 'Status'].map(
        lambda s: 'APPROVED' if s == 'DECLINED' else 'DECLINED'
    )

    # Features + labels
    X = transactions[['AccountID_enc', 'TransactionAmount']]
    y = transactions['Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = []

    # Models
    results.append(evaluate_model(LogisticRegression(max_iter=1000), X_train, X_test, y_train, y_test, "Logistic Regression"))
    results.append(evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test, "Random Forest"))
    results.append(evaluate_model(SVC(kernel='linear', random_state=42), X_train, X_test, y_train, y_test, "SVM"))

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1"]]
    print("\nSummary of All Models")
    print(df_results)

    # --- Plot 1: Metrics Comparison ---
    df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", figsize=(10, 6), rot=0
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Confusion Matrices ---
    for res in results:
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=["APPROVED", "DECLINED"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["APPROVED", "DECLINED"],
                    yticklabels=["APPROVED", "DECLINED"])
        plt.title(f"Confusion Matrix - {res['Model']}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
