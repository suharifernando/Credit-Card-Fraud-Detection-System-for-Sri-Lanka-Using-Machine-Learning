# Credit-Card-Fraud-Detection-System-for-Sri-Lanka-Using-Machine-Learning
Overview
This project introduces a hybrid fraud detection system designed for Sri Lanka’s financial sector. It combines SMS‑based user authentication with machine learning models (Isolation Forest, Logistic Regression, Random Forest, SVM) to detect fraudulent transactions in real time.

Datasets
Global Dataset (Kaggle): 2,512 transactions, 16 attributes (transaction details, account info, demographics, device usage).

Local Dataset (Sri Lanka Merchant): 860 transactions, with categorical identifiers (TransactionDate, ApprovalNumber, CardNumber) and numeric attributes (GrossAmount, DiscountAmount, NetAmount).

 Methodology
 
 Data Preprocessing: Cleaning duplicates, handling missing values, encoding categorical variables, and standardizing formats.

Model Selection:

Unsupervised: Isolation Forest

Supervised: Logistic Regression, Random Forest, SVM

Evaluation Metrics: Accuracy, precision, recall, F1‑score.

Hybrid Workflow: SMS verification (“YES” or “NO”) combined with ML predictions for anomaly detection.

 Results
 
Global Dataset: Accuracy 0.88, but poor fraud recall (0.06).

Local Dataset: Balanced outcomes, Random Forest (0.88 accuracy) and SVM (0.87 accuracy) with strong recall.

Hybrid Approach: SMS verification instantly rejects unauthorized transactions, reducing reliance on ML predictions alone.
