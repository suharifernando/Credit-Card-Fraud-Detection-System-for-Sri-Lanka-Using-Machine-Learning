import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

def simulate_ml_verification(transaction, model):
    """Simulate SMS verification using ML predictions (fully automated)"""
    # Prepare features for prediction (Account + Amount only)
    features = pd.DataFrame([{
        'AccountID_enc': transaction['AccountID_enc'],
        'TransactionAmount': transaction['TransactionAmount']
    }])
    
    prediction = model.predict(features)[0]  # -1 = anomaly (fraud), 1 = normal
    status = "DECLINED" if prediction == -1 else "APPROVED"
    
    return status

def main():
    try:
        print("Transaction Verification System")
        print("-------------------------------")
        
        # Load dataset
        filename = r"D:\uni\cleaned_dataset 3...csv"
        transactions = pd.read_csv(filename)

        # Rename columns to fit expected structure
        transactions = transactions.rename(columns={
            'approval_number': 'TransactionID',
            'card_number': 'AccountID',
            'net_amount': 'TransactionAmount'
        })

        # Check required columns
        required_columns = {'TransactionID', 'AccountID', 'TransactionAmount'}
        missing = required_columns - set(transactions.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"Loaded {len(transactions)} transactions.\n")

        # Encode AccountID
        le_account = LabelEncoder()
        transactions['AccountID_enc'] = le_account.fit_transform(transactions['AccountID'])

        # Features for ML (Account + Amount only)
        X = transactions[['AccountID_enc', 'TransactionAmount']]
        
        # Train Isolation Forest
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X)

        print("Begin verifying transactions (type YES or NO):")
        print("----------------------------------------------\n")
        
        statuses = []
        for idx, row in transactions.iterrows():
            print(f"SMS to user {row['AccountID']}:")
            print(f"Is this your transaction of ${row['TransactionAmount']:.2f}? Reply YES or NO.")
            user_reply = input("Your reply (YES/NO): ").strip().upper()
            while user_reply not in ['YES', 'NO']:
                user_reply = input("Please reply YES or NO: ").strip().upper()

            if user_reply == 'NO':
                status = "DECLINED"
            else:
                # Use ML prediction only when user says YES
                status = simulate_ml_verification(row, model)

            print(f"Transaction {status}\n")
            statuses.append(status)

        transactions['Status'] = statuses
        
        # Save results
        output_file = "verified_transactions_ml_auto.csv"
        transactions.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
