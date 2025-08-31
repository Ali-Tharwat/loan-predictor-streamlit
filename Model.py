# train_and_save_artifacts.py
"""
Train pipeline and save artifacts used by app.py:
  - model.pkl
  - scaler.pkl
  - pca.pkl
  - features.pkl   (list of numeric features in order for scaler)
  - defaults.pkl   (medians for numeric features)
  - encoders.pkl   (dict: categorical_col -> LabelEncoder)
  - metadata.pkl   (date_cols_present, categorical_cols)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

CSV_PATH = "feature eng.csv"  # Updated CSV path
DATE_COLS = [
    "CONTACT_DATE", "ACCOUNT_SINCE", "DATE_TIME_LOANS",
    "LOAN_START_DATE", "LOAN_END_DATE", "KYC_DATE"
]
RANDOM_STATE = 42
ARTIFACT_DIR = "."


def calc_derived_features(df):
    """Calculate all derived features based on the provided formulas"""
    # Calculate basic loan values
    P = df["LOAN_AMOUNT"]
    r = df["INTEREST_RATE"] / 100  # Convert percentage to decimal
    n = df["LOAN_MONTHS"]

    monthly_payment = np.where(
        r == 0,
        P / n,
        P * (r / 12) / (1 - (1 + r / 12) ** -n)
    )
    total_interest = monthly_payment * n - P
    total_loan = P + total_interest

    df["TOT_INTEREST_AMT"] = total_interest
    df["TOTAL_LOAN_AMOUNT"] = total_loan
    df["MONTHLY_PAYMENT"] = monthly_payment

    # Calculate all additional derived features
    df["DTI"] = np.where(df["TOTAL_MONTHLY_INCOME"] > 0,
                         df["TOTAL_LOAN_AMOUNT"] / df["TOTAL_MONTHLY_INCOME"],
                         np.nan)

    df["MONTHLY_PAYMENT_BURDEN"] = np.where(df["TOTAL_MONTHLY_INCOME"] > 0,
                                            df["MONTHLY_PAYMENT"] / df["TOTAL_MONTHLY_INCOME"],
                                            np.nan)

    df["ANNUALIZED_PAYMENT_BURDEN"] = np.where(df["TOTAL_MONTHLY_INCOME"] > 0,
                                               (df["MONTHLY_PAYMENT"] * 12) / df["TOTAL_MONTHLY_INCOME"],
                                               np.nan)

    df["CUMULATIVE_INTEREST_PCT"] = np.where(df["LOAN_AMOUNT"] > 0,
                                             df["TOT_INTEREST_AMT"] / df["LOAN_AMOUNT"],
                                             np.nan)

    df["EFFECTIVE_INTEREST_RATE"] = np.where((df["LOAN_AMOUNT"] > 0) & (df["LOAN_MONTHS"] > 0),
                                             df["TOT_INTEREST_AMT"] / (df["LOAN_AMOUNT"] * (df["LOAN_MONTHS"] / 12)),
                                             np.nan)

    df["INTEREST_TO_INCOME_RATIO"] = np.where((df["TOTAL_MONTHLY_INCOME"] > 0) & (df["LOAN_MONTHS"] > 0),
                                              df["TOT_INTEREST_AMT"] / (
                                                          df["TOTAL_MONTHLY_INCOME"] * (df["LOAN_MONTHS"] / 12)),
                                              np.nan)

    df["INTEREST_COVERAGE"] = np.where(df["TOT_INTEREST_AMT"] > 0,
                                       df["TOTAL_LOAN_AMOUNT"] / df["TOT_INTEREST_AMT"],
                                       np.nan)

    df["PAYMENT_TO_LOAN_RATIO"] = np.where(df["LOAN_AMOUNT"] > 0,
                                           df["MONTHLY_PAYMENT"] / df["LOAN_AMOUNT"],
                                           np.nan)

    df["PRINCIPAL_SHARE_PER_MONTH"] = np.where((df["LOAN_MONTHS"] > 0) & (df["MONTHLY_PAYMENT"] > 0),
                                               (df["LOAN_AMOUNT"] / df["LOAN_MONTHS"]) / df["MONTHLY_PAYMENT"],
                                               np.nan)

    df["LOAN_TO_INCOME_EXPOSURE"] = np.where((df["TOTAL_MONTHLY_INCOME"] > 0) & (df["LOAN_MONTHS"] > 0),
                                             df["LOAN_AMOUNT"] / (df["TOTAL_MONTHLY_INCOME"] * df["LOAN_MONTHS"]),
                                             np.nan)

    df["TOTAL_COST_TO_INCOME_EXPOSURE"] = np.where((df["TOTAL_MONTHLY_INCOME"] > 0) & (df["LOAN_MONTHS"] > 0),
                                                   df["TOTAL_LOAN_AMOUNT"] / (
                                                               df["TOTAL_MONTHLY_INCOME"] * df["LOAN_MONTHS"]),
                                                   np.nan)

    df["INTEREST_SHARE_TOTAL"] = np.where(df["TOTAL_LOAN_AMOUNT"] > 0,
                                          df["TOT_INTEREST_AMT"] / df["TOTAL_LOAN_AMOUNT"],
                                          np.nan)

    df["PRINCIPAL_SHARE_TOTAL"] = np.where(df["TOTAL_LOAN_AMOUNT"] > 0,
                                           df["LOAN_AMOUNT"] / df["TOTAL_LOAN_AMOUNT"],
                                           np.nan)

    df["PRINCIPAL_TO_INTEREST_RATIO"] = np.where(df["TOT_INTEREST_AMT"] > 0,
                                                 df["LOAN_AMOUNT"] / df["TOT_INTEREST_AMT"],
                                                 np.nan)

    return df


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    # Read header to know which date cols exist
    sample = pd.read_csv(CSV_PATH, nrows=0)
    date_cols_present = [c for c in DATE_COLS if c in sample.columns]

    print("Loading CSV with parse_dates for detected date columns...")
    df = pd.read_csv(CSV_PATH, parse_dates=date_cols_present)

    if 'OVERDUE_STATUS' not in df.columns:
        raise KeyError("OVERDUE_STATUS column missing")

    # Check if derived features already exist in the dataset
    derived_features = [
        "TOT_INTEREST_AMT", "TOTAL_LOAN_AMOUNT", "MONTHLY_PAYMENT", "DTI",
        "MONTHLY_PAYMENT_BURDEN", "ANNUALIZED_PAYMENT_BURDEN", "CUMULATIVE_INTEREST_PCT",
        "EFFECTIVE_INTEREST_RATE", "INTEREST_TO_INCOME_RATIO", "INTEREST_COVERAGE",
        "PAYMENT_TO_LOAN_RATIO", "PRINCIPAL_SHARE_PER_MONTH", "LOAN_TO_INCOME_EXPOSURE",
        "TOTAL_COST_TO_INCOME_EXPOSURE", "INTEREST_SHARE_TOTAL", "PRINCIPAL_SHARE_TOTAL",
        "PRINCIPAL_TO_INTEREST_RATIO"
    ]

    # Calculate derived features if they don't exist
    missing_derived = [feat for feat in derived_features if feat not in df.columns]
    if missing_derived:
        print(f"Calculating missing derived features: {missing_derived}")
        df = calc_derived_features(df)

    # map target
    mapping = {"PDO": 0, "CUR": 1}
    df['OVERDUE_STATUS'] = df['OVERDUE_STATUS'].map(mapping)
    if df['OVERDUE_STATUS'].isnull().any():
        raise ValueError("Some OVERDUE_STATUS values could not be mapped to 0/1")

    # Expand date features
    for col in date_cols_present:
        df[f'{col}_YEAR'] = df[col].dt.year
        df[f'{col}_MONTH'] = df[col].dt.month
        df[f'{col}_DAY'] = df[col].dt.day
        df[f'{col}_DAYS_SINCE'] = (df[col] - pd.Timestamp("1980-01-01")).dt.days

    # Drop raw date columns
    if date_cols_present:
        df.drop(columns=date_cols_present, inplace=True)

    # Identify categorical columns (object dtype)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")

    # Fit LabelEncoders and transform
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # convert to str to avoid issues with NaN
        df[col] = df[col].astype(str).fillna("nan_missing")
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoders[col] = le

    # X / y
    y = df['OVERDUE_STATUS']
    X = df.drop(columns=['OVERDUE_STATUS'])

    # Numeric features that will be used by scaler/model (label-encoded columns are numeric now)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric features count: {len(numeric_cols)}")

    # Fill NaNs with median and save medians
    medians = X[numeric_cols].median()
    X[numeric_cols] = X[numeric_cols].fillna(medians)

    # Fit scaler on numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])

    # PCA
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print("PCA transformed shape:", X_pca.shape)

    # Balance with SMOTE (only for training)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_pca, y)
    print("After SMOTE class counts:\n", pd.Series(y_res).value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=RANDOM_STATE,
                                                        stratify=y_res)

    # Train RandomForest
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print("\n=== METRICS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=['PDO (Rejected)', 'CUR (Approved)']))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    print("Saving artifacts...")
    with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    # features.pkl must be the exact numeric column order used by scaler
    with open(os.path.join(ARTIFACT_DIR, "features.pkl"), "wb") as f:
        pickle.dump(numeric_cols, f)
    # medians/defaults
    med_dict = {str(k): float(v) for k, v in medians.items()}
    with open(os.path.join(ARTIFACT_DIR, "defaults.pkl"), "wb") as f:
        pickle.dump(med_dict, f)
    # encoders
    with open(os.path.join(ARTIFACT_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    # metadata: date cols present and categorical cols
    metadata = {"date_cols": date_cols_present, "categorical_cols": categorical_cols}
    with open(os.path.join(ARTIFACT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("Saved: model.pkl, scaler.pkl, pca.pkl, features.pkl, defaults.pkl, encoders.pkl, metadata.pkl")
    print("Now run: streamlit run app.py")


if __name__ == "__main__":
    main()