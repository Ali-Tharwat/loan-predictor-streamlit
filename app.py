# app.py
"""
Streamlit UI for human-friendly single-customer prediction.
Requires artifacts created by train_and_save_artifacts.py:
  - model.pkl, scaler.pkl, pca.pkl, features.pkl, defaults.pkl, encoders.pkl, metadata.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import date

# Load custom CSS
def load_css():
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Using default styling.")

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
load_css()

# Enhanced title with custom styling
st.markdown("""
<div class="header">
    <h1>🏦 Loan Repayment Risk Prediction</h1>
    <p>Advanced ML-powered loan approval system</p>
</div>
""", unsafe_allow_html=True)

# Enhanced mode selection - inline layout
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 🎯 Select Input Mode")
with col2:
    st.markdown('<div style="margin-top: -10px;">', unsafe_allow_html=True)
    mode = st.radio(
        "",
        options=["Human-friendly input", "Paste a CSV row"],
        index=0,
        horizontal=True,
        help="Select your preferred input method for customer data entry"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

ARTIFACT_FILES = {
    "model": "model.pkl",
    "scaler": "scaler.pkl",
    "pca": "pca.pkl",
    "features": "features.pkl",
    "defaults": "defaults.pkl",
    "encoders": "encoders.pkl",
    "metadata": "metadata.pkl"
}


@st.cache_resource
def load_artifacts():
    missing = [name for name, path in ARTIFACT_FILES.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing artifact(s): " + ", ".join(missing))
    with open(ARTIFACT_FILES["model"], "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACT_FILES["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACT_FILES["pca"], "rb") as f:
        pca = pickle.load(f)
    with open(ARTIFACT_FILES["features"], "rb") as f:
        features = pickle.load(f)
    with open(ARTIFACT_FILES["defaults"], "rb") as f:
        defaults = pickle.load(f)
    with open(ARTIFACT_FILES["encoders"], "rb") as f:
        encoders = pickle.load(f)
    with open(ARTIFACT_FILES["metadata"], "rb") as f:
        metadata = pickle.load(f)
    return model, scaler, pca, features, defaults, encoders, metadata


def calc_loan_values(loan_amount, interest_rate, loan_months):
    """Calculate derived loan values based on loan amount, interest rate, and duration"""
    P = loan_amount
    r = interest_rate / 100  # Convert percentage to decimal
    n = loan_months

    if r == 0:  # interest-free loan
        monthly_payment = P / n
        total_interest = 0.0
    else:
        i = r / 12  # monthly rate
        monthly_payment = P * (i / (1 - (1 + i) ** -n))
        total_interest = monthly_payment * n - P

    total_loan = P + total_interest
    return total_interest, total_loan, monthly_payment


def calc_all_derived_features(loan_amount, interest_rate, loan_months, total_monthly_income):
    """Calculate all derived features based on the provided formulas"""
    # Calculate basic loan values
    total_interest, total_loan, monthly_payment = calc_loan_values(loan_amount, interest_rate, loan_months)

    # Initialize derived features dictionary
    derived = {
        "TOT_INTEREST_AMT": total_interest,
        "TOTAL_LOAN_AMOUNT": total_loan,
        "MONTHLY_PAYMENT": monthly_payment
    }

    # Calculate additional derived features
    if total_monthly_income > 0:
        derived["DTI"] = total_loan / total_monthly_income
        derived["MONTHLY_PAYMENT_BURDEN"] = monthly_payment / total_monthly_income
        derived["ANNUALIZED_PAYMENT_BURDEN"] = (monthly_payment * 12) / total_monthly_income
        derived["INTEREST_TO_INCOME_RATIO"] = total_interest / (total_monthly_income * (loan_months / 12))
        derived["LOAN_TO_INCOME_EXPOSURE"] = loan_amount / (total_monthly_income * loan_months)
        derived["TOTAL_COST_TO_INCOME_EXPOSURE"] = total_loan / (total_monthly_income * loan_months)
    else:
        # Set to 0 if no income
        derived.update({
            "DTI": 0,
            "MONTHLY_PAYMENT_BURDEN": 0,
            "ANNUALIZED_PAYMENT_BURDEN": 0,
            "INTEREST_TO_INCOME_RATIO": 0,
            "LOAN_TO_INCOME_EXPOSURE": 0,
            "TOTAL_COST_TO_INCOME_EXPOSURE": 0
        })

    # Calculate other ratios
    if loan_amount > 0:
        derived["CUMULATIVE_INTEREST_PCT"] = total_interest / loan_amount
        derived["PAYMENT_TO_LOAN_RATIO"] = monthly_payment / loan_amount
        derived["PRINCIPAL_SHARE_TOTAL"] = loan_amount / total_loan
    else:
        derived.update({
            "CUMULATIVE_INTEREST_PCT": 0,
            "PAYMENT_TO_LOAN_RATIO": 0,
            "PRINCIPAL_SHARE_TOTAL": 0
        })

    if loan_months > 0 and monthly_payment > 0:
        derived["PRINCIPAL_SHARE_PER_MONTH"] = (loan_amount / loan_months) / monthly_payment
    else:
        derived["PRINCIPAL_SHARE_PER_MONTH"] = 0

    if loan_amount > 0 and loan_months > 0:
        derived["EFFECTIVE_INTEREST_RATE"] = total_interest / (loan_amount * (loan_months / 12))
    else:
        derived["EFFECTIVE_INTEREST_RATE"] = 0

    if total_interest > 0:
        derived["INTEREST_COVERAGE"] = total_loan / total_interest
        derived["PRINCIPAL_TO_INTEREST_RATIO"] = loan_amount / total_interest
        derived["INTEREST_SHARE_TOTAL"] = total_interest / total_loan
    else:
        derived.update({
            "INTEREST_COVERAGE": 0,
            "PRINCIPAL_TO_INTEREST_RATIO": 0,
            "INTEREST_SHARE_TOTAL": 0
        })

    return derived


def make_date_inputs(date_cols):
    """Return dict date_col -> datetime.date (user choice)."""
    date_inputs = {}
    st.subheader("Date fields")
    for col in date_cols:
        # default to today
        d = st.date_input(col, value=date.today(), key=f"date_{col}")
        date_inputs[col] = pd.to_datetime(d)
    return date_inputs


def build_ui_inputs(features, defaults, encoders, metadata):
    categorical_cols = metadata.get("categorical_cols", [])
    date_cols = metadata.get("date_cols", [])

    cat_inputs = {}
    numeric_inputs = {}
    date_inputs = {}

    # -----------------------------
    # Geographical Info
    # -----------------------------
    geo_cols = ["RESIDENCE", "BIRTHPLACE", "NATIONALITY", "EG_FGN_RESIDENT", "ID_TYPE"]
    with st.expander("🌍 Geographical Information", expanded=True):
        for col in geo_cols:
            if col in categorical_cols:
                le = encoders.get(col)
                if le:
                    # Get the categorical default (mode) for this column
                    categorical_default = defaults.get(f"{col}_categorical", le.classes_[0])
                    # Find the index of the default value in the classes
                    try:
                        default_index = list(le.classes_).index(categorical_default)
                    except ValueError:
                        default_index = 0  # fallback if not found

                    sel = st.selectbox(
                        f"📍 {col.replace('_', ' ').title()}",
                        list(le.classes_),
                        index=default_index,
                        key=f"cat_{col}",
                        help=f"Select the {col.lower().replace('_', ' ')}"
                    )
                    cat_inputs[col] = str(sel)
            elif col in features:
                val = st.number_input(
                    f"📊 {col.replace('_', ' ').title()}",
                    value=float(defaults.get(col, 0.0)),
                    key=f"num_{col}",
                    help=f"Enter the {col.lower().replace('_', ' ')}"
                )
                numeric_inputs[col] = val

    # -----------------------------
    # Personal
    # -----------------------------
    personal_cols = ["CUSTOMER_SEGMENT", "INDUSTRY", "JOB_TITLE", "INCOME_SOURCE",
                     "SEX", "TITLE", "AGE", "MARITAL_STATUS", "TOTAL_MONTHLY_INCOME"]
    with st.expander("👤 Personal & Demographics", expanded=True):
        col1, col2 = st.columns(2)

        for i, col in enumerate(personal_cols):
            container = col1 if i % 2 == 0 else col2

            with container:
                if col in categorical_cols:
                    le = encoders.get(col)
                    if le:
                        # Get the categorical default (mode) for this column
                        categorical_default = defaults.get(f"{col}_categorical", le.classes_[0])
                        # Find the index of the default value in the classes
                        try:
                            default_index = list(le.classes_).index(categorical_default)
                        except ValueError:
                            default_index = 0  # fallback if not found

                        sel = st.selectbox(
                            f"👥 {col.replace('_', ' ').title()}",
                            list(le.classes_),
                            index=default_index,
                            key=f"cat_{col}",
                            help=f"Select the {col.lower().replace('_', ' ')}"
                        )
                        cat_inputs[col] = str(sel)
                elif col == "AGE":
                    # replace AGE input with birthdate picker
                    bdate = st.date_input(
                        "🎂 Birthdate",
                        value=pd.to_datetime("1980-01-01").date(),
                        min_value=pd.to_datetime("1925-01-01").date(),
                        max_value=pd.to_datetime("2025-12-31").date(),
                        key="birthdate",
                        help="Select the customer's birth date"
                    )
                    today = pd.Timestamp.today()
                    bdate_ts = pd.Timestamp(bdate)

                    # Calculate age with decimal precision
                    age_years = today.year - bdate_ts.year
                    age_days = (today - bdate_ts.replace(year=today.year)).days
                    if age_days < 0:  # Birthday hasn't occurred this year
                        age_years -= 1
                        age_days = (today - bdate_ts.replace(year=today.year - 1)).days

                    age_decimal = age_years + (age_days / 365.25)
                    age_formatted = round(age_decimal, 2)

                    numeric_inputs[col] = age_formatted
                    st.info(f"Calculated age: **{age_formatted:.2f} years**")
                elif col in features:
                    val = st.number_input(
                        f"💰 {col.replace('_', ' ').title()}",
                        value=float(defaults.get(col, 0.0)),
                        step=0.01,
                        format="%.2f",
                        key=f"num_{col}",
                        help=f"Enter the {col.lower().replace('_', ' ')}"
                    )
                    numeric_inputs[col] = val

    # -----------------------------
    # Account / Contact Info
    # -----------------------------
    account_cols = ["CONTACT_DATE", "ACCOUNT_SINCE", "DATE_TIME_LOANS",
                    "CUSTOMER_TYPE", "CUSTOMER_STATUS", "CURRENCY"]
    with st.expander("🏦 Account / Contact Info", expanded=False):
        for col in account_cols:
            if col in date_cols:
                d = st.date_input(
                    col,
                    value=pd.to_datetime(defaults.get(col, pd.Timestamp.today())),
                    min_value=pd.to_datetime("1900-01-01"),
                    max_value=pd.to_datetime("2100-12-31"),
                    key=f"date_{col}"
                )
                date_inputs[col] = pd.to_datetime(d)
            elif col in categorical_cols:
                le = encoders.get(col)
                if le:
                    # Get the categorical default (mode) for this column
                    categorical_default = defaults.get(f"{col}_categorical", le.classes_[0])
                    # Find the index of the default value in the classes
                    try:
                        default_index = list(le.classes_).index(categorical_default)
                    except ValueError:
                        default_index = 0  # fallback if not found

                    sel = st.selectbox(col, list(le.classes_), index=default_index, key=f"cat_{col}")
                    cat_inputs[col] = str(sel)
            elif col in features:
                val = st.number_input(col, value=float(defaults.get(col, 0.0)), key=f"num_{col}")
                numeric_inputs[col] = val

    # -----------------------------
    # KYC / Risk / Compliance
    # -----------------------------
    kyc_cols = ["KYC_DATE", "KYC_REMAINING_DAYS", "KYC_YEARS_INT",
                "KYC_RISK_RATE", "KYC_RISK_CATEG"]
    # In the KYC / Risk / Compliance section:
    with st.expander("⚠️ KYC / Risk / Compliance", expanded=False):
        for col in kyc_cols:
            if col in date_cols:
                d = st.date_input(
                    col,
                    value=pd.to_datetime(defaults.get(col, pd.Timestamp.today())),
                    min_value=pd.to_datetime("1900-01-01"),
                    max_value=pd.to_datetime("2100-12-31"),
                    key=f"date_{col}"
                )
                date_inputs[col] = pd.to_datetime(d)
            elif col in categorical_cols:
                le = encoders.get(col)
                if le:
                    # Get the categorical default (mode) for this column
                    categorical_default = defaults.get(f"{col}_categorical", le.classes_[0])
                    # Find the index of the default value in the classes
                    try:
                        default_index = list(le.classes_).index(categorical_default)
                    except ValueError:
                        default_index = 0  # fallback if not found

                    sel = st.selectbox(col, list(le.classes_), index=default_index, key=f"cat_{col}")
                    cat_inputs[col] = str(sel)
            elif col in features:
                val = st.number_input(col, value=float(defaults.get(col, 0.0)), key=f"num_{col}")
                numeric_inputs[col] = val

    # -----------------------------
    # Loan Details (Enhanced)
    # -----------------------------
    loan_cols = ["LOAN_START_DATE", "LOAN_END_DATE", "INTEREST_BASIS", "INTEREST_RATE",
                 "LOAN_MONTHS", "LOAN_AMOUNT", "CONTRACT_TYPE", "ISL&SME_REPAY_TYPE",
                 "LOAN_TYPE", "LOAN_SUB_TYPE"]

    with st.expander("💰 Loan Details", expanded=True):
        # Date inputs in columns
        col1, col2 = st.columns(2)
        with col1:
            loan_start = st.date_input(
                "📅 Loan Start Date",
                value=pd.to_datetime("2020-01-01"),
                min_value=pd.to_datetime("1900-01-01"),
                max_value=pd.to_datetime("2100-12-31"),
                key="loan_start",
                help="Select when the loan starts"
            )
        with col2:
            loan_end = st.date_input(
                "📅 Loan End Date",
                value=pd.to_datetime("2025-01-01"),
                min_value=pd.to_datetime("1900-01-01"),
                max_value=pd.to_datetime("2100-12-31"),
                key="loan_end",
                help="Select when the loan ends"
            )

        date_inputs["LOAN_START_DATE"] = pd.to_datetime(loan_start)
        date_inputs["LOAN_END_DATE"] = pd.to_datetime(loan_end)

        # calculate loan duration in months
        duration_months = (loan_end.year - loan_start.year) * 12 + (loan_end.month - loan_start.month)
        if loan_end.day < loan_start.day:
            duration_months -= 1  # adjust if end-day is before start-day

        numeric_inputs["LOAN_MONTHS"] = max(duration_months, 0)  # no negative durations

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0; color: white;">📅 Loan Duration: {duration_months} months</h4>
        </div>
        """, unsafe_allow_html=True)

        # Other loan fields in organized layout
        for col in loan_cols:
            if col in ["LOAN_START_DATE", "LOAN_END_DATE", "LOAN_MONTHS"]:
                continue  # skip, since we handle them above
            if col in categorical_cols:
                le = encoders.get(col)
                if le:
                    # Get the categorical default (mode) for this column
                    categorical_default = defaults.get(f"{col}_categorical", le.classes_[0])
                    # Find the index of the default value in the classes
                    try:
                        default_index = list(le.classes_).index(categorical_default)
                    except ValueError:
                        default_index = 0  # fallback if not found

                    sel = st.selectbox(
                        f"📋 {col.replace('_', ' ').title()}",
                        list(le.classes_),
                        index=default_index,
                        key=f"cat_{col}",
                        help=f"Select the {col.lower().replace('_', ' ')}"
                    )
                    cat_inputs[col] = str(sel)
            elif col in features:
                val = st.number_input(
                    f"💵 {col.replace('_', ' ').title()}",
                    value=float(defaults.get(col, 0.0)),
                    step=0.01,
                    format="%.2f",
                    key=f"num_{col}",
                    help=f"Enter the {col.lower().replace('_', ' ')}"
                )
                numeric_inputs[col] = val

        # Calculate ALL derived loan values
        loan_amount = numeric_inputs.get("LOAN_AMOUNT", 0.0)
        interest_rate = numeric_inputs.get("INTEREST_RATE", 0.0)
        loan_months = numeric_inputs.get("LOAN_MONTHS", 1.0)
        total_monthly_income = numeric_inputs.get("TOTAL_MONTHLY_INCOME", 0.0)

        if loan_months > 0 and loan_amount > 0:
            # Calculate all derived features
            derived_features = calc_all_derived_features(loan_amount, interest_rate, loan_months, total_monthly_income)

            # Update numeric_inputs with all derived features
            for key, value in derived_features.items():
                numeric_inputs[key] = value

            # Enhanced display of calculated values
            st.markdown("### 📊 Calculated Loan Metrics")

            # Create metrics in a grid layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>💰 Financial Overview</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Total Interest", f"${derived_features['TOT_INTEREST_AMT']:,.2f}")
                st.metric("Total Loan", f"${derived_features['TOTAL_LOAN_AMOUNT']:,.2f}")
                st.metric("Monthly Payment", f"${derived_features['MONTHLY_PAYMENT']:,.2f}")
                st.metric("DTI Ratio", f"{derived_features['DTI']:.2f}")
                st.metric("Monthly Burden", f"{derived_features['MONTHLY_PAYMENT_BURDEN']:.2f}")

            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>📈 Risk Metrics</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Annual Burden", f"{derived_features['ANNUALIZED_PAYMENT_BURDEN']:.2f}")
                st.metric("Interest %", f"{derived_features['CUMULATIVE_INTEREST_PCT']:.2f}")
                st.metric("Effective Rate", f"{derived_features['EFFECTIVE_INTEREST_RATE']:.2f}")
                st.metric("Interest/Income", f"{derived_features['INTEREST_TO_INCOME_RATIO']:.2f}")
                st.metric("Interest Coverage", f"{derived_features['INTEREST_COVERAGE']:.2f}")

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>⚖️ Exposure Ratios</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Payment/Loan", f"{derived_features['PAYMENT_TO_LOAN_RATIO']:.2f}")
                st.metric("Principal/Month", f"{derived_features['PRINCIPAL_SHARE_PER_MONTH']:.2f}")
                st.metric("Loan Exposure", f"{derived_features['LOAN_TO_INCOME_EXPOSURE']:.2f}")
                st.metric("Total Exposure", f"{derived_features['TOTAL_COST_TO_INCOME_EXPOSURE']:.2f}")
                st.metric("Principal/Interest", f"{derived_features['PRINCIPAL_TO_INTEREST_RATIO']:.2f}")

        else:
            st.warning("⚠️ Loan duration and amount must be greater than 0 to calculate derived values.")
            # Set all derived features to 0 to avoid errors
            derived_features_list = ["TOT_INTEREST_AMT", "TOTAL_LOAN_AMOUNT", "MONTHLY_PAYMENT", "DTI",
                                     "MONTHLY_PAYMENT_BURDEN", "ANNUALIZED_PAYMENT_BURDEN", "CUMULATIVE_INTEREST_PCT",
                                     "EFFECTIVE_INTEREST_RATE", "INTEREST_TO_INCOME_RATIO", "INTEREST_COVERAGE",
                                     "PAYMENT_TO_LOAN_RATIO", "PRINCIPAL_SHARE_PER_MONTH", "LOAN_TO_INCOME_EXPOSURE",
                                     "TOTAL_COST_TO_INCOME_EXPOSURE", "INTEREST_SHARE_TOTAL", "PRINCIPAL_SHARE_TOTAL",
                                     "PRINCIPAL_TO_INTEREST_RATIO"]
            for feat in derived_features_list:
                numeric_inputs[feat] = 0.0


    # -----------------------------
    # Charges / Posting
    # -----------------------------
    charge_cols = ["HAS_CHRG", "CHRG_CODE", "POSTING_RESTRICT", "JOINT_SEPARAT"]
    with st.expander("💳 Charges / Posting", expanded=False):
        for col in charge_cols:
            if col == "HAS_CHRG":
                val = st.checkbox("HAS_CHRG", value=bool(defaults.get(col, 0)), key=f"chk_{col}")
                numeric_inputs[col] = 1 if val else 0
            elif col in categorical_cols:
                le = encoders.get(col)
                if le:
                    # Get the categorical default (mode) for this column
                    categorical_default = defaults.get(f"{col}_categorical", le.classes_[0])
                    # Find the index of the default value in the classes
                    try:
                        default_index = list(le.classes_).index(categorical_default)
                    except ValueError:
                        default_index = 0  # fallback if not found

                    sel = st.selectbox(col, list(le.classes_), index=default_index, key=f"cat_{col}")
                    cat_inputs[col] = str(sel)
            elif col in features:
                val = st.number_input(col, value=float(defaults.get(col, 0.0)), key=f"num_{col}")
                numeric_inputs[col] = val

    return cat_inputs, date_inputs, numeric_inputs


def assemble_feature_vector(features, cat_inputs, date_inputs, numeric_inputs, encoders, metadata):
    """
    Build final feature vector in the exact order `features` expects:
      - For categorical features: use encoder.transform([string])[0]
      - For date-derived features: compute from date_inputs
      - For numeric features: use numeric_inputs
    Returns DataFrame with single row (columns = features)
    """
    row = []
    date_derived_suffixes = ["_YEAR", "_MONTH", "_DAY", "_DAYOFWEEK", "_DAYS_SINCE"]
    date_cols = metadata.get("date_cols", [])
    categorical_cols = metadata.get("categorical_cols", [])

    # precompute date-derived dict
    date_derived = {}
    for dcol, dt in date_inputs.items():
        # dt is a pd.Timestamp
        y = int(dt.year)
        m = int(dt.month)
        d = int(dt.day)
        dow = int(dt.dayofweek)
        days_since = int((dt - pd.Timestamp("1980-01-01")).days)
        date_derived[f"{dcol}_YEAR"] = y
        date_derived[f"{dcol}_MONTH"] = m
        date_derived[f"{dcol}_DAY"] = d
        date_derived[f"{dcol}_DAYOFWEEK"] = dow
        date_derived[f"{dcol}_DAYS_SINCE"] = days_since

    for feat in features:
        if feat in categorical_cols:
            raw_val = cat_inputs.get(feat, "")
            le = encoders.get(feat)
            if le is None:
                # cannot encode -> fallback to 0
                encoded = 0
            else:
                # encoder was fit on strings (we used astype(str) in training). If the chosen raw_val
                # is not in the encoder classes_ (rare), attempt to handle by adding mapping to 'nan_missing' if present
                if raw_val in le.classes_:
                    encoded = int(le.transform([raw_val])[0])
                else:
                    # if there was a 'nan_missing' used during training, use that code; else fallback to 0
                    if "nan_missing" in le.classes_:
                        encoded = int(le.transform(["nan_missing"])[0])
                    else:
                        encoded = 0
            row.append(encoded)
        elif feat in date_derived:
            row.append(float(date_derived[feat]))
        else:
            # numeric feature
            val = numeric_inputs.get(feat, 0.0)
            row.append(float(val))

    X_new = pd.DataFrame([row], columns=features, dtype=float)
    return X_new


def preprocess_paste_input(raw_text, features, encoders, metadata, defaults):
    from io import StringIO
    try:
        df_input = pd.read_csv(StringIO(raw_text))
    except:
        df_input = pd.read_csv(StringIO(raw_text), header=None)
        df_input.columns = features
    df_copy = df_input.copy()

    categorical_cols = metadata.get("categorical_cols", [])
    date_cols = metadata.get("date_cols", [])

    # --- encode categorical safely ---
    for col, le in encoders.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).fillna("nan_missing")
            df_copy[col] = df_copy[col].apply(
                lambda x: x if x in le.classes_ else "nan_missing" if "nan_missing" in le.classes_ else le.classes_[0]
            )
            df_copy[col] = le.transform(df_copy[col])

    # --- handle dates ---
    date_derived_suffixes = ["_YEAR", "_MONTH", "_DAY", "_DAYOFWEEK", "_DAYS_SINCE"]
    for dcol in date_cols:
        if dcol in df_copy.columns:
            df_copy[dcol] = pd.to_datetime(df_copy[dcol], errors="coerce").fillna(pd.Timestamp.today())
            df_copy[f"{dcol}_YEAR"] = df_copy[dcol].dt.year
            df_copy[f"{dcol}_MONTH"] = df_copy[dcol].dt.month
            df_copy[f"{dcol}_DAY"] = df_copy[dcol].dt.day
            df_copy[f"{dcol}_DAYOFWEEK"] = df_copy[dcol].dt.dayofweek
            df_copy[f"{dcol}_DAYS_SINCE"] = (df_copy[dcol] - pd.Timestamp("1980-01-01")).dt.days

    # drop original dates
    df_copy.drop(columns=[c for c in date_cols if c in df_copy.columns], inplace=True)

    # Calculate derived loan values if needed columns are present
    if all(col in df_copy.columns for col in ["LOAN_AMOUNT", "INTEREST_RATE", "LOAN_MONTHS"]):
        P = df_copy["LOAN_AMOUNT"]
        r = df_copy["INTEREST_RATE"] / 100  # Convert percentage to decimal
        n = df_copy["LOAN_MONTHS"]

        # Vectorized calculation
        monthly_payment = np.where(
            r == 0,
            P / n,
            P * (r / 12) / (1 - (1 + r / 12) ** -n)
        )

        total_interest = monthly_payment * n - P
        total_loan = P + total_interest

        df_copy["TOT_INTEREST_AMT"] = total_interest
        df_copy["TOTAL_LOAN_AMOUNT"] = total_loan
        df_copy["MONTHLY_PAYMENT"] = monthly_payment

        # Calculate all additional derived features
        if "TOTAL_MONTHLY_INCOME" in df_copy.columns:
            income = df_copy["TOTAL_MONTHLY_INCOME"]
            df_copy["DTI"] = np.where(income > 0, total_loan / income, 0)
            df_copy["MONTHLY_PAYMENT_BURDEN"] = np.where(income > 0, monthly_payment / income, 0)
            df_copy["ANNUALIZED_PAYMENT_BURDEN"] = np.where(income > 0, (monthly_payment * 12) / income, 0)
            df_copy["INTEREST_TO_INCOME_RATIO"] = np.where((income > 0) & (n > 0),
                                                           total_interest / (income * (n / 12)), 0)
            df_copy["LOAN_TO_INCOME_EXPOSURE"] = np.where((income > 0) & (n > 0),
                                                          P / (income * n), 0)
            df_copy["TOTAL_COST_TO_INCOME_EXPOSURE"] = np.where((income > 0) & (n > 0),
                                                                total_loan / (income * n), 0)

        # Calculate other ratios
        df_copy["CUMULATIVE_INTEREST_PCT"] = np.where(P > 0, total_interest / P, 0)
        df_copy["PAYMENT_TO_LOAN_RATIO"] = np.where(P > 0, monthly_payment / P, 0)
        df_copy["PRINCIPAL_SHARE_TOTAL"] = np.where(total_loan > 0, P / total_loan, 0)
        df_copy["PRINCIPAL_SHARE_PER_MONTH"] = np.where((n > 0) & (monthly_payment > 0),
                                                        (P / n) / monthly_payment, 0)
        df_copy["EFFECTIVE_INTEREST_RATE"] = np.where((P > 0) & (n > 0),
                                                      total_interest / (P * (n / 12)), 0)
        df_copy["INTEREST_COVERAGE"] = np.where(total_interest > 0, total_loan / total_interest, 0)
        df_copy["PRINCIPAL_TO_INTEREST_RATIO"] = np.where(total_interest > 0, P / total_interest, 0)
        df_copy["INTEREST_SHARE_TOTAL"] = np.where(total_loan > 0, total_interest / total_loan, 0)

    # fill missing numeric features
    for feat in features:
        if feat not in df_copy.columns:
            df_copy[feat] = float(defaults.get(feat, 0.0))

    # reorder columns
    df_copy = df_copy[features]

    return df_copy


def main():
    st.markdown("""
    <div class="form-container">
        <p style="font-size: 1.1rem; color: #64748b; text-align: center;">
            🎯 Enter customer information using human-friendly inputs. Our advanced ML model will
            process and predict loan approval risk with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    try:
        model, scaler, pca, features, defaults, encoders, metadata = load_artifacts()
    except Exception as e:
        st.error("❌ Could not load artifacts. Make sure you've run the training script to create them.")
        st.info(str(e))
        st.stop()

    # Enhanced success message
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="margin: 0; color: white;">✅ Artifacts loaded successfully!</h4>
        <p style="margin: 0.5rem 0 0 0; color: white;">
            Model: <code style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 4px;">{ARTIFACT_FILES['model']}</code> •
            Features: <strong>{len(features)}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if mode == "Human-friendly input":
        # -----------------------------
        # Human-friendly inputs
        # -----------------------------
        cat_inputs, date_inputs, numeric_inputs = build_ui_inputs(features, defaults, encoders, metadata)

        st.markdown("---")

        # Enhanced preview section
        if st.checkbox("🔍 Preview assembled feature vector", value=False):
            try:
                X_preview = assemble_feature_vector(features, cat_inputs, date_inputs, numeric_inputs, encoders, metadata)
                st.markdown("### 📋 Feature Vector Preview")
                st.dataframe(X_preview.T.rename(columns={0: "value"}), use_container_width=True)
            except Exception as e:
                st.error("❌ Could not assemble preview: " + str(e))

        # Enhanced predict button
        if st.button("🚀 Predict Loan Risk", type="primary", use_container_width=True):
            try:
                with st.spinner("🔄 Processing prediction..."):
                    X_new = assemble_feature_vector(features, cat_inputs, date_inputs, numeric_inputs, encoders, metadata)
                    # Apply scaler and pca then predict
                    X_scaled = scaler.transform(X_new)
                    X_pca = pca.transform(X_scaled)
                    proba = model.predict_proba(X_pca)[0][1] if hasattr(model, "predict_proba") else None
                    pred = int(model.predict(X_pca)[0])

                if pred == 1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                        <h2 style="margin: 0; color: white;">✅ LOAN APPROVED</h2>
                        <p style="margin: 1rem 0 0 0; font-size: 1.2rem; color: white;">
                            Probability Score: <strong>{proba:.4f}</strong>
                        </p>
                    </div>
                    """ if proba is not None else """
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                        <h2 style="margin: 0; color: white;">✅ LOAN APPROVED</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                                color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                        <h2 style="margin: 0; color: white;">❌ LOAN REJECTED</h2>
                        <p style="margin: 1rem 0 0 0; font-size: 1.2rem; color: white;">
                            Probability Score: <strong>{proba:.4f}</strong>
                        </p>
                    </div>
                    """ if proba is not None else """
                    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                                color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                        <h2 style="margin: 0; color: white;">❌ LOAN REJECTED</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("🔍 Show technical details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Scaled Shape", f"{X_scaled.shape[0]}×{X_scaled.shape[1]}")
                    with col2:
                        st.metric("PCA Shape", f"{X_pca.shape[0]}×{X_pca.shape[1]}")
                    with col3:
                        st.metric("Features Used", len(features))

                    st.write("**PCA Vector (first 10 components):**")
                    st.code(X_pca[0][:10].tolist())
            except Exception as e:
                st.exception("❌ Error during prediction: " + str(e))


    elif mode == "Paste a CSV row":
        # Enhanced paste input section
        st.markdown("### 📝 Quick CSV Input")
        st.markdown("""
        <div class="form-container">
            <p>Paste a CSV row below. The system will automatically preprocess and predict.</p>
        </div>
        """, unsafe_allow_html=True)

        paste_input = st.text_area(
            "📋 CSV Data",
            height=150,
            placeholder="RESIDENCE,BIRTHPLACE,NATIONALITY,...",
            help="Paste your CSV row here (header optional)"
        )

        paste_button = st.button("🚀 Predict from CSV", type="primary", use_container_width=True)

        if paste_button:
            if not paste_input.strip():
                st.warning("⚠️ Please paste a CSV row first.")
            else:
                try:
                    with st.spinner("🔄 Processing CSV data..."):
                        X_new = preprocess_paste_input(paste_input, features, encoders, metadata, defaults)
                        X_scaled = scaler.transform(X_new)
                        X_pca = pca.transform(X_scaled)
                        proba = model.predict_proba(X_pca)[0][1] if hasattr(model, "predict_proba") else None
                        pred = int(model.predict(X_pca)[0])

                    # Same enhanced result display as above
                    if pred == 1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                    color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                            <h2 style="margin: 0; color: white;">✅ LOAN APPROVED</h2>
                            <p style="margin: 1rem 0 0 0; font-size: 1.2rem; color: white;">
                                Probability Score: <strong>{proba:.4f}</strong>
                            </p>
                        </div>
                        """ if proba is not None else """
                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                    color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                            <h2 style="margin: 0; color: white;">✅ LOAN APPROVED</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                                    color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                            <h2 style="margin: 0; color: white;">❌ LOAN REJECTED</h2>
                            <p style="margin: 1rem 0 0 0; font-size: 1.2rem; color: white;">
                                Probability Score: <strong>{proba:.4f}</strong>
                            </p>
                        </div>
                        """ if proba is not None else """
                        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                                    color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                            <h2 style="margin: 0; color: white;">❌ LOAN REJECTED</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("🔍 Show technical details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Scaled Shape", f"{X_scaled.shape[0]}×{X_scaled.shape[1]}")
                        with col2:
                            st.metric("PCA Shape", f"{X_pca.shape[0]}×{X_pca.shape[1]}")
                        with col3:
                            st.metric("Components", 100)

                        st.write("**PCA Vector (first 100 components):**")
                        st.code(X_pca[0][:100].tolist())
                except Exception as e:
                    st.exception("❌ Error processing pasted row: " + str(e))


if __name__ == "__main__":
    main()
