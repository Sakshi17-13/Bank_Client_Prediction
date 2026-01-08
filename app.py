# ======================= app.py =======================
import streamlit as st
import pandas as pd
import joblib

# -------------------- LOAD SAVED OBJECTS --------------------
model = joblib.load("bank_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

CAT_COLS = list(encoder.feature_names_in_)
NUM_COLS = list(scaler.feature_names_in_)

# -------------------- UI CONFIG --------------------
st.set_page_config(
    page_title="Bank Client Prediction",
    layout="wide",
    page_icon="🏦"
)

# -------------------- HEADER --------------------
st.title("🏦 Bank Client Subscription Prediction")
st.markdown(
    "Predict whether a bank client will **subscribe to a term deposit** using ML."
)
st.markdown("---")

# -------------------- INPUT FORM --------------------
with st.form("client_input", clear_on_submit=False):
    st.subheader("Client Information")
    
    # Use columns for better UI
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 95, 30)
        job = st.selectbox("Job", encoder.categories_[0])
        marital = st.selectbox("Marital Status", encoder.categories_[1])
        education = st.selectbox("Education", encoder.categories_[2])
    with col2:
        default = st.selectbox("Credit Default", encoder.categories_[3])
        housing = st.selectbox("Housing Loan", encoder.categories_[4])
        loan = st.selectbox("Personal Loan", encoder.categories_[5])
        contact = st.selectbox("Contact Type", encoder.categories_[6])
    with col3:
        month = st.selectbox("Month", encoder.categories_[7])
        day_of_week = st.selectbox("Day of Week", encoder.categories_[8])
        poutcome = st.selectbox("Previous Outcome", encoder.categories_[9])
    
    st.subheader("Campaign & Economic Indicators")
    col4, col5, col6 = st.columns(3)
    with col4:
        duration = st.number_input("Call Duration (seconds)", value=300)
        campaign = st.number_input("Campaign Contacts", value=1)
    with col5:
        pdays = st.number_input("Days Since Last Contact", value=999)
        previous = st.number_input("Previous Contacts", value=0)
    with col6:
        cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
        euribor3m = st.number_input("Euribor 3M", value=1.3)
    
    submitted = st.form_submit_button("Predict")

# -------------------- PREDICTION --------------------
if submitted:
    # Prepare input data
    row = {
        "age": age,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "emp.var.rate": 1.1,       # placeholder for required columns
        "nr.employed": 5000        # placeholder
    }

    row.update({
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "poutcome": poutcome
    })

    input_data = pd.DataFrame([row])

    # Encode categorical features
    input_data[CAT_COLS] = encoder.transform(input_data[CAT_COLS])
    # Scale numeric features
    input_data[NUM_COLS] = scaler.transform(input_data[NUM_COLS])
    # Arrange columns in correct order
    input_data = input_data[features]

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max() * 100

    # -------------------- DISPLAY RESULTS --------------------
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ The customer WILL subscribe to a term deposit.")
    else:
        st.error(f"❌ The customer will NOT subscribe.")

    st.metric("Prediction Probability", f"{probability:.2f}%")

    st.info(
        "💡 This prediction is based on a machine learning stacked ensemble model "
        "trained on historical bank client data."
    )
