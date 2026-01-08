# ======================= app.py =======================
import streamlit as st
import pandas as pd
import joblib

# -------------------- LOAD SAVED OBJECTS --------------------
model = joblib.load("bank_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# EXACT columns used during training
CAT_COLS = list(encoder.feature_names_in_)
NUM_COLS = list(scaler.feature_names_in_)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Bank Client Prediction",
    page_icon="🏦",
    layout="centered"
)

# -------------------- HEADER --------------------
st.title("🏦 Bank Client Subscription Prediction")
st.write(
    "Predict whether a customer is likely to **subscribe to a term deposit** "
    "based on personal, contact, and economic information."
)
st.divider()

# -------------------- INPUT FORM --------------------
with st.form("input_form"):

    st.subheader("👤 Client Details")
    age = st.slider("Age", 18, 95, 30)

    col1, col2 = st.columns(2)
    with col1:
        job = st.selectbox("Job", encoder.categories_[0])
        marital = st.selectbox("Marital Status", encoder.categories_[1])
        education = st.selectbox("Education", encoder.categories_[2])

    with col2:
        default = st.selectbox("Credit Default", encoder.categories_[3])
        housing = st.selectbox("Housing Loan", encoder.categories_[4])
        loan = st.selectbox("Personal Loan", encoder.categories_[5])

    st.subheader("📞 Contact Information")
    col3, col4 = st.columns(2)
    with col3:
        contact = st.selectbox("Contact Type", encoder.categories_[6])
        month = st.selectbox("Contact Month", encoder.categories_[7])
    with col4:
        day_of_week = st.selectbox("Day of Week", encoder.categories_[8])
        poutcome = st.selectbox("Previous Campaign Outcome", encoder.categories_[9])

    st.subheader("📊 Campaign & Economic Indicators")
    duration = st.number_input("Call Duration (seconds)", value=300)
    campaign = st.number_input("Number of Contacts in Campaign", value=1)
    pdays = st.number_input("Days Since Last Contact", value=999)
    previous = st.number_input("Previous Contacts", value=0)

    cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=1.3)

    submitted = st.form_submit_button("🔮 Predict")

# -------------------- PREDICTION --------------------
if submitted:
    with st.spinner("Analyzing customer profile..."):

        # Create full row with ALL numeric columns
        row = {
            "age": age,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "emp.var.rate": 1.1,
            "nr.employed": 5000
        }

        # Add categorical values
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

        # Encode categorical columns
        input_data[CAT_COLS] = encoder.transform(input_data[CAT_COLS])

        # Scale numeric columns
        input_data[NUM_COLS] = scaler.transform(input_data[NUM_COLS])

        # Final correct feature order
        input_data = input_data[features]
        X_final = input_data.to_numpy()

        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final).max() * 100

    st.divider()

    if prediction == 1:
        st.success(f"✅ **Customer WILL subscribe**  \nConfidence: **{probability:.2f}%**")
    else:
        st.error(f"❌ **Customer will NOT subscribe**  \nConfidence: **{probability:.2f}%**")
