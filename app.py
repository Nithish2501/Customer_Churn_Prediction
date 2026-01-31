import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import requests

# -----------------------------
# Page Config & Styling
# -----------------------------
st.set_page_config(page_title="Churn Prediction Pro", page_icon="📉", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        color: #fff;
    }
    .title {
        font-size:42px !important;
        font-weight:800;
        text-align:center;
        color: #F39C12;
        margin-bottom:10px;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        margin-bottom:30px;
        color: #ECF0F1;
    }
    .metric-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: rgba(255,255,255,0.1);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Lottie Animation Loader
# -----------------------------
def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_fyye8szy.json"
try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.write("Please install `streamlit-lottie` using: pip install streamlit-lottie")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = le.fit_transform(df[col])
    return df

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    return rf, scaler, X.columns, X_test, y_test

# -----------------------------
# App Layout
# -----------------------------
st.markdown('<div class="title">📉 Customer Churn Prediction Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An AI-Powered Dashboard to Predict & Prevent Customer Loss</div>', unsafe_allow_html=True)

df = load_data()
model, scaler, feature_names, X_test, y_test = train_model(df)

# Navigation
menu = st.sidebar.radio("📂 Navigate", ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ About"])

# -----------------------------
# Home Page
# -----------------------------
if menu == "🏠 Home":
    st.write("### Welcome to Churn Prediction Pro")
    st.write("This dashboard helps businesses *predict churn*, analyze key drivers, and take proactive actions to improve retention.")
    lottie_animation = load_lottie(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, height=300, key="churn")
    col1, col2, col3 = st.columns(3)
    churn_rate = df["Churn"].mean() * 100
    roc_auc = roc_auc_score(y_test, model.predict(X_test))
    col1.markdown(f"<div class='metric-card'><h3>Churn Rate</h3><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>ROC-AUC</h3><h2>{roc_auc:.2f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>Dataset Size</h3><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)

# -----------------------------
# Prediction Page (Updated)
# -----------------------------
elif menu == "🔮 Prediction":
    st.subheader("Enter Customer Details")

    # Store user predictions in session
    if "user_predictions" not in st.session_state:
        st.session_state.user_predictions = pd.DataFrame()

    # Label mappings
    label_mappings = {
        "gender": {0: "Female", 1: "Male"},
        "SeniorCitizen": {0: "No", 1: "Yes"},
        "Partner": {0: "No", 1: "Yes"},
        "Dependents": {0: "No", 1: "Yes"},
        "PhoneService": {0: "No", 1: "Yes"},
        "MultipleLines": {0: "No", 1: "Yes", 2: "No phone service"},
        "InternetService": {0: "DSL", 1: "Fiber optic", 2: "No"},
        "OnlineSecurity": {0: "No", 1: "Yes", 2: "No internet service"},
        "OnlineBackup": {0: "No", 1: "Yes", 2: "No internet service"},
        "DeviceProtection": {0: "No", 1: "Yes", 2: "No internet service"},
        "TechSupport": {0: "No", 1: "Yes", 2: "No internet service"},
        "StreamingTV": {0: "No", 1: "Yes", 2: "No internet service"},
        "StreamingMovies": {0: "No", 1: "Yes", 2: "No internet service"},
        "Contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
        "PaymentMethod": {0: "Bank transfer (automatic)", 1: "Credit card (automatic)", 2: "Electronic check", 3: "Mailed check"},
        "PaperlessBilling": {0: "No", 1: "Yes"}
    }

    customer_data = {}
    for col in feature_names:
        if col in label_mappings:
            customer_data[col] = st.selectbox(col, list(label_mappings[col].values()))
        elif df[col].nunique() <= 10:
            options = sorted(df[col].unique())
            customer_data[col] = st.selectbox(col, options)
        else:
            customer_data[col] = st.slider(col, int(df[col].min()), int(df[col].max()), int(df[col].median()))

    # Convert to numeric
    customer_input = {}
    for col in customer_data:
        if col in label_mappings:
            inv_map = {v: k for k, v in label_mappings[col].items()}
            customer_input[col] = inv_map[customer_data[col]]
        else:
            customer_input[col] = customer_data[col]

    customer_df = pd.DataFrame([customer_input])
    customer_scaled = scaler.transform(customer_df)

    if st.button("🔮 Predict Now"):
        prediction = model.predict(customer_scaled)[0]
        prob = model.predict_proba(customer_scaled)[0][1]

        st.markdown("### 📋 Customer Profile")
        st.dataframe(pd.DataFrame([customer_data]))

        if prediction == 1:
            st.error(f"⚠ High Risk! This customer is **LIKELY to CHURN** (probability {prob:.2f})")
        else:
            st.success(f"✅ Safe! This customer is **NOT likely to churn** (probability {1-prob:.2f})")

        # Save prediction
        customer_input["Churn"] = prediction
        st.session_state.user_predictions = pd.concat(
            [st.session_state.user_predictions, pd.DataFrame([customer_input])],
            ignore_index=True
        )

        st.success("✅ Prediction saved! Go to 📊 Analytics to see the updated charts.")

# -----------------------------
# Analytics Page (Updated)
# -----------------------------
elif menu == "📊 Analytics":
    st.subheader("Churn Analytics")

    # Merge live predictions with dataset
    if "user_predictions" in st.session_state and not st.session_state.user_predictions.empty:
        df_live = pd.concat([df, st.session_state.user_predictions], ignore_index=True)
    else:
        df_live = df.copy()

    # Churn Distribution
    fig = px.pie(df_live, names="Churn", title="Updated Churn Distribution",
                 color_discrete_sequence=["#27AE60", "#E74C3C"])
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)
    fig = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation="h",
                 title="Top 10 Features Driving Churn")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (Model Performance)")
    st.pyplot(fig)

    st.info(f"📊 Showing analytics for {len(df_live)} total records (including {len(st.session_state.user_predictions)} new predictions).")

# -----------------------------
# About Page
# -----------------------------
elif menu == "ℹ About":
    st.write("### About this App")
    st.info("""
    *Churn Prediction Pro* is an AI-powered demo app built with *Streamlit* and *Scikit-learn*.  
    It enables businesses to:
    - Predict customer churn in real time  
    - Analyze key factors influencing churn  
    - Visualize customer insights in an interactive dashboard  

    💡 Built with ❤ using Python, Streamlit, and Machine Learning.
    """)
