import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="🏠 House Rent Predictor by Harshit Saxena",
    layout="wide",
    page_icon="🏘️"
)

st.title("🏠 House Rent Predictor by Harshit Saxena")
st.caption("B.Tech CSE | Maharaja Surajmal Institute of Technology (MSIT)")
st.markdown("---")

st.write("### 🔍 Predict monthly house rent using ML models — Linear Regression & Random Forest Regressor.")
st.info("💡 *Trained on real housing data using both linear and ensemble learning techniques.*")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

DATA_PATH = "house_rent_data.csv"

try:
    df = load_data(DATA_PATH)
    st.success(f"✅ Dataset loaded successfully with {df.shape[0]:,} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    st.error("❌ Dataset 'house_rent_data.csv' not found. Please upload it to the app folder before running.")
    st.stop()

with st.expander("📂 View Dataset Overview"):
    st.dataframe(df.head(10))
    st.write("### Column Summary")
    col_summary = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(col_summary)

# ------------------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------------------
@st.cache_resource
def train_models(df):
    if "Rent" not in df.columns:
        st.error("❌ The dataset must contain a 'Rent' column as the target variable.")
        st.stop()

    X = df.drop(columns="Rent")
    y = df["Rent"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("numerical", "passthrough", num_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Random Forest Model
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            max_depth=20
        ))
    ])
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

    metrics = {
        "Linear Regression": {"R²": r2_score(y_test, y_pred_lr), "RMSE": rmse(y_test, y_pred_lr)},
        "Random Forest": {"R²": r2_score(y_test, y_pred_rf), "RMSE": rmse(y_test, y_pred_rf)}
    }

    return lr_model, rf_model, metrics, cat_cols, num_cols, X_train, y_train


with st.spinner("⏳ Training models... Please wait (optimized for large datasets)..."):
    lr_model, rf_model, metrics, cat_cols, num_cols, X_train, y_train = train_models(df)

st.success("✅ Models trained successfully!")
st.markdown(f"**📈 Random Forest (R² = {metrics['Random Forest']['R²']:.3f}) | Linear Regression (R² = {metrics['Linear Regression']['R²']:.3f})**")

# ------------------------------------------------------------
# SIDEBAR - Model Metrics
# ------------------------------------------------------------
st.sidebar.header("📊 Model Performance (Test Set)")
st.sidebar.write(f"**Random Forest** — R²: {metrics['Random Forest']['R²']:.3f}, RMSE: {metrics['Random Forest']['RMSE']:.2f}")
st.sidebar.write(f"**Linear Regression** — R²: {metrics['Linear Regression']['R²']:.3f}, RMSE: {metrics['Linear Regression']['RMSE']:.2f}")
st.sidebar.markdown("---")
st.sidebar.write("**Project:** House Rent Predictor  \n**Author:** Harshit Saxena  \n**Institute:** MSIT, CSE Dept.")

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["🏘️ Predict Rent", "📈 Model Insights"])

# ------------------------------------------------------------
# TAB 1: PREDICTION
# ------------------------------------------------------------
with tab1:
    st.header("🏘️ Enter Property Details for Prediction")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", sorted(df["City"].unique()))
        area = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=int(df["Area (sqft)"].median()), step=10)
        bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1)

    with col2:
        bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=1, step=1)
        furnish = st.selectbox("Furnishing Status", sorted(df["Furnishing Status"].unique()))
        tenant = st.selectbox("Tenant Preferred", sorted(df["Tenant Preferred"].unique()))

    st.markdown("---")

    model_choice = st.radio("Select Prediction Model", ["Random Forest", "Linear Regression"])
    predict_btn = st.button("💰 Predict Rent")

    if predict_btn:
        input_df = pd.DataFrame({
            "City": [city],
            "Area (sqft)": [area],
            "BHK": [bhk],
            "Bathroom": [bathroom],
            "Furnishing Status": [furnish],
            "Tenant Preferred": [tenant]
        })

        model = rf_model if model_choice == "Random Forest" else lr_model
        predicted_rent = model.predict(input_df)[0]

        st.success(f"💰 **Predicted Monthly Rent: ₹ {predicted_rent:,.0f}**")

        with st.expander("📋 View Input Summary"):
            st.write(input_df.T)

        st.info(f"📈 Model Used: **{model_choice}** — R²: {metrics[model_choice]['R²']:.3f}, RMSE: {metrics[model_choice]['RMSE']:.2f}")

# ------------------------------------------------------------
# TAB 2: MODEL INSIGHTS
# ------------------------------------------------------------
with tab2:
    st.subheader("📊 Model Comparison")
    st.dataframe(pd.DataFrame(metrics).T.style.format({"R²": "{:.3f}", "RMSE": "{:.2f}"}))

    st.markdown("---")
    st.subheader("📈 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns to display correlation heatmap.")

    st.markdown("---")
    st.subheader("🌲 Random Forest Feature Importance")
    rf_reg = rf_model.named_steps["regressor"]
    if hasattr(rf_reg, "feature_importances_"):
        ohe = rf_model.named_steps["preprocessor"].named_transformers_["categorical"]
        cat_features = ohe.get_feature_names_out(cat_cols)
        all_features = np.concatenate([cat_features, num_cols])
        importances = pd.Series(rf_reg.feature_importances_, index=all_features).sort_values(ascending=False)[:10]
        st.bar_chart(importances)
    else:
        st.info("Feature importance unavailable for this model.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ by **Harshit Saxena (B.Tech CSE, MSIT)**")
st.caption("Optimized ML web app using Streamlit, Scikit-learn & Random Forest Regression.")
