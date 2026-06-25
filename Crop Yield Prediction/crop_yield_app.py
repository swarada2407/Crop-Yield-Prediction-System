
# Crop Yield Prediction Web App (Soft Faint Green Theme)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Page Configuration

st.set_page_config(page_title="🌾 Crop Yield Predictor", page_icon="🌱", layout="wide")


#  Custom Page Styling (Faint Green, No Dark Colors)

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #E8F5E9; /* Faint pastel green background */
}
[data-testid="stHeader"] {
    background: #A5D6A7; /* Soft green header */
}
h1, h2, h3, h4, h5 {
    color: #2E7D32 !important; /* Fresh medium green text */
}
div.stButton > button {
    background-color: #66BB6A;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    border: none;
}
div.stButton > button:hover {
    background-color: #57A05A;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #C8E6C9; /* Light green sidebar */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

#  Title Section

st.title("🌾 Crop Yield Prediction Dashboard")
st.write("Predict crop yield based on environmental and agricultural factors using machine learning.")


# Step 1️⃣: Load Dataset

df = pd.read_csv("crop_yield.csv")
st.success("✅ Dataset Loaded Successfully!")

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())


# Step 2️⃣: Clean Data

df.columns = df.columns.str.strip().str.lower()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.lower()
df = df.fillna(df.mean(numeric_only=True))


# Step 3️⃣: Identify Target Column

target_col = None
for col in ['yield', 'crop_yield', 'hg/ha_yield', 'production']:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    st.error("❌ Target column not found! Please rename your yield column to 'yield' or 'crop_yield'.")
    st.stop()


# Step 4️⃣: Encode Categorical Columns

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Step 5️⃣: Split Data

X = df.drop(columns=[target_col])
y = df[target_col]


# Step 6️⃣: Train Model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


# Step 7️⃣: Evaluate Model

y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Step 8️⃣: Dashboard Summary

st.markdown("## 🌱 Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
col2.metric("Features", len(df.columns) - 1)
col3.metric("Unique Crops", len(label_encoders[list(label_encoders.keys())[0]].classes_) if label_encoders else 0)
col4.metric("Average Yield", f"{df[target_col].mean():.2f}")


# Step 9️⃣: Data Insights Tabs

tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Correlation Heatmap", "🌿 Feature Importance"])

with tab1:
    st.subheader("📊 Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error", f"{mae:.2f}")
    c2.metric("Root Mean Square Error", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.2f}")

with tab2:
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("🌿 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.sort_values(ascending=False).plot(kind='bar', color='#66BB6A', ax=ax)
    st.pyplot(fig)


# Step 🔟: Prediction Section

st.markdown("---")
st.subheader("🧮 Predict Crop Yield")

with st.form("prediction_form"):
    st.write("Enter details to predict crop yield:")

    user_inputs = {}
    for col_name in X.columns:
        if col_name in label_encoders:
            options = label_encoders[col_name].classes_
            user_inputs[col_name] = st.selectbox(f"{col_name.capitalize()}", options)
        else:
            user_inputs[col_name] = st.number_input(f"{col_name.capitalize()}", value=float(df[col_name].mean()))

    submitted = st.form_submit_button("🌾 Predict Yield")

    if submitted:
        input_data = []
        for col_name in X.columns:
            if col_name in label_encoders:
                le = label_encoders[col_name]
                input_data.append(le.transform([user_inputs[col_name].lower()])[0])
            else:
                input_data.append(user_inputs[col_name])

        input_df = pd.DataFrame([input_data], columns=X.columns)
        pred_yield = model.predict(input_df)[0]
        st.success(f"🌻 Predicted Crop Yield: **{pred_yield:.2f} kg/ha**")
