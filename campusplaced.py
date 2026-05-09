import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Placement Prediction Dashboard",
    layout="wide"
)

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("Placement_Data_Full_Class.csv")

model = joblib.load("placement_updatedmodel.pkl")
# =========================================
# TITLE
# =========================================

st.title("🎓 Campus Placement Prediction Dashboard")

st.markdown("---")

# =========================================
# KPI CARDS
# =========================================

total_students = len(df)

placed_students = len(
    df[df['status'] == 'Placed']
)

not_placed = len(
    df[df['status'] == 'Not Placed']
)

placement_rate = round(
    (placed_students / total_students) * 100,
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    total_students
)

col2.metric(
    "Placed Students",
    placed_students
)

col3.metric(
    "Not Placed",
    not_placed
)

col4.metric(
    "Placement Rate",
    f"{placement_rate}%"
)

st.markdown("---")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("📌 Student Details")

# Numerical Inputs

ssc_p = st.sidebar.slider(
    "10th Percentage",
    40,
    100,
    70
)

hsc_p = st.sidebar.slider(
    "12th Percentage",
    40,
    100,
    70
)

degree_p = st.sidebar.slider(
    "Degree Percentage",
    40,
    100,
    70
)

etest_p = st.sidebar.slider(
    "Employability Test Percentage",
    40,
    100,
    70
)

mba_p = st.sidebar.slider(
    "MBA Percentage",
    40,
    100,
    70
)

# Categorical Inputs

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

workex = st.sidebar.selectbox(
    "Work Experience",
    ["Yes", "No"]
)

# =========================================
# PREDICTION
# =========================================

if st.sidebar.button("Predict Placement"):

    # Convert categorical values into encoded format

    gender = 1 if gender == "Male" else 0

    workex = 1 if workex == "Yes" else 0

    # Create DataFrame matching training columns

    input_data = pd.DataFrame({
        'gender': [gender],
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'workex': [workex],
        'etest_p': [etest_p],
        'mba_p': [mba_p],
        
    })

    # Prediction

    prediction = model.predict(input_data)

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Student Will Be Placed")

    else:
        st.error("❌ Student May Not Be Placed")

st.markdown("---")

# =========================================
# ANALYTICS SECTION
# =========================================

st.subheader("📊 Placement Analytics")

col1, col2 = st.columns(2)

# =========================================
# PLACEMENT DISTRIBUTION
# =========================================

with col1:

    fig, ax = plt.subplots()

    sns.countplot(
        x='status',
        data=df,
        ax=ax
    )

    ax.set_title("Placement Distribution")

    st.pyplot(fig)

# =========================================
# DEGREE DISTRIBUTION
# =========================================

with col2:

    fig, ax = plt.subplots()

    sns.histplot(
        df['degree_p'],
        kde=True,
        ax=ax
    )

    ax.set_title("Degree Percentage Distribution")

    st.pyplot(fig)

# =========================================
# WORK EXPERIENCE ANALYSIS
# =========================================

st.subheader("💼 Work Experience vs Placement")

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    x='workex',
    hue='status',
    data=df,
    ax=ax
)

st.pyplot(fig)

# =========================================
# GENDER ANALYSIS
# =========================================

st.subheader("👨‍🎓 Gender-wise Placement Analysis")

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    x='gender',
    hue='status',
    data=df,
    ax=ax
)

st.pyplot(fig)

# =========================================
# CORRELATION HEATMAP
# =========================================

st.subheader("🔥 Correlation Heatmap")

# Encode temporary copy for heatmap

temp_df = df.copy()

categorical_cols = temp_df.select_dtypes(include='object').columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in categorical_cols:
    temp_df[col] = le.fit_transform(temp_df[col])

corr = temp_df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    ax=ax
)

st.pyplot(fig)

# =========================================
# DATA PREVIEW
# =========================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption(
    "Developed using Python, Machine Learning, Streamlit & Joblib"
)