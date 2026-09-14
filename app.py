import streamlit as st
import pandas as pd

st.title("Breast Cancer Detection")

st.write("Breast Cancer Detection using Machine Learning")

df = pd.read_csv("brca.csv")

st.write("Dataset loaded successfully!")

st.write(df.head())

st.write("Dataset Shape:", df.shape)

st.write("Column Names:", df.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


