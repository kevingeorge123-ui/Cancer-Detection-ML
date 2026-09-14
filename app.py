import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("Breast Cancer Detection")

st.write("Breast Cancer Detection using Machine Learning")

df = pd.read_csv("brca.csv")

X = df.drop("y", axis=1)
y = df["y"]

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


