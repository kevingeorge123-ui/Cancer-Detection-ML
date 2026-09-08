import streamlit as st
import pandas as pd

st.title("Kevin C George - Breast Cancer Detection")

st.write("Breast Cancer Detection using Machine Learning")

df = pd.read_csv("brca.csv")

st.write("Dataset loaded successfully!")

st.write(df.head())

