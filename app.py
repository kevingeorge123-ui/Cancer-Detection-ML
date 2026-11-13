
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction")
st.write("Breast Cancer Prediction using Machine Learning")

st.warning(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("brca.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("brca.csv was not found. Add it to your project folder.")
    st.stop()

st.success("Dataset loaded successfully!")

# -----------------------------------
# DATASET INFORMATION
# -----------------------------------

st.header("1. Dataset Information")

st.write("First five rows:")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)
st.write("Column Names:", df.columns.tolist())

if "y" not in df.columns:
    st.error("The dataset must contain a target column named 'y'.")
    st.stop()

# -----------------------------------
# PREPARE DATA
# -----------------------------------

st.header("2. Data Preparation")

X = df.drop("y", axis=1)
y = df["y"]

# Convert features to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    st.error(
        "Some features contain missing or non-numeric values. "
        "Clean the dataset before training."
    )
    st.stop()

if y.isnull().any():
    st.error("The target column contains missing values.")
    st.stop()

if y.nunique() != 2:
    st.error("This application requires exactly two target classes.")
    st.stop()

if y.value_counts().min() < 2:
    st.error("Each target class needs at least two samples.")
    st.stop()

# Split dataset
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
except ValueError as error:
    st.error(f"Unable to split dataset: {error}")
    st.stop()

st.write("Training data shape:", X_train.shape)
st.write("Testing data shape:", X_test.shape)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------

st.header("3. Machine Learning Model")

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

st.success("Random Forest model trained successfully!")

# -----------------------------------
# MODEL EVALUATION
# -----------------------------------

st.header("4. Model Evaluation")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.metric(
    label="Model Accuracy",
    value=f"{accuracy * 100:.2f}%"
)

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")

labels = list(model.classes_)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels]
)

st.dataframe(cm_df)

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

st.header("5. Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature").head(15)
)

# -----------------------------------
# USER PREDICTION
# -----------------------------------

st.header("6. Try a Prediction")

st.write(
    "Enter feature values below to test the trained model. "
    "The initial values are dataset medians."
)

st.info(
    "Target class meanings must be verified from the "
    "dataset documentation before interpreting predictions."
)

input_data = {}

with st.form("prediction_form"):

    for column in X.columns:
        default_value = float(X[column].median())

        input_data[column] = st.number_input(
            label=column,
            value=default_value,
            format="%.6f"
        )

    submit = st.form_submit_button("Predict")

if submit:

    input_df = pd.DataFrame(
        [input_data],
        columns=X.columns
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    st.success(f"Predicted target class: {prediction}")

    probability_df = pd.DataFrame({
        "Target Class": model.classes_,
        "Model Score": probabilities
    })

    st.dataframe(probability_df)

    st.caption(
        "These scores are model estimates, not validated "
        "clinical risk probabilities."
    )

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.write("Breast Cancer Prediction | Machine Learning Project")
st.caption("Built with Python, Pandas, Scikit-learn and Streamlit")
st.caption("For educational and research demonstration only.")
