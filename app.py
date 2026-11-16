import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

TARGET_NAMES = ["y", "diagnosis", "target", "class", "label", "outcome"]


# -----------------------------------
# HELPERS
# -----------------------------------
def find_target(columns):
    """Guess the target column from common names (case-insensitive)."""
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in TARGET_NAMES:
        if name in lowered:
            return lowered[name]
    return columns[-1]


def prepare_features(df, target):
    """Return (X, y, notes). Drops id/empty columns, converts to numbers, fills gaps."""
    notes = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    drop = [c for c in df.columns if c != target and (
        c.lower() in ("id", "index", "patient_id") or c.lower().startswith("unnamed"))]
    if drop:
        df = df.drop(columns=drop)
        notes.append(f"Dropped identifier columns: {drop}")

    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")

    empty = [c for c in X.columns if X[c].isnull().all()]
    if empty:
        X = X.drop(columns=empty)
        notes.append(f"Dropped empty or non-numeric columns: {empty}")

    missing = int(X.isnull().sum().sum())
    if missing:
        X = X.fillna(X.median())
        notes.append(f"Filled {missing} missing values with column medians")
    return X, y, notes


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# -----------------------------------
# PAGE
# -----------------------------------
def main():
    st.title("🩺 Breast Cancer Prediction")
    st.write("Breast Cancer Prediction using Machine Learning")
    st.warning("Educational demonstration only. This application is not a medical diagnostic tool.")

    # Load dataset: from the repo, or let the user upload one
    try:
        df = load_data("brca.csv")
    except FileNotFoundError:
        st.error("brca.csv was not found in the repository. Upload it to GitHub next to app.py, "
                 "or upload it here to test.")
        up = st.file_uploader("Upload a CSV", type="csv")
        if up is None:
            st.stop()
        df = pd.read_csv(up)
    st.success("Dataset loaded successfully!")

    st.header("1. Dataset Information")
    st.dataframe(df.head())
    st.write("Dataset shape:", df.shape)
    st.write("Column names:", df.columns.tolist())

    cols = [str(c).strip() for c in df.columns]
    target = st.selectbox("Target column (what we predict)", cols,
                          index=cols.index(find_target(cols)))

    st.header("2. Data Preparation")
    X, y, notes = prepare_features(df, target)
    for n in notes:
        st.info(n)

    if X.shape[1] == 0:
        st.error("No numeric feature columns were found.")
        st.stop()
    if y.nunique() != 2:
        st.error(f"The target '{target}' must have exactly two classes, "
                 f"but it has {y.nunique()}: {list(y.unique())[:10]}")
        st.stop()
    if y.value_counts().min() < 2:
        st.error("Each target class needs at least two samples.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    st.write("Training data shape:", X_train.shape)
    st.write("Testing data shape:", X_test.shape)

    st.header("3. Machine Learning Model")
    model = train_model(X_train, y_train)
    st.success("Random Forest model trained successfully!")

    st.header("4. Model Evaluation")
    y_pred = model.predict(X_test)
    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    st.dataframe(pd.DataFrame(cm,
                              index=[f"Actual {l}" for l in labels],
                              columns=[f"Predicted {l}" for l in labels]))

    st.header("5. Feature Importance")
    importance = pd.DataFrame({"Feature": X.columns,
                               "Importance": model.feature_importances_}
                              ).sort_values("Importance", ascending=False)
    st.bar_chart(importance.set_index("Feature").head(15))

    st.header("6. Try a Prediction")
    st.write("Enter feature values below. The initial values are dataset medians.")
    st.info("Check the dataset documentation for what each target class means "
            "(for example B = benign, M = malignant) before interpreting predictions.")

    input_data = {}
    with st.form("prediction_form"):
        for column in X.columns:
            input_data[column] = st.number_input(
                label=column, value=float(X[column].median()), format="%.6f")
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([input_data], columns=X.columns)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        st.subheader("Prediction Result")
        st.success(f"Predicted target class: {prediction}")
        st.dataframe(pd.DataFrame({"Target Class": model.classes_,
                                   "Model Score": probabilities}))
        st.caption("These scores are model estimates, not validated clinical risk probabilities.")

    st.divider()
    st.caption("Built with Python, Pandas, Scikit-learn and Streamlit. "
               "For educational and research demonstration only.")


main()
