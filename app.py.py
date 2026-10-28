import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Student Service Request Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Service Request Monitoring")
st.caption("Demonstration application for predicting possible SLA breaches.")

DATA_PATH = "/content/student_service_requests.csv"
MODEL_PATH = "/content/sample_sla_model.joblib"

# Load demonstration data and model
if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Dataset or model file is missing. Please check the project files.")
    st.stop()

df = pd.read_csv(DATA_PATH)
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a module:",
    [
        "1. Dashboard",
        "2. Dataset Upload",
        "3. Data Exploration",
        "4. SLA Breach Prediction",
        "5. Reports and Limitations"
    ]
)

if page == "1. Dashboard":
    st.header("Request Monitoring Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sample Requests", len(df))
    c2.metric("Recorded SLA Breaches", int((df["SLA_Breach"] == "Yes").sum()))
    c3.metric("No Recorded Breach", int((df["SLA_Breach"] == "No").sum()))

    st.subheader("Sample Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("SLA Breach Distribution")
    st.bar_chart(df["SLA_Breach"].value_counts())

elif page == "2. Dataset Upload":
    st.header("Upload a CSV Dataset")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Dataset preview:")
            st.dataframe(uploaded_df.head(), use_container_width=True)
            st.write("Rows and columns:", uploaded_df.shape)
            st.write("Missing values:")
            st.dataframe(uploaded_df.isnull().sum().rename("Missing Values"))
            st.write("Duplicate rows:", int(uploaded_df.duplicated().sum()))

            missing_cols = [c for c in features if c not in uploaded_df.columns]
            if missing_cols:
                st.error(f"Required columns are missing: {missing_cols}")
            else:
                st.success("Required prediction columns are present.")
                st.info(
                    "This demo uses its bundled model. Uploaded data is previewed "
                    "but does not automatically retrain the model."
                )
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")

elif page == "3. Data Exploration":
    st.header("Data Exploration")
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Requests by Service Type")
    st.bar_chart(df["Service_Type"].value_counts())

    st.subheader("Requests by Priority")
    st.bar_chart(df["Priority"].value_counts())

    st.subheader("SLA Breach by Submission Channel")
    chart_data = pd.crosstab(df["Submission_Channel"], df["SLA_Breach"])
    st.bar_chart(chart_data)

elif page == "4. SLA Breach Prediction":
    st.header("Predict SLA Breach Risk")
    st.warning(
        "Demonstration only: the model was trained on 10 synthetic sample records. "
        "Predictions are not validated for real-world decisions."
    )

    with st.form("prediction_form"):
        service = st.selectbox(
            "Service Type",
            sorted(df["Service_Type"].unique())
        )
        channel = st.selectbox(
            "Submission Channel",
            sorted(df["Submission_Channel"].unique())
        )
        priority = st.selectbox(
            "Priority",
            ["Low", "Medium", "High"]
        )
        documents = st.selectbox(
            "Documents Complete?",
            ["Yes", "No"]
        )
        queue = st.number_input(
            "Queue Length at Submission",
            min_value=0,
            max_value=10000,
            value=5
        )
        days = st.number_input(
            "Requested Processing Days",
            min_value=1,
            max_value=365,
            value=3
        )
        day = st.selectbox(
            "Submission Day",
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
        )
        previous = st.number_input(
            "Previous Request Count",
            min_value=0,
            max_value=10000,
            value=1
        )

        submitted = st.form_submit_button("Predict SLA Breach")

    if submitted:
        row = pd.DataFrame([{
            "Service_Type": service,
            "Submission_Channel": channel,
            "Priority": priority,
            "Documents_Complete": documents,
            "Queue_Length_At_Submission": queue,
            "Requested_Processing_Days": days,
            "Submission_Day": day,
            "Previous_Request_Count": previous
        }])[features]

        prediction = model.predict(row)[0]
        st.subheader("Prediction Result")
        if str(prediction).lower() == "yes":
            st.error("Predicted result: Possible SLA Breach")
        else:
            st.success("Predicted result: No SLA Breach predicted")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = model.classes_
            prob_df = pd.DataFrame({
                "Outcome": classes,
                "Predicted Probability": probabilities
            })
            st.dataframe(prob_df, use_container_width=True)

elif page == "5. Reports and Limitations":
    st.header("Reports and Project Limitations")
    st.subheader("Dataset Summary")
    st.write(f"Number of sample records: {len(df)}")
    st.write("Target variable: SLA_Breach")

    st.subheader("Download Sample Dataset")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_service_requests.csv",
        mime="text/csv"
    )

    st.subheader("Important Limitations")
    st.markdown("""
    - The dataset contains synthetic demonstration records.
    - The model was trained on only 10 records.
    - This is not the original midterm model or dataset.
    - Predictions have not been validated for real-world use.
    - Recover the original midterm notebook, dataset, and saved model
      if the instructor requires reuse of the actual midterm outputs.
    """)
