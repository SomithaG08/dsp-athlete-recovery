import streamlit as st
import requests
import os
import pandas as pd

st.set_page_config(
    page_title="Athlete Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Athlete Recovery Prediction")

st.write("Predict a single athlete with the form, or upload a CSV for a batch prediction.")

st.header("Single prediction")

st.markdown("---")

with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Non-binary"]
    )

    sport = st.selectbox(
        "Sport Type",
        [
            "Endurance",
            "Strength",
            "Team Sport",
            "Mixed",
            "Combat"
        ]
    )

    training_type = st.selectbox(
        "Training Type",
        [
            "Cardio",
            "Strength",
            "HIIT",
            "Rest",
            "Yoga"
        ]
    )

    duration = st.number_input(
        "Training Duration (Minutes)",
        min_value=1,
        max_value=300,
        value=90
    )

    intensity = st.slider(
        "Training Intensity",
        1,
        10,
        7
    )

    sleep = st.slider(
        "Sleep Duration (Hours)",
        0.0,
        12.0,
        8.0
    )

    caffeine = st.number_input(
        "Caffeine Intake (mg)",
        min_value=0,
        max_value=500,
        value=100
    )

    stress = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High"],
        index=0,
    )

    heart_rate = st.number_input(
        "Resting Heart Rate",
        min_value=30,
        max_value=120,
        value=60
    )

    hrv = st.number_input(
        "HRV (ms)",
        min_value=10,
        max_value=200,
        value=85
    )

    mood = st.slider(
        "Mood Score",
        1,
        10,
        8
    )

    soreness = st.slider(
        "Muscle Soreness",
        1,
        10,
        2
    )

    energy = st.slider(
        "Energy Level",
        1,
        10,
        9
    )

    submit = st.form_submit_button("🚀 Predict Recovery Score")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if submit:

    api_url = os.getenv("API_URL", "http://fastapi:8000")
    payload = {
        "Age": age,
        "Gender": gender,
        "Sport_Type": sport,
        "Training_Type": training_type,
        "Training_Duration_Min": duration,
        "Training_Intensity": intensity,
        "Sleep_Duration_Hours": sleep,
        "Caffeine_Intake_mg": caffeine,
        "Stress_Level": stress,
        "Resting_Heart_Rate": heart_rate,
        "HRV_ms": hrv,
        "Mood_Score": mood,
        "Muscle_Soreness": soreness,
        "Energy_Level": energy
    }

    try:

        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:

            prediction = response.json()["prediction"]
            st.success("Prediction completed successfully!")

            st.metric(
                label="🏃 Predicted Recovery Score",
                value=f"{prediction:.2f}"
            )
            result_row = {
                **payload,
                "Predicted_Recovery_Score": prediction,
                "model_version": response.json().get("model_version", "unknown"),
            }
            st.subheader("Features and prediction")
            st.dataframe(pd.DataFrame([result_row]), use_container_width=True, hide_index=True)

            if prediction >= 80:
                st.success("Excellent Recovery Expected ✅")

            elif prediction >= 60:
                st.warning("Moderate Recovery ⚠️")

            else:
                st.error("Poor Recovery Expected ❌")

        else:
            st.error("Prediction Failed")
            st.write(response.text)

    except Exception as e:
        st.error("Could not connect to FastAPI Server.")
        st.exception(e)


# ---------------------------------------------------
# Batch prediction (CSV upload)
# ---------------------------------------------------

st.markdown("---")
st.header("Batch prediction (CSV)")
st.write("Upload a CSV containing athlete features. The target label is optional and ignored.")

REQUIRED_COLUMNS = [
    "Age", "Gender", "Sport_Type", "Training_Type", "Training_Duration_Min",
    "Training_Intensity", "Sleep_Duration_Hours", "Caffeine_Intake_mg",
    "Stress_Level", "Resting_Heart_Rate", "HRV_ms", "Mood_Score",
    "Muscle_Soreness", "Energy_Level",
]

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
if uploaded_file is not None:
    batch_api_url = os.getenv("API_URL", "http://fastapi:8000")
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("Input preview")
        st.dataframe(input_df.head(10), use_container_width=True)
        if st.button("🚀 Predict batch", type="primary"):
            missing = [column for column in REQUIRED_COLUMNS if column not in input_df.columns]
            if missing:
                st.error(f"CSV is missing required columns: {', '.join(missing)}")
            elif input_df.empty:
                st.warning("The uploaded CSV contains no rows.")
            else:
                # Strip labels/identifiers; the API receives the data itself, never a file path.
                features_df = input_df.drop(columns=["Recovery_Score", "Athlete_ID"], errors="ignore")
                records = features_df.astype(object).where(
                    pd.notna(features_df), None).to_dict(orient="records")
                with st.spinner("Requesting predictions from FastAPI..."):
                    response = requests.post(
                        f"{batch_api_url}/predict",
                        json={"features": records, "source": "webapp"},
                        timeout=120,
                    )
                response.raise_for_status()
                body = response.json()
                results = body.get("results", [])
                if not results:
                    st.warning("The API returned no predictions.")
                else:
                    result_df = pd.DataFrame([{
                        **item.get("features", {}),
                        "Predicted_Recovery_Score": item.get("prediction"),
                        "model_version": item.get("model_version", body.get("model_version")),
                    } for item in results])
                    st.success(f"Predicted {len(result_df)} rows")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download predictions CSV", result_df.to_csv(index=False),
                        "batch_predictions.csv", "text/csv")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"The uploaded file is not a readable CSV: {exc}")
    except requests.RequestException as exc:
        st.error(f"The prediction API could not process this file: {exc}")
    except (ValueError, TypeError) as exc:
        st.error(f"Could not interpret the API response: {exc}")
