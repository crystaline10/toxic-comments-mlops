import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

if "result" not in st.session_state:
    st.session_state.result = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

st.set_page_config(
    page_title="Toxic Comment Moderation",
    page_icon="💬",
)

st.title("Toxic Comment Moderation")

st.write(
    "Enter a comment below to classify it across six toxicity categories."
)

comment = st.text_area(
    "Comment",
    placeholder="Enter a comment to analyze...",
    height=150,
)

if st.button("Analyze Comment"):
    if not comment.strip():
        st.warning("Please enter a comment.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": comment},
                timeout=30,
            )

            response.raise_for_status()

            #result = response.json()
            st.session_state.result = response.json()
            st.session_state.feedback_submitted = False

        except requests.RequestException as error:
            st.error(f"Unable to connect to the prediction API: {error}")

if st.session_state.result is not None:
    result = st.session_state.result

    st.subheader("Prediction Results")

    predictions = result["prediction"]

    for label, value in predictions.items():
        if value == 1:
            st.write(f"🔴 {label}: Toxic")
        else:
            st.write(f"🟢 {label}: Not detected")

    st.caption(
        f"Prediction latency: {result['latency_ms']} ms"
    )

    st.subheader("Feedback")
    st.write("Was this prediction correct?")

    if not st.session_state.feedback_submitted:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Yes"):
                feedback_response = requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "request_id": result["request_id"],
                        "is_correct": True,
                    },
                    timeout=30,
                )

                if feedback_response.ok:
                    st.session_state.feedback_submitted = True
                    st.rerun()

        with col2:
            if st.button("No"):
                feedback_response = requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "request_id": result["request_id"],
                        "is_correct": False,
                    },
                    timeout=30,
                )

                if feedback_response.ok:
                    st.session_state.feedback_submitted = True
                    st.rerun()
    else: 
        st.success("Thank you for your feedback.")