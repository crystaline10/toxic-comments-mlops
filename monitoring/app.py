from decimal import Decimal

import boto3
import pandas as pd
import streamlit as st


AWS_REGION = "us-east-2"
TABLE_NAME = "toxic-comment-predictions"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


@st.cache_resource
def get_table():
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=AWS_REGION,
    )
    return dynamodb.Table(TABLE_NAME)


def load_prediction_data():
    table = get_table()
    response = table.scan()

    items = response.get("Items", [])

    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response.get("Items", []))

    return items


def prepare_dataframe(items):
    rows = []

    for item in items:
        prediction = item.get("prediction", {})

        row = {
            "request_id": item.get("request_id"),
            "timestamp": item.get("timestamp"),
            "text": item.get("text"),
            "latency_ms": float(
                item.get("latency_ms", Decimal("0"))
            ),
            "feedback_correct": item.get("feedback_correct"),
        }

        for label in LABEL_COLUMNS:
            row[label] = int(prediction.get(label, 0))

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )
        df = df.sort_values("timestamp")

    return df


st.set_page_config(
    page_title="Toxic Comment Model Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("Toxic Comment Model Monitoring")

items = load_prediction_data()
df = prepare_dataframe(items)

if df.empty:
    st.warning("No prediction data is available.")
    st.stop()

# st.metric(
#     "Total Predictions",
#     len(df),
# )

feedback_df = df[df["feedback_correct"].notna()]

total_feedback = len(feedback_df)

if total_feedback > 0:
    correct_predictions = feedback_df["feedback_correct"].sum()
    live_accuracy = correct_predictions / total_feedback

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", len(df))
    col2.metric("Feedback Responses", total_feedback)
    col3.metric("Live Accuracy", f"{live_accuracy:.1%}")
else:
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", len(df))
    col2.metric("Feedback Responses", 0)
    col3.metric("Live Accuracy", "N/A")

st.subheader("Prediction Latency Over Time")

latency_chart = df.set_index("timestamp")[["latency_ms"]]

st.line_chart(latency_chart)

st.subheader("Predicted Class Distribution")

class_counts = df[LABEL_COLUMNS].sum()

st.bar_chart(class_counts)