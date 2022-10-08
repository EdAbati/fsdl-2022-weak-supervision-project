import json
import os

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from transformers import pipeline

LABEL_MAPPING = {
    "World": "World 🌎",
    "Sports": "Sports 🏆",
    "Business": "Business 💹",
    "Sci/Tech": "Sci/Tech 📡",
}

DEFAULT_LAMBDA_URL = (
    f"http://lambda:8080/2015-03-31/functions/function/invocations"
)
LAMBDA_URL = os.environ.get("LAMBDA_URL", DEFAULT_LAMBDA_URL)


def request_to_lambda(
    request_text: str, lambda_url: str = LAMBDA_URL
) -> pd.DataFrame:
    # curl -POST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"text": "A test sentence."}'
    data = json.dumps({"text": request_text})
    response = requests.post(lambda_url, data=data).json()
    body = response["body"]

    df_response = (
        pd.read_json(body)
        .reset_index()
        .rename(
            {
                "index": "label",
                "predicted_labels": "score",
            },
            axis=1,
        )
    )
    return df_response


def pretty_print_label(label: str):
    pretty_label = LABEL_MAPPING.get(label, "Undefined 😵‍💫")
    return pretty_label


def get_classifier(model_id):
    return pipeline("text-classification", model=model_id)


def classify_news(classifier, text):
    preds = classifier(text, return_all_scores=False)
    label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    text_label = label.get(int(preds[0]["label"][-1]))
    preds = classifier(text, return_all_scores=True)
    for items in preds[0]:
        items["label"] = label.get(int(items["label"][-1]))

    return text_label, preds[0]


def preprocess_preds_df(preds_df: pd.DataFrame) -> pd.DataFrame:
    preds_df["label"] = preds_df["label"].apply(pretty_print_label)
    preds_df["prob"] = preds_df["score"].apply(lambda x: f"{100 * x:.2f}%")
    return preds_df


def create_table(preds_df: pd.DataFrame):
    st.write(preds_df)


def create_chart(raw_text: str, preds_df: pd.DataFrame):

    c = (
        alt.Chart(preds_df)
        .mark_bar()
        .encode(
            x="label:O",
            y="score:Q",
            color="score:Q",
            tooltip=["label", "score"],
        )
    )

    st.altair_chart(c, use_container_width=True)


def main():
    st.title("News Classifier App")
    menu = ["Home", "Saved Logs", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    raw_text = ""

    # model_id = "KushalRamaiya/distilbert-base-uncased-finetuned-news"
    # classifier = get_classifier(model_id)

    if choice == "Home":
        st.subheader("Classify News from Headlines")
        with st.form(key="news_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1, col2 = st.columns(2)

            # pred_label, pred_prob = classify_news(classifier, raw_text)
            pred_label: pd.DataFrame = request_to_lambda(request_text=raw_text)
            max_label: str = pred_label.max().to_dict()["label"]

            with col1:
                st.info("Original Text")
                st.write(raw_text)
                st.info("Prediction")
                st.write(pretty_print_label(max_label))

            with col2:
                st.info("Prediction Probability")
                preds_df = pd.DataFrame(pred_label)
                preds_df = preprocess_preds_df(preds_df)

                create_table(preds_df)
                create_chart(raw_text, preds_df)

    if choice == "Saved Logs":
        st.subheader("Logs")
        # data = pd.DataFrame(columns=["Text","Probability"])
        # st.text("Logs")
        # df_log_process = st.dataframe(data)

        # if submit_text:
        #     data = data.append(
        #         {
        #             'Text':raw_text,
        #             'Probability':"something",
        #         }, ignore_index=True)

        #     df_log_process = df_log_process.dataframe(data)

    if choice == "About":
        st.subheader("About")
    pass


if __name__ == "__main__":
    main()
