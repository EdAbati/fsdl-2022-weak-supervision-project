import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

LABEL_MAPPING = {
    "World": "World 🌎",
    "Sports": "Sports 🏆",
    "Business": "Business 💹",
    "Sci/Tech": "Sci/Tech 📡",
}


DEFAULT_LAMBDA_URL = (
    "http://lambda:8080/2015-03-31/functions/function/invocations"
)
LAMBDA_URL = os.environ.get("LAMBDA_URL", DEFAULT_LAMBDA_URL)


def request_to_lambda(
    request_text: str,
    lambda_url: str = LAMBDA_URL,
) -> pd.DataFrame:

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


def pretty_print_label(label: str) -> str:
    pretty_label = LABEL_MAPPING.get(label, "Undefined 😵‍💫")
    return pretty_label


def preprocess_preds_df(preds_df: pd.DataFrame) -> pd.DataFrame:
    preds_df["label"] = preds_df["label"].apply(pretty_print_label)
    preds_df["prob"] = preds_df["score"].apply(lambda x: f"{100 * x:.2f}%")
    return preds_df.sort_values(by="score", ascending=False)


def create_table(preds_df: pd.DataFrame):
    st.table(preds_df)


def create_chart(preds_df: pd.DataFrame):
    c = (
        alt.Chart(preds_df)
        .mark_bar()
        .encode(
            x=alt.X("label:O", sort="-y"),
            y="score:Q",
            color="score:Q",
            tooltip=["label", "score"],
        )
    )
    st.altair_chart(c, use_container_width=True)


def main():
    st.title("📰 News Classifier App")
    st.subheader("Classify a news article!")
    st.markdown("Write a news article headline in the prompt below.")
    st.markdown(
        "The app will classify it in one of 4 different categories: World, Sport, Business or Sci/Tech."
    )

    with st.form(key="news_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        with st.spinner("Getting predictions..."):
            pred_label: pd.DataFrame = request_to_lambda(request_text=raw_text)
            pred_string: str = pred_label.loc[
                pred_label["score"].idxmax(), "label"
            ]

        st.info("Prediction")
        st.markdown(pretty_print_label(pred_string))

        col1, col2 = st.columns(2)
        with col1:
            st.info("Original Text")
            st.markdown(raw_text)

        with col2:
            st.info("Prediction Probability")
            preds_df = pd.DataFrame(pred_label)
            preds_df = preprocess_preds_df(preds_df)

            create_table(preds_df)
            create_chart(preds_df)


if __name__ == "__main__":
    main()
