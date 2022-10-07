import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline

model_id = "KushalRamaiya/distilbert-base-uncased-finetuned-news"
classifer = pipeline("text-classification", model=model_id)


def classify_news(text):
    preds = classifer(text, return_all_scores=False)
    label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    text_label = label.get(int(preds[0]["label"][-1]))
    preds = classifer(text, return_all_scores=True)
    for items in preds[0]:
        items["label"] = label.get(int(items["label"][-1]))

    return text_label, preds[0]


def main():
    st.title("News Classifer App")
    menu = ["Home", "Saved Logs", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    raw_text = ""

    if choice == "Home":
        st.subheader("Classify News from Headlines")
        with st.form(key="news_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1, col2 = st.columns(2)

            pred_label, pred_prob = classify_news(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(pred_label)

            with col2:
                st.success("Prediction Probability")
                preds_df = pd.DataFrame(pred_prob)
                st.write(preds_df)
                labels = ["World", "Sports", "Business", "Sci/Tech"]
                plt.bar(labels, 100 * preds_df["score"], color="C0")
                plt.title(f'"{raw_text}"')
                plt.ylabel("Class Probability (%)")
                st.pyplot(fig=plt)

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
