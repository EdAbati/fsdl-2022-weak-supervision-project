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


DEFAULT_LAMBDA_URL = "http://lambda:8080/2015-03-31/functions/function/invocations"
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


def load_example():

    examples = {
        "science": """As a society, we trust innovation will resolve the world's environmental crisis: Science and Technology will find the answer, just as they have many times in the past. Yet according to the Global Sustainable Investment Review, only a very small percentage of the $35T in ESG investments per year is allocated to accelerating high-impact, disruptive businesses, the so-called “Impact Unicorns.” The ones that can transform entire industries to become carbon-free or restore natural ecosystems. Instead, most of the investments go into expanding the application of existing technologies to public infrastructure or large corporations.""",
        "business": """Reacting to a mixed bag of big bank earnings reports and futures sliding down slightly, Odeon Capital Group chief financial strategist Dick Bove argued people are paying attention to the wrong numbers on "Mornings with Maria" Friday, claiming loan loss provisions and common equity are "not good" and "getting slammed. MARKET EXPERTS REACT TO HIGHER THAN EXPECTED C.P.I. REPORT: ‘THE FED IS LOSING ITS INFLATION FIGH DICK BOVE: It doesn't [get better] from the perspective of the banks. And again, the loan loss provision is an estimate that the bank makes of what it expects to see in the economy over the next few quarters, possibly even the next year or so. And when the loan loss provision goes from a massive, if you will, return of money to a massive cut in money, then you know that the banks believe that things are exactly what Jamie Dimon said in the last few comments that he's made, that they don't see good things happening.""",
        "sports": """Broncos guard Dalton Risner was asked about that on Tuesday. Risner was quick to praise his quarterback and he then brushed off the notion of worrying about an injury slowing Wilson down. “You know, Russell is a warrior,” Risner said. “For me as an offensive lineman, I love blocking for that guy, and I love going to war for him. I'll continue to go to war for him each and every day, each and every game. It's no question now that you guys know that he's been battling an injury. The cool thing about Russell is that it wasn't something that he let everyone know. It wasn't something that he wasn't complaining about during the week. He wanted to go to war and help us get a win.""",
        "world": """ Tech billionaire Elon Musk warned Friday that he cannot “indefinitely” continue to fund the Starlink satellite terminals he has provided to Ukraine to help the country stay online during the war, calling the financial burden “unreasonable.” He also implicitly linked the decision to a Ukrainian official’s rude remark after he publicly offered an unsolicited peace plan that involved conceding Ukrainian territory to Russia.""",
    }
    
    return examples


def main():
    st.title("📰 News Classifier App")
    st.subheader("Classify a news article!")
    st.markdown("Write a news article headline in the prompt below.")
    st.markdown("The app will classify it in one of 4 different categories: World, Sport, Business or Sci/Tech.")

    example_data = st.selectbox("Select an example", list(load_example().keys()), key="example")

    with st.form(key="news_clf_form"):
        raw_text = st.text_area("Type Here", value=load_example()[example_data])
        submit_text = st.form_submit_button(label="Submit")


    if submit_text:
        with st.spinner("Getting predictions..."):
            pred_label: pd.DataFrame = request_to_lambda(request_text=raw_text)
            pred_string: str = pred_label.loc[pred_label["score"].idxmax(), "label"]

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
