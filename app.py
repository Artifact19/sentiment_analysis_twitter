import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt


# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


# Load saved model & preprocessing tools
model = joblib.load("sentiment_model_logreg.pkl")
vectorizer = joblib.load("tfidf.pkl")
encoder = joblib.load("label_encoder.pkl")
accuracy = joblib.load("logreg_accuracy.pkl")


# Preprocessing 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Removing links & mentions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    # Keeping hashtags, removing other symbols
    text = re.sub(r"[^a-zA-Z#]", " ", text)
    text = text.lower()

    tokens = text.split()
    tagged_tokens = pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(
            word,
            {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
            }.get(tag[0].upper(), wordnet.NOUN)
        )
        for word, tag in tagged_tokens if len(word) > 1
    ]
    return " ".join(lemmatized)


# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("Twitter Sentiment Analyzer")

st.markdown(f"**Model Accuracy:** `{accuracy * 100:.2f}%`")

user_input = st.text_area("Enter a Tweet:", placeholder="Type something here...")

if st.button("Check Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vectorized)
        sentiment = encoder.inverse_transform(prediction)[0]

        # Probabilities
        proba = model.predict_proba(vectorized)[0]
        labels = encoder.classes_

        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")

        # Showing probabilities of each sentiment
        st.subheader("Prediction Probabilities")
        for lbl, p in zip(labels, proba):
            st.write(f"**{lbl.capitalize()}**: {p * 100:.2f}%")

        # Bar chart
        st.bar_chart(dict(zip(labels, proba)))
