import streamlit as st
import joblib
import re
from nltk.corpus import wordnet # stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk

# Download necessary NLTK resource
# nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load saved components
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
encoder = joblib.load("label_encoder.pkl")
accuracy = joblib.load("model_accuracy.pkl") 

# Initializing NLP tools
lemmatizer = WordNetLemmatizer()
#stopwords_set = set(stopwords.words('english'))

# Preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tagged_tokens = pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }.get(tag[0].upper(), wordnet.NOUN))
        for word, tag in tagged_tokens
        # if word not in stopwords_set  # Optional stopword removal
    ]
    return " ".join(lemmatized)

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("Twitter Sentiment Analyzer")

st.markdown(f"**Model Accuracy:** `{accuracy * 100:.2f}%`") 

user_input = st.text_area("Tweet Text", placeholder=" ")

if st.button("Check Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        sentiment = encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")

        if sentiment.lower() == "positive":
            st.balloons()
