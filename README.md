# sentiment_analysis_twitter
A simple and effective sentiment analysis app for tweets, built using Streamlit, Logistic Regression, and TF-IDF vectorization. This project predicts whether a tweet expresses a positive, neutral, or negative sentiment.

Dataset Source: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

Model Pipeline:

1] Text Preprocessing

-> Remove URLs, mentions, and non-alphabetic characters

-> Lowercase conversion

-> Tokenization

-> POS tagging

-> Lemmatization using WordNetLemmatizer

-> (Stopword removal skipped for better sentiment retention)

2] Feature Extraction

-> TfidfVectorizer with ngram_range=(1,2) and max_features=5000

3] Model

-> Logistic Regression model used

-> Trained on 85% of the data and evaluated on 15%

4] Evaluation

-> Accuracy displayed on the Streamlit UI

-> Classification report printed during training

Achieved an overall accuracy of 70% while testing the data
