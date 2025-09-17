# sentiment_analysis_twitter
A simple and effective sentiment analysis app for tweets, built using Streamlit, Logistic Regression, and TF-IDF vectorization. This project predicts whether a tweet expresses a positive, neutral, or negative sentiment.

Dataset Source: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

Model Pipeline

1] Text Preprocessing

Remove URLs and mentions (http://, @username)

Keep hashtags (as they may carry sentiment value)

Remove all other non-alphabetic characters

Convert text to lowercase

Tokenize into words

Part-of-Speech (POS) tagging using NLTK

Lemmatization using WordNetLemmatizer with POS mapping

Stopword removal skipped (to preserve sentiment-bearing words like not, never)

2] Feature Extraction

TF-IDF Vectorizer used

Parameters:

ngram_range=(1, 2) (captures both unigrams and bigrams)

max_features=5000

min_df=2 (ignore rare words)

3] Model

Logistic Regression with:

max_iter=500 (ensures convergence)

C=2.0 (controls regularization strength)

solver="lbfgs" (efficient for multiclass classification)

Dataset split: 85% training, 15% testing (stratified)

4] Evaluation

During training:

Printed accuracy

Displayed classification report (precision, recall, F1-score)

Plotted confusion matrix

On Streamlit UI:

Displayed overall model accuracy

Predicted sentiment (Positive, Neutral, Negative)

Displayed probabilities for each sentiment (text + bar chart)

5] Achieved Performance

Overall accuracy ~70% on test data
