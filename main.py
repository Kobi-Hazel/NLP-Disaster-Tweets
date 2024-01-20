# Import libraries
import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from functools import partial
import spacy
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# Import custom functions module
import functions

# Load training and test datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Handle missing values in 'keyword' and 'location'
train_df["keyword"] = train_df["keyword"].fillna("")
train_df["location"] = train_df["location"].fillna("")

test_df["keyword"] = test_df["keyword"].fillna("")
test_df["location"] = test_df["location"].fillna("")

# Extract target values from the training dataset
targets = train_df["target"]

# Load spaCy model and geopy geolocator
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="tweet_locator")
geocode = partial(geolocator.geocode, language="en")

# Convert GloVe to Word2Vec format if not already done
if not os.path.exists("glove.twitter.27B.200d.word2vec"):
    glove2word2vec("glove.twitter.27B.200d.txt", "glove.twitter.27B.200d.word2vec")

# Load Word2Vec model
word_vectors = KeyedVectors.load_word2vec_format('glove.twitter.27B.200d.word2vec', binary=False)

# Apply data cleaning and preprocessing to text columns
# text_updated - remove hashtags and spell check - use for computing average text vectors
# text_preprocess - keep hashtags and do not correct spell check - use to fill in keyword and location blanks
tqdm.pandas(desc="Processing")
train_df["text_updated"] = train_df["text"].progress_apply(lambda x: functions.clean(x))
train_df["text_preprocess"] = \
    train_df["text"].progress_apply(lambda x: functions.clean(x))
test_df["text_updated"] = \
    test_df["text"].progress_apply(lambda x: functions.clean(x))
test_df["text_preprocess"] = \
    test_df["text"].progress_apply(lambda x: functions.clean(x, remove_hashtags=False, correct_spelling=False))

# Initialise and apply keyword classifier to fill in keyword blanks
keyword_clf = functions.KeywordClassifier(train_df)
train_df["keyword_updated"] = \
    train_df.progress_apply(lambda x: functions.get_keyword(x["keyword"], x["text_preprocess"], keyword_clf), axis=1)
test_df["keyword_updated"] = \
    test_df.progress_apply(lambda x: functions.get_keyword(x["keyword"], x["text_preprocess"], keyword_clf), axis=1)

# Extract location information from text to fill in location blanks
train_df["location_updated"] = \
    train_df.progress_apply(lambda x: functions.get_location(x["location"], x["text_preprocess"], nlp), axis=1)
test_df["location_updated"] = \
    test_df.progress_apply(lambda x: functions.get_location(x["location"], x["text_preprocess"], nlp), axis=1)

# Get country information based on location
train_df["country"] = train_df["location_updated"].progress_apply(lambda x: functions.get_country(x, geocode))
test_df["country"] = test_df["location_updated"].progress_apply(lambda x: functions.get_country(x, geocode))

# Extract sentiment from text
train_df["sentiment"] = train_df["text_updated"].progress_apply(lambda x: functions.get_sentiment(x))
test_df["sentiment"] = test_df["text_updated"].progress_apply(lambda x: functions.get_sentiment(x))

# Convert text to average vectors using Word2Vec
train_text_vectors = [functions.text_to_avg_vector(text, word_vectors) for text in train_df["text_updated"]]
train_text_matrix = np.vstack(train_text_vectors)
test_text_vectors = [functions.text_to_avg_vector(text, word_vectors) for text in test_df["text_updated"]]
test_text_matrix = np.vstack(test_text_vectors)

# One-hot encode categorical features
train_unique_countries = train_df["country"].unique().reshape(-1, 1)
test_unique_countries = test_df["country"].unique().reshape(-1, 1)
unique_countries = np.vstack((train_unique_countries, test_unique_countries))

ohe_country = OneHotEncoder().fit(unique_countries)
ohe_keyword = OneHotEncoder()

train_country_encoded = ohe_country.transform(train_df["country"].to_numpy().reshape(-1, 1)).toarray()
train_keyword_encoded = ohe_keyword.fit_transform(train_df["keyword_updated"].to_numpy().reshape(-1, 1)).toarray()

test_country_encoded = ohe_country.transform(test_df["country"].to_numpy().reshape(-1, 1)).toarray()
test_keyword_encoded = ohe_keyword.transform(test_df["keyword_updated"].to_numpy().reshape(-1, 1)).toarray()

# Create an array the sentiment scores and reshape
train_sentiment = np.array(train_df["sentiment"]).reshape(-1, 1)
test_sentiment = np.array(test_df["sentiment"]).reshape(-1, 1)

# Combine features for training and testing datasets
train_features = np.hstack((train_country_encoded, train_keyword_encoded, train_text_matrix, train_sentiment))
test_features = np.hstack((test_country_encoded, test_keyword_encoded, test_text_matrix, test_sentiment))

# Define hyperparameter grid for SVM
param_grid = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1]
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5, scoring="accuracy", refit=True, verbose=3)
grid_search.fit(train_features, targets)

# Display best estimator and score
print(f"Best Estimator:  {grid_search.best_estimator_}")
print(f"Best Score: {grid_search.best_score_}")

# Make predictions on the test dataset
predictions = grid_search.predict(test_features)

# Create a submission DataFrame and save it to a CSV file
submission_df = pd.DataFrame({"id": test_df["id"], "target": predictions})
output = submission_df.to_csv("submission_file.csv", index=False)
