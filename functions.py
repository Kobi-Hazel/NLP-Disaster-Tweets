# Import libraries
from spellchecker import SpellChecker
import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Function to clean text
def clean(text, remove_hashtags=True, correct_spelling=True):

    """
    Clean and preprocess a given text.

    Parameters:
    - text (str): The input text to be cleaned.
    - remove_hashtags (bool): If True, remove hashtags from the text. Default is True.
    - correct_spelling (bool): If True, attempt to correct spelling using a SpellChecker. Default is True.

    Returns:
    - str: The cleaned and preprocessed text.
    """

    # Create a SpellChecker instance for correcting spelling
    spell = SpellChecker()

    # Remove URLs from the text
    text_no_urls = re.sub(r'https?://\S+|www\.\S+', "", text.lower())

    # Remove hashtags if specified
    if remove_hashtags:
        text_no_hashtags = re.sub(r"#\w+", "", text_no_urls)
    else:
        text_no_hashtags = text_no_urls

    # Remove Twitter handles
    text_no_handles = re.sub(r"@\w+", "", text_no_hashtags)

    # Remove special characters (excluding alphanumeric characters and spaces)
    text_no_special_chars = re.sub(r'[^\w\s]', "", text_no_handles)

    # Correct spelling if specified
    if correct_spelling:
        # Tokenize the cleaned text, correct spelling, and join the corrected words back into a sentence
        clean_text = " ".join(spell.correction(word) or word for word in text_no_special_chars.split())
    else:
        # If spelling correction is not specified, join the cleaned words back into a sentence
        clean_text = " ".join(word for word in text_no_special_chars.split())

    return clean_text


# Function to get keyword based on text and a trained classifier
def get_keyword(keyword, text, clf, threshold=0.4):

    """
    Get a keyword prediction for a given text using a trained classifier.

    Parameters:
    - keyword (str): The initial keyword. If provided, this keyword is returned without further prediction.
    - text (str): The input text for which to predict a keyword.
    - clf: The trained classifier with a tf-idf vectorizer, classification model, and label encoder.
    - threshold (float): The confidence threshold for accepting a keyword prediction. Default is 0.4.

    Returns:
    - str: The predicted keyword for the input text, or an empty string if the confidence is below the threshold
           or if an initial keyword is provided.
    """

    # Check if an initial keyword is provided
    if keyword == "":
        # Transform the input text using the classifier's tf-idf vectorizer
        tfidf_matrix = clf.tfidf_vectorizer.transform([text])

        # Predict the encoded keyword and calculate the prediction probability
        keyword_prediction_encoded = clf.model.predict(tfidf_matrix)
        prediction_probability = np.max(clf.model.predict_proba(tfidf_matrix), axis=1)

        # Check if the prediction probability is above the specified threshold
        if prediction_probability >= threshold:
            # Decode the predicted keyword using the label encoder and return it
            return clf.le.inverse_transform(keyword_prediction_encoded)[0]
        else:
            # Return an empty string if the confidence is below the threshold
            return ""
    else:
        # Return the initial keyword if provided
        return keyword


# Function to get location based on text and named entity recognition
def get_location(location, text, nlp):

    """
    Get a location based on the provided location or by performing named entity recognition (NER) on the input text.

    Parameters:
    - location (str): The initial location. If provided, this location is returned without further processing.
    - text (str): The input text on which to perform NER for extracting geographical entities.
    - nlp: The spaCy NLP model used for named entity recognition.

    Returns:
    - str: The extracted location from the provided text using NER or the initial location if provided.
    """

    # Clean the provided location if it is not provided
    clean_location = clean(location, remove_hashtags=False, correct_spelling=False)

    # Check if a cleaned location is not provided
    if clean_location == "":
        # Process the input text using spaCy NLP for named entity recognition (NER)
        doc = nlp(text)

        # Extract geographical entities (GPE) from the recognized entities in the text
        gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

        # Join the extracted geographical entities into a single string and return
        return " ".join(gpe_entities)
    else:
        # Return the cleaned location if it is provided
        return clean_location


# Function to get country based on location and geocoding
def get_country(location, geocode):

    """
    Get the country from a given location using a geocoding function.

    Parameters:
    - location (str): The location for which to obtain the country.
    - geocode (function): A geocoding function that takes a location and returns geolocation information.

    Returns:
    - str: The country obtained from the geolocation.
           Returns an empty string if geolocation is not available or an error occurs during the process.
    """

    try:
        # Attempt to obtain geolocation information using the provided geocoding function
        geolocation = geocode(location)

        # Check if geolocation information is available
        if geolocation:
            # Extract the country from the address and convert it to lowercase
            country = geolocation.address.split(",")[-1].strip().lower()
            return country
        else:
            # Return an empty string if geolocation is not available
            return ""
    except Exception as e:
        # Handle other unexpected exceptions
        print(f"Error occurred: {e}")
        return ""


# Function to convert text to an average vector using Word2Vec
def text_to_avg_vector(text, word_vectors):

    """
    Convert a given text into an average vector representation based on pre-trained word vectors.

    Parameters:
    - text (str): The input text to be converted.
    - word_vectors: Pre-trained word vectors, typically obtained from a word embedding model.

    Returns:
    - numpy.ndarray: An average vector representation of the input text based on word vectors.
                    Returns a zero vector if none of the words in the text are present in the word vectors.
    """

    # Split the input text into individual words
    words = text.split()

    # Extract word vectors for words that exist in the pre-trained word vectors
    vectors = [word_vectors[word] for word in words if word in word_vectors]

    # Check if any valid word vectors were obtained
    if vectors:
        # Calculate the mean (average) vector along the specified axis (axis=0 means averaging along columns)
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if no valid word vectors were found
        return np.zeros(word_vectors.vector_size)


# Function using VADER to get sentiment score based on text
def get_sentiment(text):

    """
    Get sentiment polarity based on the compound score from the Sentiment Intensity Analyzer (SIA).

    Parameters:
    - text (str): The input text for sentiment analysis.

    Returns:
    - int: The sentiment label, where 1 indicates positive sentiment, -1 indicates negative sentiment,
           and 0 indicates neutral sentiment.
    """

    # Create an instance of the Sentiment Intensity Analyzer (SIA)
    sid = SentimentIntensityAnalyzer()

    # Get the compound score from the SIA for the input text
    compound_score = sid.polarity_scores(text)["compound"]

    # Determine the sentiment label based on the compound score
    if compound_score > 0.333:
        return 1  # Positive sentiment
    elif compound_score < -0.333:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment


# Function to tokenize and lemmatize text for TF-IDF vectorization
def tokenizer(text):

    """
    Tokenize and lemmatize the input text, removing English stopwords.

    Parameters:
    - text (str): The input text to be tokenized and lemmatized.

    Returns:
    - list: A list of lemmatized tokens without English stopwords.
    """

    # Tokenize the input text into words
    tokens = word_tokenize(text.lower())

    # Create an instance of the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize tokens, removing stopwords in the process
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words("english")]

    # Return the list of lemmatized tokens
    return tokens


# Function to perform TF-IDF vectorization
def tfidf_vectorize(text):

    """
    Vectorize the input text using the TF-IDF (Term Frequency-Inverse Document Frequency) method.

    Parameters:
    - text (list): A list of strings representing the documents for vectorization.

    Returns:
    - tuple: A tuple containing the TF-IDF matrix (vector) and the fitted TfidfVectorizer.
    """

    # Create an instance of the TfidfVectorizer with a custom tokenizer
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)

    # Fit and transform the input text using the TfidfVectorizer
    vector = vectorizer.fit_transform(text)

    # Return a tuple containing the TF-IDF matrix (vector) and the fitted TfidfVectorizer
    return vector, vectorizer


# Class for Keyword Classifier
class KeywordClassifier:

    """
    A class for training a keyword classifier using a RandomForest model.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing columns "text_preprocess" and "keyword."

    Attributes:
    - keyword_index (pandas.Series): A boolean series indicating the rows where the "keyword" column is not empty.
    - keyword_text (pandas.Series): The preprocessed text corresponding to non-empty "keyword" rows.
    - keyword_targets (pandas.Series): The target keywords corresponding to non-empty "keyword" rows.
    - le (LabelEncoder): A label encoder for encoding target keywords.
    - keyword_targets_encoded (numpy.ndarray): Encoded target keywords.
    - keyword_tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix for keyword text.
    - tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for vectorizing keyword text.
    - model (RandomForestClassifier): The trained RandomForest classifier.
    """

    def __init__(self, df):
        # Filter rows where "keyword" is not empty
        self.keyword_index = df["keyword"] != ""

        # Extract relevant columns for training
        self.keyword_text = df["text_preprocess"][self.keyword_index]
        self.keyword_targets = df["keyword"][self.keyword_index]

        # Initialize LabelEncoder and encode target keywords
        self.le = LabelEncoder()
        self.keyword_targets_encoded = self.le.fit_transform(self.keyword_targets)

        # Vectorize keyword text using TF-IDF
        self.keyword_tfidf_matrix, self.tfidf_vectorizer = tfidf_vectorize(self.keyword_text)

        # Initialize and train the RandomForest model
        self.model = RandomForestClassifier()
        self.model.fit(self.keyword_tfidf_matrix, self.keyword_targets_encoded)
