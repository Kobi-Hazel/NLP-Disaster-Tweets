# NLP-Disaster-Tweets

Welcome to my repository for the Kaggle competition [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started). Here, you'll find the source code and associated files for my submission to the competition, which is focused on using NLP techniques to build a model for accurately predicting whether a given tweet is referring to a real disaster or not.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)

## Overview

This project uses a Support Vector Machine (SVM) model for disaster-related tweet classification. The feature extraction pipeline incorporates sentiment analysis using VADER, keyword classification, average text vectors using GloVe pre-trained word vectors, and  location-based features using geocoding. After hyperparameter tuning through grid search and cross-validation, the final model achieves a 77.9% accuracy on the test dataset.

## Requirements

Ensure you have the following prerequisites installed:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Gensim
- spaCy
- NLTK
- tqdm
- ...

Additionally, download the GloVe Twitter pre-trained word vectors (glove.twitter.27B.200d.txt) from the official website [here](https://nlp.stanford.edu/projects/glove/) and save it in the project directory.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
   
```bash
git clone https://github.com/your-username/your-repo.git
```

2. Navigate to the project directory:
 
```bash
cd your-repo
```

3. Convert GloVe to Word2Vec format:
   
```bash
python -m gensim.scripts.glove2word2vec -i glove.twitter.27B.200d.txt -o glove.twitter.27B.200d.word2vec
```

4. Run the main script:
   
```bash
python main.py
```
   
