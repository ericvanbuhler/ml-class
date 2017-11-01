import pandas as pd
import numpy as np
import re, string

# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# The rows of  the "emotion" column have one of four strings:
# 'Positive emotion'
# 'Negative emotion'
# 'No emotion toward brand or product'
# 'I can't tell'

# Remove the blank rows from the series:
fixed_target = target[pd.notnull(text)]
fixed_text = text[pd.notnull(text)]

def tokenize_0(text):
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    return re.compile(r'\W+').split(text) # split at non words

def tokenize_1(text):
    return text.split()

# Perform feature extraction and train a model with this data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

p = Pipeline(steps=[('counts', CountVectorizer(tokenizer=tokenize_0)),
                ('multinomialnb', MultinomialNB())])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(p, fixed_text, fixed_target, cv=10)
print(scores)
print(scores.mean())