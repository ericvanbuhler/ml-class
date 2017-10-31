import pandas as pd
import numpy as np
from sklearn.externals import joblib

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
fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

# Perform feature extraction and train a model with this data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

p = Pipeline(steps=[('counts', CountVectorizer(ngram_range=(1, 2))),
                ('multinomialnb', MultinomialNB())])

p.fit(fixed_text, fixed_target)

# Save the model
joblib.dump(p, 'sentiment-model.pkl')
