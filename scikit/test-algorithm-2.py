import pandas as pd
import numpy as np

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

p = Pipeline(steps=[('counts', CountVectorizer()),
                ('multinomialnb', MultinomialNB())])

# (Tweets 0 to 5999 are used for training data)
p.fit(fixed_text[0:6000], fixed_target[0:6000])

# See what the classifier predicts for some new tweets:
# (Tweets 6000 to 9091 are used for testing)
predictions = p.predict(fixed_text[6000:9092])
print(len(predictions))
correct_predictions = sum(predictions == fixed_target[6000:9092])
print('Percent correct: ', 100.0 * correct_predictions / 3092)
