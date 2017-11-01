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
fixed_target = target[pd.notnull(text)]
fixed_text = text[pd.notnull(text)]

from sklearn.base import BaseEstimator, TransformerMixin

class NumBangExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe outputs average word length"""

    def __init__(self):
        pass

    def num_bang(self, str):
        """Helper code to compute number of exclamation points"""
        return str.count('!')

    def transform(self, inp, y=None):
        out = np.array([self.num_bang(x) for x in inp])
        return out.reshape(-1,1)


    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

# Perform feature extraction and train a model with this data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion

p = Pipeline(steps=[('feats', FeatureUnion([
                            ('numbang', NumBangExtractor()),
                            ('counts', CountVectorizer())
                            ])),
                ('multinomialnb', MultinomialNB())])

p.fit(fixed_text, fixed_target)
#print(p.predict(["I love my iphone!"]))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(p, fixed_text, fixed_target, cv=10)
print(scores)
print(scores.mean())