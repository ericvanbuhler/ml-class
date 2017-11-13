import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, SimpleRNN
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

with open('book.pkl', 'rb') as input:
    cached_data = pickle.load(input)
    char_to_int = cached_data['char_to_int']
    int_to_char = cached_data['int_to_char']
    X = cached_data['X']
    y = cached_data['y']
    print("Read cache file {}.".format(input.name))

start = np.random.randint(0, len(X)-1)
pattern = [int(x) for x in X[start] * len(char_to_int)]

model = load_model('book-lstm.h5')

print("Starting with: \"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for i in range(10000):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(int_to_char))
	prediction = model.predict(x, verbose=0)
	index = np.random.choice(range(len(prediction[0])), p = prediction[0]  / sum(prediction[0] ))
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	sys.stdout.flush()
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
