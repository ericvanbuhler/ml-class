
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
	print("Read cache file %s." % input.name)

# define the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=10, batch_size=128)
model.save("book-lstm.h5")
