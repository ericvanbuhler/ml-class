import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.text import Tokenizer
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

config.max_words = 2000
config.batch_size = 32
config.epochs = 10
config.filters = 10
config.kernel_size = 3
config.embedding_dims = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=config.max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

# Vectorizing sequence data...
tokenizer = Tokenizer(num_words=config.max_words)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.sequences_to_matrix(x_train, mode='tfidf')
x_test = tokenizer.sequences_to_matrix(x_test, mode='tfidf')

# One hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = Sequential()
model.add(Embedding(config.max_words,
                    config.embedding_dims,
                    input_length=config.max_words))
model.add(Dropout(0.5))
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.6))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    validation_data=(x_test, y_test), callbacks=[WandbCallback()])

