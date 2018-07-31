import imdb
import numpy as np
from keras.preprocessing import text
import wandb
from wandb.keras import WandbCallback
from sklearn.linear_model import LogisticRegression

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.layers import LSTM

wandb.init()
config = wandb.config
config.vocab_size = 1000
config.batch_size = 50
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()


tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

bow_model = LogisticRegression()
bow_model.fit(X_train, y_train)

pred_train = bow_model.predict(X_train)
acc = np.sum(pred_train==y_train)/len(pred_train)

pred_test = bow_model.predict(X_test)
val_acc = np.sum(pred_test==y_test)/len(pred_test)
wandb.log({"val_acc": val_acc, "acc": acc})

#bow_model = Sequential()
#bow_model.add(Dense(25000, input_shape=(config.vocab_size,)))
#bow_model.add(Dense(1, activation='sigmoid'))

#bow_model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

#bow_model.fit(X_train, y_train,
#          batch_size=config.batch_size,
#          epochs=config.epochs,
#          validation_data=(X_test, y_test), callbacks=[WandbCallback()])
