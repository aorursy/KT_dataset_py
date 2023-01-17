import pickle

import tensorflow as tf

import matplotlib.pyplot as plt



from os import path

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import RMSprop

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense
filepath = "../input/keras-data.pickle"



data = pickle.load(open(path.abspath(filepath), "rb"))

X_train, y_train, X_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]

vocab_size, max_length = data["vocab_size"], data["max_length"]



# Dataset to huge for 6 hours of compute, so had to partition the  data

partition = round(len(X_train)/2)



# Prep and pad the training and test data

X_train = pad_sequences(sequences=X_train[:partition], maxlen=max_length)

X_test = pad_sequences(sequences=X_test[:partition], maxlen=max_length)

y_train = to_categorical(y_train[:partition], num_classes=2)

y_test = to_categorical(y_test[:partition], num_classes=2)

# initialize the Sequential model

model = Sequential()



# add layers to the model

model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length))

model.add(LSTM(256))

model.add(Dense(2, activation="sigmoid"))



# compile the model

# chose RMSProp as the optimizer, since it is usually a good choice for recurrent neural networks.

model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])



# output model

model.summary()
# run on Tesla P100-PCIE-16GB

with tf.device('/GPU:0'):

    # train the model

    history = model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=1)



# save the model

model.save("LSTM_model.h5")

# Should acquire at least 90 % accuracy from the LSTM

loss, acc = model.evaluate(X_test, y_test) 



# Plot the loss & accuracy in training

plt.plot(model.history.history['acc'], 'blue')

plt.plot(model.history.history['loss'], 'orange')

plt.title('Loss and Accuracy')

plt.legend(['accuracy',  'loss'], loc='upper right')

plt.xlabel('Epochs')

plt.ylabel('%')

plt.show()



print(f"Loss:\t{loss}\nAccuracy:\t{acc}")
