import keras

keras.__version__
from keras.datasets import imdb



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data[0]
train_labels[0]
max([max(sequence) for sequence in train_data])
# word_index is a dictionary mapping words to an integer index

word_index = imdb.get_word_index()

# We reverse it, mapping integer indices to words

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# We decode the review; note that our indices were offset by 3

# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
decoded_review
import numpy as np



def vectorize_sequences(sequences, dimension=10000):

    # Create an all-zero matrix of shape (len(sequences), dimension)

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.  # set specific indices of results[i] to 1s

    return results



# Our vectorized training data

x_train = vectorize_sequences(train_data)

# Our vectorized test data

x_test = vectorize_sequences(test_data)
x_train[0]
# Our vectorized labels

y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=["acc"])
from keras import optimizers



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics=["acc"])
from keras import losses

from keras import metrics



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss=losses.binary_crossentropy,

              metrics=[metrics.binary_accuracy])
x_val = x_train[:10000]

partial_x_train = x_train[10000:]



y_val = y_train[:10000]

partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=20,

                    batch_size=512,

                    validation_data=(x_val, y_val))
history_dict = history.history

history_dict.keys()
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=["acc"])



model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
results
model.predict(x_test)