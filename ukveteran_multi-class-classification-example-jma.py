from keras.datasets import reuters



(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
len(train_data)
len(test_data)
train_data[10]
word_index = reuters.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Note that our indices were offset by 3

# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
decoded_newswire
train_labels[10]
import numpy as np



def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.

    return results



# Our vectorized training data

x_train = vectorize_sequences(train_data)

# Our vectorized test data

x_test = vectorize_sequences(test_data)
def to_one_hot(labels, dimension=46):

    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):

        results[i, label] = 1.

    return results



# Our vectorized training labels

one_hot_train_labels = to_one_hot(train_labels)

# Our vectorized test labels

one_hot_test_labels = to_one_hot(test_labels)
from keras.utils.np_utils import to_categorical



one_hot_train_labels = to_categorical(train_labels)

one_hot_test_labels = to_categorical(test_labels)
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
x_val = x_train[:1000]

partial_x_train = x_train[1000:]



y_val = one_hot_train_labels[:1000]

partial_y_train = one_hot_train_labels[1000:]
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=20,

                    batch_size=512,

                    validation_data=(x_val, y_val))
import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))



model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(partial_x_train,

          partial_y_train,

          epochs=8,

          batch_size=512,

          validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)
results