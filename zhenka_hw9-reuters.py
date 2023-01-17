from tensorflow.keras.datasets import reuters

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.layers import Dense, Dropout, Embedding, Activation, LSTM, GRU, SpatialDropout1D, Flatten

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline 
cb_EarlyStopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)

cb_ModelCheckpoint = ModelCheckpoint(filepath='model.best.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

cb_ReduceLROnPlateau = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1, mode='max', min_lr=0.00000001)
num_words=10000
data = np.load('../input/train.npz', allow_pickle=True) 

x_train_src = data['x'] 

y_train_src = data['y'] 
x_test_src = np.load('../input/test.npy', allow_pickle=True)
index = 0

print(x_train_src[index])

print(y_train_src[index])
word_index = reuters.get_word_index()
maxlen = 800

x_train = pad_sequences(x_train_src, maxlen=maxlen)

x_test = pad_sequences(x_test_src, maxlen=maxlen)
!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove.6B.zip
!ls -la
embeddings_index = dict()

with open('glove.6B.300d.txt', 'r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        word_vector = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = word_vector
embedding_dim = 300

embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():

    if i < num_words - 3:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i + 3] = embedding_vector

            
y_train = utils.to_categorical(y_train_src)
num_classes=len(y_train[0])

print(f'Колличество классов: {num_classes}')
model = Sequential()

model.add(Embedding(num_words, embedding_dim, input_length=maxlen))

model.add(LSTM(512))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))
model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = False
model.summary()
model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
history = model.fit(x_train, 

                    y_train, 

                    epochs=50,

                    batch_size=128,

                    callbacks = [cb_EarlyStopping, cb_ModelCheckpoint, cb_ReduceLROnPlateau],

                    validation_split=0.1,

                    verbose=1)
plt.plot(history.history['acc'], 

         label='Доля верных ответов на обучающем наборе')

plt.plot(history.history['val_acc'], 

         label='Доля верных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля верных ответов')

plt.legend()

plt.show()
predictions = model.predict(x_test)
predictions[:2]
predictions = np.argmax(predictions, axis=1)
predictions[:5]
out = np.column_stack((range(1, predictions.shape[0]+1), predictions))
out[:5]
np.savetxt('submission.csv', out, header="Id,Category", 

            comments="", fmt="%d,%d")
!cat submission.csv