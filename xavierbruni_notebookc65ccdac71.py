from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Dropout
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train[0]
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_words, padding='post')
X_train[0]
model = Sequential()
model.add(Embedding(10000,32, input_length=500))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=200, batch_size=128, verbose=2)
model = Sequential()
model.add(Embedding(10000,32, input_length=500))
model.add(MaxPool1D())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model = Sequential()
model.add(Embedding(10000,32, input_length=500))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dropout(rate=0.1))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=5, batch_size=128, verbose=2)
