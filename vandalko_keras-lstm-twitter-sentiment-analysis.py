import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
 
# for reproducibility
from numpy.random import seed
from tensorflow import set_random_seed
random_seed = 1
seed(random_seed)
random_seed += 1
set_random_seed(random_seed)
random_seed += 1

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, SpatialDropout1D
from keras.layers import LSTM, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
print('reading CSV')

csv = pd.read_csv('../input/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header=None)
print('parsing CSV')

X, Y = [], []

for index, row in csv.iterrows():
    X.append(row[5])
    y_part = row[0]
    if y_part == 0:
        yy = np.array([0])
    elif y_part == 4:
        yy = np.array([1])
    else:
        raise Exception('Invalid y_part value=' + y_part)
    Y.append(yy)
print('build words map')

max_features = 50000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)
X, Xt, Y, Yt = train_test_split(X, Y, test_size = 0.3, random_state = random_seed)

validation_size = 1500
X_validate = Xt[-validation_size:]
Y_validate = Yt[-validation_size:]
Xt = Xt[:-validation_size]
Yt = Yt[:-validation_size]

maxlen = 0
def wrap_array(x, maxlen):
    for index in range(len(x)):
        xx = x[index]
        if len(xx) > maxlen:
            maxlen = len(xx)
        x[index] = np.array(xx)
    return np.array(x), maxlen

X, maxlen = wrap_array(X, maxlen)
Xt, maxlen = wrap_array(Xt, maxlen)
X_validate, maxlen = wrap_array(X_validate, maxlen)
Y, maxlen = wrap_array(Y, maxlen)
Yt, maxlen = wrap_array(Yt, maxlen)
Y_validate, maxlen = wrap_array(Y_validate, maxlen)
print('build model')

batch_size = 256

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(124, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
print(model.summary())

print('Train...')
model.fit(X, Y, batch_size=batch_size, epochs=2, validation_data=(Xt, Yt), verbose=2)

score, acc = model.evaluate(X_validate, Y_validate, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('Trying to predict:')
text = "This is a presidential decree that Rada's defence committee approved and suggested MPs to support. As some MPs write, there is no agreement on a restriction of certain freedoms(frdm of assembly among them). They also want it written down that elections will take place on March 31"
print(text)
tokens = tokenizer.texts_to_sequences([text])
tokens = pad_sequences(tokens, maxlen=maxlen)
sentiment = model.predict(np.array(tokens), batch_size=1, verbose = 2)[0][0]
print()
print('Sentiment =', sentiment)
if (round(sentiment) == 0):
    print('Negative')
else:
    print('Positive')