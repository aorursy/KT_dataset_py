from keras.models import Sequential

from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, LSTM, MaxPooling1D,BatchNormalization,GRU, SpatialDropout1D

from keras.layers.core import Dense, Dropout, Activation

from keras.preprocessing.text import Tokenizer

from keras import metrics, regularizers

from keras.preprocessing import sequence

import pandas as pd

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.layers import Dropout

import re

from nltk.corpus import stopwords

from nltk import word_tokenize

STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
data = pd.read_csv('../input/Top30.csv')
data.info()
data.Query.value_counts()
data = data.reset_index(drop=True)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.

    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = text.replace('x', '')

#    text = re.sub(r'\W+', '', text)

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text

data['Description'] = data['Description'].apply(clean_text)

data['Query'] = data['Query'].apply(clean_text)
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 250

# This is fixed.

EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(data['Description'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(data['Description'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(data['Query']).values

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(30, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
epochs = 5

batch_size = 64



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();


plt.title('Accuracy')

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show();