import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, SpatialDropout1D

import tensorflow as tf
session_config = tf.ConfigProto(
    log_device_placement=True,
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0,
    allow_soft_placement=True)
sess = tf.Session(config=session_config)


print(os.listdir('../input/'))
train = pd.read_csv('../input/train.tsv', sep='\t')

print('train set: {0}'.format(train.shape))
train.head(10)
test = pd.read_csv('../input/test.tsv', sep='\t')

print('test set: {0}'.format(train.shape))
test.head(10)
plt.figure(figsize=(10, 8))

plt.hist(train['Sentiment'], 5, alpha=0.2, density=True)

plt.hist(
    [train.loc[train['Phrase'].apply(lambda p: ',' in p), 'Sentiment'],
     train.loc[train['Phrase'].apply(lambda p: '.' in p), 'Sentiment'],
     train.loc[train['Phrase'].apply(lambda p: '!' in p), 'Sentiment'],
     train.loc[train['Phrase'].apply(lambda p: '?' in p), 'Sentiment']],
    5, alpha=0.5, density=True,
    label=[',', '.', '!', '?'])

plt.xlabel('sentiment')
plt.ylabel('probability')
plt.grid(alpha=0.25)
plt.legend(loc='upper right')
replace_list = {r"i'm": 'i am',
                r"'re": ' are',
                r"let’s": 'let us',
                r"'s":  ' is',
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan’t": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',
                ',': ' ,',
                '.': ' .',
                '!': ' !',
                '?': ' ?',
                '\s+': ' '}

def clean_text(text):
    text = text.lower()
    for s in replace_list:
        text = text.replace(s, replace_list[s])
    text = ' '.join(text.split())
    return text

X_train = train['Phrase'].apply(lambda p: clean_text(p))
phrase_len = X_train.apply(lambda p: len(p.split(' ')))
max_phrase_len = phrase_len.max()+10
print('max phrase len: {0}'.format(max_phrase_len-10)+
      '\nuse maxlen: {0}'.format(max_phrase_len))

plt.figure(figsize=(10, 8))
plt.hist(phrase_len, alpha=0.2, density=True)
plt.xlabel('phrase len')
plt.ylabel('probability')
plt.grid(alpha=0.25)
y_train = train['Sentiment']

tokenizer = Tokenizer(num_words=8192,
                      filters='"#$%&()*+-/:;<=>@[\]^_`{|}~')
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_phrase_len)
y_train = to_categorical(y_train.values)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
print('X_train size: {0}, '.format(X_train.shape)+
      'y_train size: {0}\n'.format(y_train.shape)+
      'X_val size: {0}, '.format(X_val.shape)+
      'y_val size: {0}'.format(y_val.shape))

model_cnn = Sequential()
model_cnn.add(Embedding(8192, 256))
model_cnn.add(SpatialDropout1D(0.5))
model_cnn.add(Conv1D(128, 3, padding='same', activation='relu', strides=1))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(256, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(5, activation='softmax'))
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.summary()
model_cnn.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=8, batch_size=1024, verbose=1)
model_lstm = Sequential()
model_lstm.add(Embedding(8192, 256))
model_lstm.add(SpatialDropout1D(0.3))
model_lstm.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
model_lstm.add(Dense(256, activation='relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(5, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_lstm.summary()
# for lstm batch_size should not be too large
model_lstm.fit(X_train, y_train,
               validation_data=(X_val, y_val),
               epochs=8, batch_size=512, verbose=1)
X_test = test['Phrase'].apply(lambda p: clean_text(p))

print('X_train size: {0}'.format(X_test.apply(lambda p: len(p.split(' '))).max()))

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_phrase_len)

sub = pd.read_csv('../input/sampleSubmission.csv')
sub.head()
y_cnn = model_cnn.predict(X_test)
y_lstm = model_lstm.predict(X_test)

y = y_cnn+y_lstm
y = np.argmax(y, axis=1)
y[:10]
sub = pd.read_csv('../input/sampleSubmission.csv')
sub['Sentiment'] = y
sub.head()
sub.to_csv('test_Submission_3.csv', index=False)
