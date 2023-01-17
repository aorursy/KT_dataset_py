from __future__ import division

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
plt.style.use('fivethirtyeight')
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier
from deap import creator
from sklearn.model_selection import cross_val_score
data = pd.read_csv('../input/Sentiment.csv')
data.head()
data.groupby(['candidate']).size().drop('No candidate mentioned').sort_values().plot(kind = 'barh')
pd.crosstab(data.candidate, data.sentiment).drop('No candidate mentioned').sort_values('Negative', ascending = False)
data.groupby('user_timezone').size().sort_values(ascending = False)[:10].plot(kind='barh')
pd.crosstab(data.subject_matter, data.candidate)\
.drop('No candidate mentioned', axis = 1)\
.drop('None of the above').sort_values('Donald Trump', ascending = False)
# Keep the sentiments only
data = data[['text','sentiment']]
# Lowercase, clean non-alphanumeric characters and remove 'rt's
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x.lower()))\
.str\
.replace('rt', '') 
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
y = pd.get_dummies(data['sentiment'])
X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
tpot = TPOTClassifier(generations=5, max_time_mins=15, max_eval_time_mins=0.04, population_size=50, verbosity = 2)
tpot.fit(X_train, np.argmax(y_train.as_matrix(), axis = 1))
print(tpot.score(X_test, np.argmax(y_test.as_matrix(), axis = 1)))
pd.DataFrame(dict(list(tpot.evaluated_individuals_.items()))).T\
.replace([np.inf, -np.inf], np.nan)\
.dropna()\
.drop('generation', axis = 1)\
.sort_values('internal_cv_score', ascending = False)\
.head()
y_pred_tpot = tpot.predict(X_test)
conf_mat = cf(np.argmax(y_test.as_matrix(), axis = 1), y_pred_tpot)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
# Deep Learning Architecture Parameters
input_dim = 2000
output_dim = 128
dropout = 0.8
rec_dpout = 0.8
LSTM_units = 256

# Model fit Parameters
batch_size = 200
epochs = 30
val_split = 0.3
dense_out = 3 # three categories
model = Sequential()

model.add( Embedding(input_dim=input_dim, output_dim = output_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(dropout))
model.add(LSTM(LSTM_units, dropout = dropout, recurrent_dropout=rec_dpout,return_sequences=True))
model.add(LSTM(LSTM_units, dropout = dropout, recurrent_dropout=rec_dpout, return_sequences=False))
model.add( Dense(dense_out, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split=val_split)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
score,acc = model.evaluate(X_test, y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
y_pred = model.predict(X_test)

conf_mat = cf(np.argmax(y_test.as_matrix(), axis = 1), np.argmax(y_pred, axis = 1))

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()