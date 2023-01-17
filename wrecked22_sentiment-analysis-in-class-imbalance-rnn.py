import numpy as np 

import pandas as pd 

import os

import re

import itertools

import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras_preprocessing.text import Tokenizer 

from keras_preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix
SA = pd.read_csv("../input/Sentiment.csv")

SA = SA[["sentiment","text"]]

SA['text'] = SA['text'].map(lambda x: x.lstrip('RT @').rstrip('@'));
SA = SA[SA.sentiment!="Neutral"]

SA['text'] = SA['text'].apply(lambda x: x.lower())

SA['text'] = SA['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
sns.countplot(SA.sentiment)

plt.title('Categories') 
SA = SA.drop(SA[SA.sentiment == "Negative"].iloc[:5000].index)
max_fatures = 4000

tokenizer = Tokenizer(num_words = max_fatures, split=' ')

tokenizer.fit_on_texts(SA['text'].values)

X = tokenizer.texts_to_sequences(SA['text'].values)

X = pad_sequences(X)

Y = SA['sentiment']
Y = Y.values



l =[]

for i in range(5729):

    if Y[i]=="Negative":

                   l.append(0)

    elif Y[i]=="Positive":

                   l.append(1)
sns.countplot(l)

plt.title('Categories') 
train_X , test_X , train_Y, test_Y = train_test_split(X, l, test_size=0.3, random_state=2, shuffle= True, stratify=l)
embed_dim = 128

lstm_out = 196

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(max_fatures, 128, input_length=train_X.shape[1]))

model.add(tf.keras.layers.SpatialDropout1D(0.4))

model.add(tf.keras.layers.LSTM(196, dropout = 0.3, recurrent_dropout = 0.3 ))

model.add(tf.keras.layers.Dense(100, activation = tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.45))

model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))
model.summary()
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
Model = model.fit(train_X, train_Y, 

                           validation_split=0.2,

                           epochs=10,

                           verbose=2,

                           batch_size = 32)
score = model.evaluate(test_X, test_Y, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



y_pred = model.predict(test_X)

y_pred_classes = np.argmax(y_pred,axis = 1) 

confusion_mtx = confusion_matrix(test_Y, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 
twt = ['Meetings: Because none of us is as dumb as all of us.']

twt = tokenizer.texts_to_sequences(twt)

twt = pad_sequences(twt, maxlen=29, dtype='int32', value=0)

print(twt)

sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]

if(np.argmax(sentiment) == 0):

    print("negative")

elif (np.argmax(sentiment) == 1):

    print("positive")