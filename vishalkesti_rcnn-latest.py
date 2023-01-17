# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.python.client import device_lib



print(device_lib.list_local_devices())
import numpy as np

import pandas as pd

from collections import defaultdict

import re





from bs4 import BeautifulSoup



import sys

import os

import keras



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical



from keras.layers import Embedding

from keras import regularizers

from keras.callbacks import EarlyStopping

from keras.layers import Dense, Input, Flatten

from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout, BatchNormalization

from keras.models import Model

from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.layers.normalization import BatchNormalization

from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU

from keras.layers import Dense, Embedding, LSTM, GRU





MAX_SEQUENCE_LENGTH = 1000

MAX_NB_WORDS = 200000

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2
data = pd.read_csv('/kaggle/input/ticnn-fake-real/TICNN_fake_real.csv')

data.head()
def clean_str(string):

    """

    Cleaning of dataset

    """

    string = re.sub(r"\\", "", string)    

    string = re.sub(r"\'", "", string)    

    string = re.sub(r"\"", "", string)    

    return string.strip().lower()

# Input Data preprocessing

# data_train = pd.read_csv('./data/TI CNN fake news dataset all_data.csv')

data['type'] = data['type'].replace('fake',1)

data['type'] = data['type'].replace('real',0)

print(data.columns)

print('What the raw input data looks like:')

print(data[0:5])

texts = []

labels = []



for i in range(data.text.shape[0]):

    text1 = data.title[i]

    text2 = data.text[i]

    text = str(text1) +""+ str(text2)

    texts.append(text)

    labels.append(data.type[i])

    

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
# Pad input sequences

final_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels),num_classes = 2)

print('Shape of data tensor:', final_data.shape)

print('Shape of label tensor:', labels.shape)
# Train test validation Split

from sklearn.model_selection import train_test_split



indices = np.arange(final_data.shape[0])

np.random.shuffle(indices)

final_data = final_data[indices]

labels = labels[indices]

x_train, x_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.20, random_state=42)

x_test, x_val, y_test, y_val = train_test_split(final_data, labels, test_size=0.50, random_state=42)

print('Size of train, validation, test:', len(y_train), len(y_val), len(y_test))



print('real & fake news in train,valt,test:')

print(y_train.sum(axis=0))

print(y_val.sum(axis=0))

print(y_test.sum(axis=0))
#Using Pre-trained word embeddings

GLOVE_DIR = "/kaggle/input/glove6b100dtxt/" 

embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")

for line in f:

    values = line.split()

    #print(values[1:])

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Total %s word vectors in Glove.' % len(embeddings_index))



embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

        

embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH)
# Simple CNN model

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)

l_pool1 = MaxPooling1D(5)(l_cov1)

l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)

l_pool2 = MaxPooling1D(5)(l_cov2)

l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)

l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling



l_flat = Flatten()(l_pool3)



l_dense = Dense(128, activation='relu')(l_flat)

l_b = BatchNormalization()(l_dense)

preds = Dense(2, activation='softmax')(l_b)



model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['acc'])





print("Fitting the simple convolutional neural network model")

model.summary()



# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks=[es], epochs=10, batch_size=128)
import matplotlib.pyplot as plt

%matplotlib inline 

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
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
#convolutional approach 2

convs = []

filter_sizes = [3,4,5]



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)



for fsz in filter_sizes:

    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)

    l_pool = MaxPooling1D(5)(l_conv)

    convs.append(l_pool)

    

l_concatenate = Concatenate(axis=-1)(convs)

l_cov1= Conv1D(filters=128, kernel_size=5, activation='relu')(l_concatenate)

l_pool1 = MaxPooling1D(5)(l_cov1)

l_cov2 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_pool1)

l_pool2 = MaxPooling1D(30)(l_cov2)

l_flat = Flatten()(l_pool2)

l_dense = Dense(128, activation='relu')(l_flat)

preds = Dense(2, activation='softmax')(l_dense)



model2 = Model(sequence_input, preds)

model2.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['acc'])



print("Fitting a more complex convolutional neural network model")

model2.summary()

# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history2 = model2.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks=[es], epochs=10, batch_size=50)

model2.save('model.h5')
# list all data in history

print(history2.history.keys())

import matplotlib.pyplot as plt

%matplotlib inline 

# summarize history for accuracy

plt.plot(history2.history['acc'])

plt.plot(history2.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
score = model2.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Recurrent CNN model

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)

l_pool1 = MaxPooling1D(5)(l_cov1)

l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)

l_pool2 = MaxPooling1D(5)(l_cov2)

l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)

l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling

l_pool4 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(l_pool3)

# l_flat = Flatten()(l_pool4)

l_b = BatchNormalization()(l_pool4)

l_dense1 = Dense(128,kernel_regularizer=regularizers.l2(0.001), activation='relu')(l_b)

l_dense2 = Dense(64, activation='relu')(l_dense1)



preds = Dense(2, activation='softmax')(l_dense2)



model3 = Model(sequence_input, preds)

model3.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['acc'])





print("Fitting the simple convolutional neural network model")

model3.summary()



# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history3 = model3.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[es],epochs=10, batch_size=128)

# list all data in history

print(history3.history.keys())

import matplotlib.pyplot as plt

%matplotlib inline 

# summarize history for accuracy

plt.plot(history3.history['acc'])

plt.plot(history3.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history3.history['loss'])

plt.plot(history3.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
score = model3.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Test model 1

test_preds = model.predict(x_test)

test_preds = np.round(test_preds)

correct_predictions = float(sum(test_preds == y_test)[0])

print("Correct predictions:", correct_predictions)

print("Total number of test examples:", len(y_test))

print("Accuracy of model1: ", correct_predictions/float(len(y_test)))



# Creating the Confusion Matrix

from sklearn.metrics import confusion_matrix

x_pred = model.predict(x_test)

x_pred = np.round(x_pred)

x_pred = x_pred.argmax(1)

y_test_s = y_test.argmax(1)

cm = confusion_matrix(y_test_s, x_pred)

plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')

plt.title('Confusion matrix - model1')

plt.colorbar()

plt.ylabel('expected label')

plt.xlabel('predicted label')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, test_preds, labels=[0,1]))
#Test model 2

test_preds2 = model2.predict(x_test)

test_preds2 = np.round(test_preds2)

correct_predictions = float(sum(test_preds2 == y_test)[0])

print("Correct predictions:", correct_predictions)

print("Total number of test examples:", len(y_test))

print("Accuracy of model2: ", correct_predictions/float(len(y_test)))



# Creating the Confusion Matrix

x_pred = model2.predict(x_test)

x_pred = np.round(x_pred)

x_pred = x_pred.argmax(1)

y_test_s = y_test.argmax(1)

cm = confusion_matrix(y_test_s, x_pred)

plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest',)

plt.title('Confusion matrix - model2')

plt.colorbar()

plt.ylabel('expected label')

plt.xlabel('predicted label')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, test_preds2, labels=[0,1]))
#Test model 3

test_preds3 = model3.predict(x_test)

test_preds3 = np.round(test_preds3)

correct_predictions = float(sum(test_preds3 == y_test)[0])

print("Correct predictions:", correct_predictions)

print("Total number of test examples:", len(y_test))

print("Accuracy of model3: ", correct_predictions/float(len(y_test)))



# Creating the Confusion Matrix

x_pred = model3.predict(x_test)

x_pred = np.round(x_pred)

x_pred = x_pred.argmax(1)

y_test_s = y_test.argmax(1)

cm = confusion_matrix(y_test_s, x_pred)

plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest',)

plt.title('Confusion matrix - model3')

plt.colorbar()

plt.ylabel('expected label')

plt.xlabel('predicted label')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, test_preds3, labels=[0,1]))