# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path='../input/'

card_data = pd.read_csv(path+'creditcard.csv')
card_data = card_data.dropna()
from sklearn.utils import shuffle

card_data = shuffle(card_data)

print(card_data.shape)
trn_features = card_data.iloc[:278000,1:30]

val_features=card_data.iloc[278000:,1:30]

trn_labels = card_data.iloc[:278000,30:]

val_labels=card_data.iloc[278000:,30:]

train=card_data.sample(frac=0.8,random_state=200)

test=card_data.drop(train.index)
trn_features = train.iloc[:,1:30]

val_features=train.iloc[:,1:30]

trn_labels = train.iloc[:,30:]

val_labels=train.iloc[:,30:]

print(trn_features.shape)
print(trn_features.shape,val_features.shape,val_labels.shape,trn_labels.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Conv1D

model = Sequential()

model.add(keras.layers.BatchNormalization(axis=-1, input_shape=(29,),momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='relu'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='relu'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(29, activation='tanh'))

model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='rmsprop',

              loss='sparse_categorical_crossentropy',

              metrics=['binary_accuracy'])
model = Sequential()

model.add(keras.layers.BatchNormalization(axis=-1, input_shape=(29,),momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

keras.layers.Conv1D(784, 10, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv1D(784, 10, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')



keras.layers.Conv1D(1568, 10, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv1D(1568, 10, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.MaxPooling1D(pool_size=1, strides=None, padding='valid')



keras.layers.Conv1D(3136, 10, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv1D(3136, 10, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.MaxPooling1D(pool_size=1, strides=None, padding='valid')



keras.layers.Conv1D(3000, 10, strides=1, padding='valid', dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv1D(3000, 10, strides=1, padding='valid', dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.MaxPooling1D(pool_size=1, strides=None, padding='valid')



keras.layers.Conv1D(3000, 10, strides=1, padding='valid', dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv1D(3000, 10, strides=1, padding='valid', dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.MaxPooling1D(pool_size=1, strides=None, padding='valid')



keras.layers.Conv1D(3000, 10, strides=1, padding='valid', dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv1D(3000, 10, strides=1, padding='valid', dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.MaxPooling1D(pool_size=1, strides=None, padding='valid')



model.add(Dense(1000, activation='tanh'))



model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='rmsprop',

              loss='sparse_categorical_crossentropy',

              metrics=['binary_accuracy'])
model.fit(x=trn_features, y=trn_labels, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.3, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
pred_labels = model.predict_classes(val_features)
import sklearn

from sklearn.metrics import average_precision_score

average_precision = sklearn.metrics.average_precision_score(trn_labels, pred_labels, average='weighted', sample_weight=None)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



precision, recall, _ = precision_recall_curve(val_labels, pred_labels)



plt.step(recall, precision, color='b', alpha=0.1,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='b')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.05f}'.format(

          average_precision))