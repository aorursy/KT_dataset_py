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
train1=pd.read_csv("/kaggle/input/eeg-dataset-of-slow-cortical-potentials/bsi_competition_ii_train1a.csv")

train2=pd.read_csv("/kaggle/input/eeg-dataset-of-slow-cortical-potentials/bsi_competition_ii_train1b.csv")
predictors = train1

target = predictors['0']
predictors.drop('0',axis=1,inplace=True)
predictors.head()
target.head()
import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from statistics import mean,stdev

from sklearn import metrics
from matplotlib import pyplot
predictors_norm = (predictors - predictors.mean()) / predictors.std()

predictors_norm.head()
n_cols = predictors.shape[1] # number of predictors
p_train, p_test, t_train, t_test = train_test_split(predictors_norm, target, test_size=0.30)
model = Sequential()

model.add(Dense(100, input_dim=5376, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model

history = model.fit(p_train, t_train, validation_data=(p_test, t_test), epochs=300, verbose=0)

# evaluate the model

_, train_acc = model.evaluate(p_train, t_train, verbose=0)

_, test_acc = model.evaluate(p_test, t_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot loss during training

pyplot.subplot(211)

pyplot.title('Loss')

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

# plot accuracy during training

pyplot.subplot(212)

pyplot.title('Accuracy')

pyplot.plot(history.history['accuracy'], label='train')

pyplot.plot(history.history['val_accuracy'], label='test')

pyplot.legend()

pyplot.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
# predict probabilities for test set

yhat_probs = model.predict(p_test, verbose=0)

# predict crisp classes for test set

yhat_classes = model.predict_classes(p_test, verbose=0)

# reduce to 1d array

yhat_probs = yhat_probs[:, 0]

yhat_classes = yhat_classes[:, 0]

 

# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(t_test, yhat_classes)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(t_test, yhat_classes)

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(t_test, yhat_classes)

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(t_test, yhat_classes)

print('F1 score: %f' % f1)

 

# kappa

kappa = cohen_kappa_score(t_test, yhat_classes)

print('Cohens kappa: %f' % kappa)

# ROC AUC

auc = roc_auc_score(t_test, yhat_probs)

print('ROC AUC: %f' % auc)

# confusion matrix

matrix = confusion_matrix(t_test, yhat_classes)

print("confusion_matrix:")

print(matrix)