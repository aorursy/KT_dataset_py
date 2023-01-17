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
# define regression model

def regression_model():

    # create model

    model = Sequential()

    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    

    # compile model

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
predictors_norm = (predictors - predictors.mean()) / predictors.std()

predictors_norm.head()
n_cols = predictors.shape[1] # number of predictors
# build the model

model = regression_model()

lst = []

acc = []

# fit the model

p_train, p_test, t_train, t_test = train_test_split(predictors_norm, target, test_size=0.30)

   

history=model.fit(p_train, t_train, epochs=1000, verbose=0)

    

predictions = model.predict(p_test, verbose=0)

    

print(mean_squared_error(t_test,predictions))

print(metrics.accuracy_score(t_test,predictions.round()))
print(history.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy

fig = plt.figure(figsize=(15,5))

plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()


# summarize history for loss

fig = plt.figure(figsize=(15,5))

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
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
model.save('model')
test1=pd.read_csv("/kaggle/input/eeg-dataset-of-slow-cortical-potentials/bsi_competition_ii_test1a.csv")
final_pred = model.predict(test1, verbose=0)

final_pred_classes = model.predict_classes(test1, verbose=0)
final_pred
final_pred_classes
test_with_pred=test1

test_with_pred['0']=final_pred_classes

test_with_pred
test_with_pred.to_csv("test1a_with_prediction.csv",index=False)