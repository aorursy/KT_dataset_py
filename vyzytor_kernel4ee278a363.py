# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Ignore warnings

import warnings

warnings.filterwarnings("ignore")

from IPython.core.debugger import set_trace



%matplotlib inline



# import base packages into the namespace for this program

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import time

import os

import subprocess

import sklearn

assert sklearn.__version__ >= "0.20"

import seaborn as sns  

import pandas as pd



#SKlearn

from sklearn.datasets import fetch_openml

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, classification_report

from sklearn.model_selection import train_test_split
# seed value for random number generators to obtain reproducible results

RANDOM_SEED = 85

# Get MNIST Data Set  ( https://github.com/ageron/handson-ml/issues/301#issuecomment-448853256 )

# The issue of obtaining MNIST data is solved by following "https://github.com/ageron/handson-ml/issues/143".

#from sklearn.datasets import fetch_openml

#mnist = fetch_openml('mnist_784', version=1, cache=True)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



os.getcwd() 

print(os.getcwd())

#Validate Current Path and create Path to data

from pathlib import Path

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)







X_test = pd.read_csv(INPUT/'test.csv')

train = pd.read_csv(INPUT/'train.csv')



X_train = train.drop(['label'], axis='columns', inplace=False)

y_train = train['label']

print(X_train.shape)

print(y_train.shape)
#rerun the experiment, using 70/30 split and two part PCA, train and test separately

#First Random Forest 



# RANDOM FOREST on Dimension Reduced Data (PCA:95% variability)

RF_clf = RandomForestClassifier(

  bootstrap = True,

  n_estimators=10,

  max_features='sqrt', 

  random_state=RANDOM_SEED

)



start_RF = time.clock()



RF_clf.fit(X_train, y_train)



RF_CrossVal = cross_val_score(

  RF_clf, 

  X_train, y_train, 

  cv=10, 

  scoring='f1_macro'

)

print(RF_CrossVal)



y_pred = cross_val_predict(

  RF_clf,

  X_train, y_train,

  cv=10

)

print(classification_report(y_train, y_pred))



RF_clf_score = RF_clf.score(X_train, y_train)

print('Accuracy Score for Random Forest: {:.3f}'.format(RF_clf_score))



f1score_RF_clf = f1_score(y_train, y_pred, average='macro')

print('F1 Score for Random Forest: {:.3f}'.format(f1score_RF_clf))



stop_RF = time.clock()

time_RF = stop_RF - start_RF



print('Start time for Random Forest: {:.3f}'.format(start_RF))

print('End_time for Random Forest: {:.3f}'.format(stop_RF))

print('Runtime for Random Forest: {:.3f}'.format(time_RF))

column_names = ['ImageId','Label']

results =pd.DataFrame(columns = column_names)

results['Label'] = pd.Series(RF_clf.predict(X_test))

IDdata = pd.DataFrame(X_test)



results['ImageId'] = X_test.index +1#sub = results[['ImageId','Label']]

sub = results[['ImageId','Label']]

sub.to_csv('submissionRF_noPCA.csv', index=False)
#Now Random Forest with reduced data

start_RF_reduced = time.clock()



rnd_pca = PCA(n_components=.95)

X_train_reduced = rnd_pca.fit_transform(X_train)

X_test_reduced = rnd_pca.transform(X_test)



# RANDOM FOREST on Dimension Reduced Data (PCA:95% variability)

RF_clf_reduced = RandomForestClassifier(

  bootstrap = True,

  n_estimators=10,

  max_features='sqrt', 

  random_state=RANDOM_SEED

)



RF_clf_reduced.fit(X_train_reduced, y_train)



RFReducedCrossVal = cross_val_score(

  RF_clf_reduced, 

  X_train_reduced, y_train, 

  cv=10, 

  scoring='f1_macro'

)

print(RFReducedCrossVal)



y_pred_reduced = cross_val_predict(

  RF_clf_reduced,

  X_train_reduced, y_train,

  cv=10

)

print(classification_report(y_train, y_pred_reduced))



RF_clf_reduced_score = RF_clf_reduced.score(X_train_reduced, y_train)

print('Accuracy Score for Random Forest Reduced: {:.3f}'.format(RF_clf_reduced_score))



f1score_RF_clf_reduced = f1_score(y_train, y_pred_reduced, average='macro')

print('F1 Score for Random Forest Reduced: {:.3f}'.format(f1score_RF_clf_reduced))



stop_RF_reduced = time.clock()

time_RF_reduced = stop_RF_reduced - start_RF_reduced



print('Start time for Random Forest PCA Compressed: {:.3f}'.format(start_RF_reduced))

print('End_time for Random Forest PCA Compressed: {:.3f}'.format(stop_RF_reduced))

print('Runtime for Random Forest PCA Compressed: {:.3f}'.format(time_RF_reduced))
column_names = ['ImageId','Label']

results =pd.DataFrame(columns = column_names)

results['Label'] = pd.Series(RF_clf_reduced.predict(X_test_reduced))

IDdata = pd.DataFrame(X_test_reduced)



results['ImageId'] = X_test.index +1#sub = results[['ImageId','Label']]

sub = results[['ImageId','Label']]

sub.to_csv('submissionRF_PCA.csv', index=False)
#Compare the results of the test

print('Compare the time:')

print('Random Forest Time no PCA: {:.3f}'.format(time_RF))

print('Random Forest Time including PCA: {:.3f}'.format(time_RF_reduced))

print(' ')

print('Compare the accuracy scores:')

print('Random Forest Accuracy Score no PCA: {:.3f}'.format(RF_clf_score))

print('Random Forest Accuracy Score with PCA: {:.3f}'.format(RF_clf_reduced_score))

print(' ')

print('Compare the F1 scores:')

print('Random Forest F1 Score no PCA: {:.3f}'.format(f1score_RF_clf))

print('Random Forest F1 Score with PCA: {:.3f}'.format(f1score_RF_clf_reduced))
#Graph the cross val scores

pred = [0,1,2,3,4,5,6,7,8,9]

plt.figure()

plt.plot(pred, RF_CrossVal, 'r',label='RF_CrossVal')

plt.plot(pred, RFReducedCrossVal, 'b',label='RF_ReducedCrossVal')

plt.xlabel('Predicted Values')

plt.ylabel('Cross Validation Score')

plt.title('Cross Validation Comparison')

plt.legend(loc="center right")



plt.show()