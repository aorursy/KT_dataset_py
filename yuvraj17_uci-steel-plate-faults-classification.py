# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import 

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/faults.csv')
data.drop('TypeOfSteel_A400', axis = 1)

features = data.values

labels = features[:,27:34]

features = features[:,0:27]
df = pd.DataFrame(features)

df.corr()

#no correlation found.
df.isnull().values.any()

# No values or NaN or Null
labels = [np.argmax(row) for row in labels]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,test_size=0.30,random_state=42)
rf_clf = RandomForestClassifier(random_state=17, min_samples_split=2, n_estimators=55)

# Parameters tuned using GridCV function below. Avoided running here for brevity and saving time

rf_clf.fit(features_train,labels_train)

rf_pred = rf_clf.predict(features_test)
print(accuracy_score(rf_pred, labels_test))

print(classification_report(rf_pred,labels_test))
# these functions will be time consuming, depending on the number of parameters

# input features and class labels

# return tuned classifier and the best score

def TuneRandomForest(features, labels):

    min_samples_split = np.arange(2,100)

    n_estimators = np.arange(10,100)

    parameters = {'n_estimators' : n_estimators,'min_samples_split':min_samples_split}

    clf = RandomForestClassifier()

    return gridCVTune(clf,parameters)



# inputs are the paramaters you wish to tune for and model and features and class labels

# output is tuned model and best score.

def gridCVTune(clf,parameters, features, labels):

    gridCV= GridSearchCV(clf,parameters)

    gridCV.fit(features,labels)

    return gridCV.best_estimator_, gridCV.best_score_