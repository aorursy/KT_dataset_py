# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

dataset = pd.read_csv('../input/digitsdata/digits.csv')

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,40].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Using RandomForest as the prediction is more accurate and optimal than other classifiers 

#Increasing the number n_estimators will increase the accuracy of the prediction but will result in overfitting, 10 is optimal for this dataset size

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix of find the number of incorrect predictions

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
#Displaying the confusion matrix, there are two incorrect predictions.

cm
#test set of labels

y_test
# predicted labels

y_pred
from sklearn.metrics import accuracy_score

print('Accuracy Score', accuracy_score(y_test, y_pred))