# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/banknote-bill-authentication/bill_authentication.csv")
data.head()
data.shape
#Spliting features and target columns

X = data.drop('Class', axis=1)

y = data['Class']
#training and test data splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train.head()
y_train.head()
#model selection and training 

classifier_model = DecisionTreeClassifier()

classifier_model.fit(X_train, y_train)
#prediction 

prediction = classifier_model.predict(X_test)

prediction
#training validation , confusion matrix and classification report 

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test,prediction))