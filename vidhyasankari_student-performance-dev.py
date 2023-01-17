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
#import more libraries

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns 
data_SF=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
data_SF.head()
data_SF.describe()
data_SF.columns
f, axes = plt.subplots(1, 3)

sns.distplot(data_SF['reading score'],ax=axes[0])

sns.distplot(data_SF['writing score'],ax=axes[1])

sns.distplot(data_SF['math score'],ax=axes[2])

sns.pairplot(data_SF, hue='lunch', height=2.5);
sns.pairplot(data_SF, hue='gender', height=2.5)
from sklearn.model_selection import train_test_split

data=data_SF.copy()

data.columns

data=data.drop(['parental level of education'],axis=1)
# creating bool series True for NaN values  

bool_series = pd.isnull(data["math score"])  

    

# filtering data  

# displaying data only with Gender = NaN  

data[bool_series]  
# Get list of categorical variables

s = (data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
data.head()
y=dict()

for i in data['test preparation course']:

    print
y= data['test preparation course']



print(y)
one_hot = pd.get_dummies(data[object_cols])

# Drop column B as it is now encoded

data = data.drop(object_cols,axis = 1)

# Join the encoded df

data = data.join(one_hot)

data
X = data.iloc[:, :-2].values

print(X)



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(train_X, train_y)
model.predict(val_X)
model.score(val_X,val_y)
confusion_matrix(val_y, model.predict(val_X))
#Import svm model

from sklearn import svm



#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(train_X, train_y)



#Predict the response for test dataset

y_pred = clf.predict(val_X)
clf.score(val_X,val_y)
confusion_matrix(val_y,y_pred)
from sklearn.svm import SVC

clf1 = SVC(kernel='poly',degree=10)

clf1.fit(train_X, train_y)

y_pred1 = clf1.predict(val_X)
clf1.score(val_X,val_y)
confusion_matrix(val_y,y_pred1)
from sklearn.svm import SVC

clf2 = SVC(kernel='sigmoid')

clf2.fit(train_X, train_y)

y_pred2 = clf2.predict(val_X)
clf2.score(val_X,val_y)
confusion_matrix(val_y,y_pred2)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



#instantiate model and train

clf3 = XGBClassifier(learning_rate = 0.05, n_estimators=60, max_depth=10)

clf3.fit(train_X, train_y)



# make predictions for test set

y_pred3 = clf3.predict(val_X)





accuracy = accuracy_score(val_y, y_pred3)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
confusion_matrix(val_y,y_pred3)
from sklearn.neural_network import MLPClassifier

accuracy={}

for i in range(1,500):

    clf4 = MLPClassifier(solver='lbfgs', alpha=1e-7,hidden_layer_sizes=(12, 2),max_iter=i,random_state=1)

    clf4.fit(X, y)

# make predictions for test set

    y_pred4 = clf4.predict(val_X)

    accuracy[i] = accuracy_score(val_y, y_pred4)

    #print("Accuracy: %.2f%%" % (accuracy[i] * 100.0))

    #confusion_matrix(val_y,y_pred4)

accuracy
#from sklearn.neural_network import MLPClassifier

clf4 = MLPClassifier(solver='lbfgs', alpha=1e-7,hidden_layer_sizes=(12, 2),max_iter=115,random_state=1)

clf4.fit(X, y)

# make predictions for test set

y_pred4 = clf4.predict(val_X)

accuracy = accuracy_score(val_y, y_pred4)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

confusion_matrix(val_y,y_pred4)
