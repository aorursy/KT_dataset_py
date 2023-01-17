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
data=pd.read_csv("../input/knearest-neighbour-knn-classification/data.csv")

df = data.copy()

df.head()
df.columns
df.info()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df.diagnosis = le.fit_transform(df['diagnosis'])
df.drop(["Unnamed: 32"], axis=1, inplace=True)
df.head()
df.isnull().sum()
# Load libraries

import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
X = df.drop(["diagnosis"], axis=1) # Features

y = df["diagnosis"] # Target variable
# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
predictors = df.drop(["diagnosis"], axis=1)

target = df["diagnosis"]

from sklearn.model_selection import train_test_split # Import train_test_split function

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state = 0)





from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier





models = []

models.append(('Logistic Regression', LogisticRegression()))

models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 

models.append(('K-NN', KNeighborsClassifier()))

models.append(('SVM', SVC()))

models.append(('RandomForestClassifier', RandomForestClassifier()))





for name, model in models:

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    from sklearn import metrics



    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred,labels=[1,0]))

import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSION MATRIX VISUALIZATION")

plt.show()
from sklearn.metrics import f1_score

f1_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

probs = model.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.metrics import log_loss

log_loss(y_test,y_pred)