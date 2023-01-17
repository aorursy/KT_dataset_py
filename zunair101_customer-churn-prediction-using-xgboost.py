import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn.tree import DecisionTreeClassifier

import gc

from imblearn.under_sampling import TomekLinks

from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import numpy as np

# import pickle

import matplotlib.cm as cm

import seaborn as sn

from collections import Counter

import lightgbm as lgb

# from kmodes.kprototypes import KPrototypes

import gc

# %reload_ext autotime
dataset = pd.read_csv('../input/customer-churn-dataset/Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values

y = dataset.iloc[:, -1].values
print(X)
print(y)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 2] = le.fit_transform(X[:, 2])
print(X)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)

#print(cnf_matrix)

accuracy_score(y_test, y_pred)
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.summer):

    plt.clf

    plt.imshow(cm, interpolation='nearest')

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(target_names))

    plt.xticks(tick_marks, target_names, rotation=45)

    plt.yticks(tick_marks, target_names)

    plt.tight_layout()

 

    width, height = cm.shape

 

    for x in range(width):

        for y in range(height):

            plt.annotate(str(cm[x][y]), xy=(y, x), 

                        horizontalalignment='center',

                        verticalalignment='center',color='black',fontsize=22)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
plot_confusion_matrix(cnf_matrix, np.unique(y_pred))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))