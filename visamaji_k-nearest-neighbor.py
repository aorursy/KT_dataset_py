import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

#matplotlib.style.use( 'ggplot' )

sns.set_style("darkgrid")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/iris/Iris.csv')
df.head()
df = df.drop("Id", axis=1)
df.shape
df.describe(include="all")
df.isnull().sum()
sns.countplot(x = 'Species', data = df)
sns.pairplot(data = df, hue = 'Species', diag_kind='kde')

plt.show()
sns.heatmap(df.corr(),annot=True)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
from sklearn.preprocessing import LabelEncoder

#label encode the target variable

encode = LabelEncoder()

df.Species = encode.fit_transform(df.Species)
df.tail()
plt.figure(figsize=(15,8))

df.boxplot()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df)

scaled_df = pd.DataFrame(scaler.transform(df))

plt.figure(figsize=(15,8))

scaled_df.boxplot() 
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
X = df.iloc[:,0:-1]

y = df.iloc[:,-1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

print(X_train.shape,X_test.shape)
classifier = KNeighborsClassifier()

parameters = {'n_neighbors':np.arange(1,40)} # dictionary that contains all the possible values that i want to test



from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(classifier, parameters, cv=3, scoring = 'accuracy', verbose=50, n_jobs=-1)

gs = gs.fit(X_train, y_train)
print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))

means = gs.cv_results_['mean_test_score']

stds = gs.cv_results_['std_test_score']

params = gs.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("Mean %f Std (%f) with: %r" % (mean, stdev, param))
plt.figure(figsize=(15,8))

plt.plot (means, color='blue', alpha=1.00)

plt.show()
best_model = gs.best_estimator_

y_pred = best_model.predict(X_test)
best_model
y_pred
print('***RESULTS ON TEST SET***')

print("precision: ", metrics.precision_score(y_test, y_pred, average='micro')) # tp / (tp + fp)

print("recall: ", metrics.recall_score(y_test, y_pred, average='micro')) # tp / (tp + fn)

print("f1_score: ", metrics.f1_score(y_test, y_pred, average='micro')) #F1 = 2 * (precision * recall) / (precision + recall)

print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m
y_pred = pd.Series(y_pred,name="Label")



y_pred.to_csv('result.csv', index = False)