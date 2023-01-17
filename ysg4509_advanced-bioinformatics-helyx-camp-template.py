# here we will import the libraries used for machine learning

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot

%matplotlib inline



from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

# from sklearn.cross_validation import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model

# Any results you write to the current directory are saved as output.

# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
# importing data

data = pd.read_csv("../input/data.csv", header=0)



#visualizing the data

data.head(2)
data.info()
data.drop("Unnamed: 32", axis = 1, inplace=True)



data.columns
data.drop("id", axis = 1, inplace=True)



data.columns
features_mean = list(data.columns[1:11])

features_se = list(data.columns[11:20])

features_worst = list(data.columns[21:31])



print(features_mean)

print('----------------------------')

print(features_se)

print('----------------------------')

print(features_worst)
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
data.describe()
sns.countplot(data['diagnosis'], label = "Count")
corr = data[features_mean].corr()



plt.figure(figsize=(14, 14))



sns.heatmap(corr, cbar = True, square = True, annot = True, fmt='.2f', annot_kws = {'size': 15}, 

            xticklabels = features_mean, yticklabels= features_mean, cmap='coolwarm')
prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
train, test = train_test_split(data, test_size = 0.3)



print(train.shape)

print(test.shape)
train_X = train[prediction_var]

train_y = train.diagnosis



test_X = test[prediction_var]

test_y = test.diagnosis
model = RandomForestClassifier(n_estimators = 100)
model.fit(train_X, train_y)
prediction = model.predict(test_X)



print(prediction)
metrics.accuracy_score(prediction, test_y)
model = svm.SVC()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print(prediction)

print('--------------------------')

metrics.accuracy_score(prediction, test_y)
prediction_var = features_mean
train_X = train[prediction_var]

train_y = train.diagnosis



test_X = test[prediction_var]

test_y = test.diagnosis
model = RandomForestClassifier(n_estimators = 100)



model.fit(train_X, train_y)

prediction = model.predict(test_X)

print(prediction)

metrics.accuracy_score(prediction, test_y)
featimp = pd.Series(model.feature_importances_, index = prediction_var).sort_values(ascending = False)

print(featimp)
prediction_var = features_worst



train_X = train[prediction_var]

train_y = train.diagnosis



test_X = test[prediction_var]

test_y = test.diagnosis



model = RandomForestClassifier(n_estimators = 100)

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print(prediction)

metrics.accuracy_score(prediction, test_y)
# Random Forest Classifier



prediction_var = features_se



train_X = train[prediction_var]

train_y = train.diagnosis



test_X = test[prediction_var]

test_y = test.diagnosis



model = RandomForestClassifier(n_estimators = 100)

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print(prediction)

metrics.accuracy_score(prediction, test_y)
# SVM

prediction_var = features_se



train_X = train[prediction_var]

train_y = train.diagnosis



test_X = test[prediction_var]

test_y = test.diagnosis



model = svm.SVC()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print(prediction)

metrics.accuracy_score(prediction, test_y)