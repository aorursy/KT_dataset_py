# Load libraries

import sys

import scipy

import numpy as np

import pandas as pd

from pandas import read_csv

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestClassifier

import pickle
# Load CSV using Pandas from URL

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(url, names=names)
# dimension (rows, columns)

data.shape
# head (first 10 rows)

data.head(10)
data.info()
# descriptive stats

data.describe()
# data types

data.dtypes
# distribution of an attribute (e.g. "class")

data.groupby('class').size()
# pairwise correlation between attributes

data.corr()
# missing values

data.isnull().sum()
# Dependent variable -- 'class'

sns.countplot(data['class'])
# Distribution of attribute -- "age"

f = plt.figure(figsize=(20,4))

f.add_subplot(1,2,1)

sns.distplot(data['age'])

f.add_subplot(1,2,2)

sns.boxplot(data['age'])
# distribution of 2 attributes to compare their shapes

f = plt.figure(figsize=(20,4))

f.add_subplot(1,3,1)

sns.countplot(data['age'], color='red')

f.add_subplot(1,3,2)

sns.countplot(data['preg'], color='yellow')

f.add_subplot(1,3,3)

sns.countplot(data['pres'], color='green')
# histograms

data.hist()

plt.show()
# box and whisker plots

data.plot(kind = 'box')

plt.show()
# scatter plot matrix

scatter_matrix(data)

plt.show()
# Standardize data (0 mean, 1 stdev)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)

rescaledX = scaler.transform(x)



# Show head of transformed data

(pd.DataFrame(rescaledX)).head(5)
# Create x (independent, input) + y (dependent, output) variables

x = data.drop(columns=['class'])

y = data['class']



# Split train/validation datasets (80-20%)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=7)



print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
x_test.shape, y_test.shape
# Prepare models

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

models.append(('RF', RandomForestClassifier(n_estimators=100, max_features=3)))
# Evaluate each model's accuracy on the validation set

print('Cross Validation Score: Mean accuracy & SD')

results = []

names = []

for name, model in models:

	kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)

	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

	results.append(cv_results)

	names.append(name)

	print('%s: %.2f%% (%.3f)' % (name, cv_results.mean()*100, cv_results.std()))

    

# Visualize model comparison

plt.boxplot(results, labels=names)

plt.title('Model Comparison: Cross Validation Score')

plt.show()
# Evaluate 1 model using Cross Validation

kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)

KNNresults = cross_val_score(KNeighborsClassifier(), x_train, y_train, cv=kfold, scoring='accuracy')

print('KNN: %.2f%% (%.3f)' % (KNNresults.mean()*100, KNNresults.std()))



# Visualize model

plt.boxplot(KNNresults)

plt.show()
print('Train Set Performance Metrics: Accuracy & ROC')

for name, model in models:

    trained_model = model.fit(x_train, y_train)

    y_train_pred = trained_model.predict(x_train)

    print('%s: %.2f%% (%.3f)' % (name, accuracy_score(y_train, y_train_pred)*100, (roc_auc_score(y_train, y_train_pred))))
print('Test Set Performance Metrics: Accuracy & ROC')

for name, model in models:

    trained_model = model.fit(x_train, y_train)

    y_test_pred = trained_model.predict(x_test)

    print('%s: %.2f%% (%.3f)' % (name, accuracy_score(y_test, y_test_pred)*100, (roc_auc_score(y_test, y_test_pred))))
# Train model

knn = KNeighborsClassifier().fit(x_train, y_train)





# Predict y on train set

y_train_pred = knn.predict(x_train)



# Train set performance metrics

print('Train Set Performance Metrics: Accuracy & ROC')

print('%.2f%% (%.3f)' % (accuracy_score(y_train, y_train_pred)*100, roc_auc_score(y_train, y_train_pred)))



# Predict y on test set

y_test_pred = knn.predict(x_test)



# Test set performance metrics

print('Train Set Performance Metrics: Accuracy & ROC')

print('%.2f%% (%.3f)' % (accuracy_score(y_test, y_test_pred)*100, roc_auc_score(y_test, y_test_pred)))



# Confusion matrix

print('Confusion Matrix: \n %s' % (confusion_matrix(y_test, y_test_pred)))



# Classification report

print(classification_report(y_test, y_test_pred))
# K-Nearest Neighbors (KNN)

#Create dictionary of hyperparameters that we want to tune

knn_params = {

    'n_neighbors':[1,3,5,7,9,11,15,17,19],

    'weights':['uniform', 'distance'],

    'metric':['euclidean', 'manhattan'],

    'leaf_size':list(range(1,50)),

    'p':[1,2,3]

}



# Create new KNN object using GridSearch

grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=10)



#Fit the model

best_model_knn = grid_knn.fit(x_train, y_train)



# Print the value of best hyperparameters

print('Best n_neighbors:', best_model_knn.best_estimator_.get_params()['n_neighbors'])

print('Best n_neighbors:', best_model_knn.best_estimator_.get_params()['weights'])

print('Best n_neighbors:', best_model_knn.best_estimator_.get_params()['metric'])

print('Best leaf_size:', best_model_knn.best_estimator_.get_params()['leaf_size'])

print('Best p:', best_model_knn.best_estimator_.get_params()['p'])

#print(best_model_knn.best_params_)
# Ridge Regression (RR)

#List hyperparameters that we want to tune

alpha = np.array([1,0.1,0.01,0.001,0.0001,0])



#Convert to dictionary

hyperparameters_ridge = dict(alpha=alpha)



# Create new Ridge object using GridSearch

grid_ridge = GridSearchCV(Ridge(), hyperparameters_ridge, cv=kfold)



#Fit the model

best_model_ridge = grid_ridge.fit(x,y)



#Print the value of best hyperparameters

print('Best score:', best_model_ridge.best_score_)

print('Best alpha:', best_model_ridge.best_estimator_.get_params()['alpha'])
print(best_model_knn.best_params_)
# Predict y on train set

y_train_pred_2 = best_model_knn.predict(x_train)



# Train set performance metrics

print('Train Set Performance Metrics: Accuracy & ROC')

print('%.2f%% (%.3f)' % (accuracy_score(y_train, y_train_pred_2)*100, roc_auc_score(y_train, y_train_pred_2)))



# Predict y on test set

y_test_pred_2 = best_model_knn.predict(x_test)



# Test set performance metrics

print('Train Set Performance Metrics: Accuracy & ROC')

print('%.2f%% (%.3f)' % (accuracy_score(y_test, y_test_pred_2)*100, roc_auc_score(y_test, y_test_pred_2)))





# Confusion matrix

print('Confusion Matrix: \n %s' % (confusion_matrix(y_test, y_test_pred_2)))



# Classification report

print(classification_report(y_test, y_test_pred_2))
# Save model to disk

FinalModel_KNN = 'FinalModel.sav'

pickle.dump(best_model_knn, open(FinalModel_KNN, 'wb'))
# Load model from disk

Load_FinalModel = pickle.load(open(FinalModel_KNN, 'rb'))
# Apply model to new dataset to make predictions

result = Load_FinalModel.score(x_test, y_test)

print('Accuracy: %.3f%%' % (result*100))