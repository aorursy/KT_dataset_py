#Importing required packages.

import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict, StratifiedKFold



%matplotlib inline

# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
# print the folder name, so it can be added to the file path in next code kernel below (before adding the csv file name)

print(os.listdir("../input"))
#Loading dataset

wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# checking a first few lines of the dataset

wine.head()
# to check the variables we are working with

wine.info()
# we can see non-null for each column but in case we want to check how many nulls in each column

wine.isnull().sum()
# check the range of values in Quality column so that bins for Good and Bad wines can be created

wine['quality'].hist()
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
# now the column has been changed form numerical to categorical

wine['quality'].unique()
# to encode bad as 0 and good as 1, use the sklearn preprocessing function

label_quality = LabelEncoder()

wine['quality'] = label_quality.fit_transform(wine['quality'])
#check the dataset again to see the quality column having binary values

wine.head(10)
# count number of good and bad quality wines

wine['quality'].value_counts()
#bar plot using Seaborn package

sns.countplot(wine['quality'])
# without using the seaborn package, generating the visualization

# wine['quality'].hist() # this treats the 0 and 1's as integers instead of categories

wine['quality'].value_counts().plot(kind='bar')
# Dividing dataset into predictor (X) and response features (y)

X = wine.drop('quality', axis = 1) # axis = 0 means row; axis = 1 means columns; so here we select all columns except quality

y = wine['quality']
# Split into train and test datasets (using sklearn package)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # using 20% of data for testing hence 0.2; random state is like setting a seed
# We scale the data using standardized scaling so that columns with higher numerical values (eg. total sulphur dioxide) are not biased compared to columns with very small numerical values (e.g. chlorides)

sc = StandardScaler()

X_train = sc.fit_transform(X_train) #fit_transform is gonna fit AND transform at the same time, much like label encoder

X_test = sc.transform(X_test) # we want the same fit (i.e. values of mean and standard deviation for each column) that we used for (centering the) training data so instead of fit_transform() (which internally calls fit() followed by transform()), we just use transform()
#checking to see how the scaled valued look like

X_train[1:10] #since X_train is an array now, not a dataframe
rfc = RandomForestClassifier(n_estimators= 200) # n_estimators equals how many forests do you need. start with a higher number and bring it down slowly as smaller the model better the fit

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
#check some of the predicted values

pred_rfc[1:40]
#confusion matrix

print(confusion_matrix(y_test, pred_rfc))
#let's check model's accuracy at prediction using the testing set we have (i.e. y_test)

print(classification_report(y_test, pred_rfc))
# another way to print out accuracy explicity (using scikit-learn)

accuracy_score(y_test, pred_rfc)
f1_score(y_test, pred_rfc)
clf = SVC()

clf.fit(X_train, y_train)

pred_clf = clf.predict(X_test)
#let's check model's accuracy at prediction using the testing set we have (i.e. y_test)

print(classification_report(y_test, pred_clf))

print(confusion_matrix(y_test, pred_clf))
# another way to print out accuracy explicity (using scikit-learn)

accuracy_score(y_test, pred_clf)
# another way to print out accuracy explicity (using scikit-learn)

clf.score(X_test, y_test)
# Grid search Cross Validation : automatically find good values for the SVM classifier parameters by using tools such as grid search and cross-validation



# Set the parameter candidates

parameter_candidates = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

}



# Create a classifier with the parameter candidates

grid_svc = GridSearchCV(estimator=clf, param_grid=parameter_candidates, scoring='accuracy', cv=10)

# estimator = model we are using for hyperparameter tuning

# param_grid = list of parameters and the range of values for each parameter of the specified estimator

# cv: to determine the hyper-parameter value set (i.e. [C = , kernel = , gamma = ]) which provides the best accuracy level



# Train the classifier on training data

grid_svc.fit(X_train, y_train)
#Best parameters for our svc model

print(grid_svc.best_params_)



#OR



# Print out the results individually

print('Best score for training data:', grid_svc.best_score_)

print('Best `C`:',grid_svc.best_estimator_.C)

print('Best kernel:',grid_svc.best_estimator_.kernel)

print('Best `gamma`:',grid_svc.best_estimator_.gamma)
#Let's run our SVC again with the best parameters.

svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf') # kernel is a similarity function used to compute similarity between training data points

svc2.fit(X_train, y_train)

pred_svc2 = svc2.predict(X_test)

print(classification_report(y_test, pred_svc2))
print(svc2.score(X_test, y_test)) #OR

print(accuracy_score(pred_svc2, y_test))
mlpc = MLPClassifier(hidden_layer_sizes = (11,11,11), max_iter = 500) # hidden_layer_size is the number of nodes in each of the three layer. we chose 11 because we have 11 predictor features in our original data

mlpc.fit(X_train, y_train)

pred_mlpc = mlpc.predict(X_test)
# checking the accuracy of model

print(classification_report(y_test, pred_mlpc))

print(confusion_matrix(y_test, pred_mlpc))
# another way to print out accuracy explicity (using scikit-learn)

accuracy_score(y_test, pred_mlpc)
# Finally we want to see what happens when we feed in brand new data into the classifier and see what sort of predictions it churns out



# to do that we first create a vector of values i.e. give as input one row of data  (here we are giving 2 inputs so we will get 2 prediction outputs)

Xnew = [[7.3, 0.58, 0.00, 2.0, 0.065, 15.0, 21.0, 0.9946, 3.36, 0.47, 10.0], 

        [9.3, 0.58, 0.20, 3.0, 0.065, 16.0, 22.0, 0.9946, 5.96, 0.47, 12.0]]

Xnew = sc.transform(Xnew) # very imp step because the classifier was designed using scaled data and any input to the classifier must also be scaled that too using the same 'sc' that was fitted using the original training data

Ynew = mlpc.predict(Xnew)

Ynew # Based on the predictions, both the wine inputs are supposedly poor quality wines (i.e. quality = 0)
sgdc = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 500) #the concrete loss function is set via the loss parameter

sgdc.fit(X_train, y_train)

pred_sgdc = sgdc.predict(X_test)
#check accuracy

accuracy_score(pred_sgdc, y_test)
# to see the model parameters to be reported in journal papers

sgdc.coef_
# to see the intercepts

sgdc.intercept_
knn = KNeighborsClassifier(n_neighbors = 2) # to see rule of thumb for selecting n_neighbour parameter, see Section 5.1 below

knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)
#checking the accuracy

accuracy_score(pred_knn, y_test)
# printing the confusion matrix

print(confusion_matrix(pred_knn, y_test))
# printing the classification report

print(classification_report(pred_knn, y_test))
print(len(X_train))

import math

print(math.sqrt(len(X_train))) # this should be the value of k in knn algo
#lets try running the knn model with new k values and see if there is any improvement in accuracy

knn_opt = KNeighborsClassifier(n_neighbors = 35, p = 2, metric= "euclidean") # p=2 because outcome can only take two values: good (0) or bad (1)

knn_opt.fit(X_train, y_train)

pred_knn_opt = knn_opt.predict(X_test)

#checking the accuracy

accuracy_score(pred_knn_opt, y_test)
# printing the confusion matrix

print(confusion_matrix(pred_knn_opt, y_test))
# also imp to look at f1 score because that one will tell you false positives/negatives in addition to accuracy which tells us how many we got right and how many we got wrong (i.e. true positives and true negatives)

print(f1_score(pred_knn_opt, y_test)) # perfect precision and recall gives f1 score as 1 and at worst 0. 
knn = KNeighborsClassifier()

parameter_candidatess = {

    'n_neighbors': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

}

cross_validation = StratifiedKFold(n_splits=10, random_state = 45)



grid_knn = GridSearchCV(estimator = knn, param_grid = parameter_candidatess, scoring = 'accuracy', cv = cross_validation)

grid_knn.fit(X_train, y_train)

grid_knn.best_params_
knn2 = KNeighborsClassifier(n_neighbors = 60, p = 2, metric= "euclidean")

knn2.fit(X_train, y_train)

pred_knn2 = knn2.predict(X_test)
#check accuracy after parameter tuning

accuracy_score(pred_knn2, y_test)
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

pred_dtc = dtc.predict(X_test)
accuracy_score(pred_dtc, y_test)

# print(accuracy_score(pred_dtc, y_test))
# As of now decision tree classifier was created using default values, but now lets try to figure out the 'best' hyperparameters



parameter_candidates = {

    'criterion': ['gini', 'entropy'],

    'min_samples_leaf' :[1,2,3,4,5],

    'max_depth': [2,3,4,5,6,7,8,9,10]

}



grid_dtc = GridSearchCV(estimator = dtc, param_grid= parameter_candidates, scoring= 'accuracy', cv = 5)
grid_dtc.fit(X_train, y_train)

grid_dtc.best_params_
# Recreating the dtc model but with the new hyperparameters obtained

dtc2 = DecisionTreeClassifier(criterion= 'gini', max_depth = 2, min_samples_leaf= 1)

dtc2.fit(X_train, y_train)

pred_dtc2 = dtc2.predict(X_test)
#check accuracy of new model

accuracy_score(pred_dtc2, y_test) 
scores = [] # to store the accuracy obtained from each 'fold'

best_clf = SVC(gamma= "auto") # we had to set the gamma parameter because we were getting a warning saying in the next version, the default value will change from gamma to auto so it is better to explicitly define one

cv = KFold(n_splits=10, random_state=42, shuffle=False) #it is a 10-fold cv so n_splits = 10

for train_index, test_index in cv.split(X):

    # printing out the indexes of the training and the testing sets in each iteration 

    # to clearly see the process of K-Fold CV where the training and testing set changes in each iteration

    print("Train Index: ", train_index, "\n")

    print("Test Index: ", test_index)

    

    # setting the training and testing sets in each iteration,

    # followed by generating the model using the X_train and y_train datasets and

    # finally recording the accuracy for each model (after testing with the X_test and y_test datasets) in the 'scores' array

    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    best_clf.fit(X_train, y_train)

    scores.append(best_clf.score(X_test, y_test))
# We appended each score to a list called 'scores' and now, 

# we get the mean value in order to determine the overall accuracy of the model.

print(np.mean(scores))
# this will give you a list of r2 scores 

best_clf_eval =  cross_val_score(best_clf, X, y, cv=10)



# we will average all the scores to get a mean value from the 'new' and improved model

best_clf_eval.mean()
# this will give you a list of predictions using the 'new' and improved model

cross_val_predict(best_clf, X, y, cv=10) # output will be 0 or 1 depending on whether wine was good (0) or bad (1)