"""Data Preparation Library"""

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



"""Models Library"""

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix,accuracy_score



"""Model Evaluation"""

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



"""Other"""

import os

import warnings

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning

%matplotlib inline

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category = ConvergenceWarning)
# Importing dataset

dt = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")



# First 5 rows of data

dt.head()
dt = dt.drop(columns=["RowNumber","CustomerId","Surname"])

dt.head()
dt.describe()
pd.DataFrame(dt.isnull().sum(),columns=["Count"])
# Exited -- CreditScore

sns.violinplot( x=dt["Exited"], y=dt["CreditScore"], linewidth=5)

plt.title("Credit Score Distribution of Churn (Exited)")

plt.show()
# Exited -- Age

sns.violinplot( x=dt["Exited"], y=dt["Age"], linewidth=5)

plt.title("Age of Customers Distribution of Churn (Exited)")

plt.show()
# Exited -- Tenure

sns.violinplot( x=dt["Exited"], y=dt["Tenure"], linewidth=5)

plt.title("Tenure of Customers Distribution of Churn (Exited)")

plt.show()
# Exited -- Balance

sns.violinplot( x=dt["Exited"], y=dt["Balance"], linewidth=5)

plt.title("Balance of Customers Distribution of Churn (Exited)")

plt.show()
# Balance boxplot

dt[["Balance"]].boxplot();
# Exited -- NumOfProducts

sns.violinplot( x=dt["Exited"], y=dt["NumOfProducts"], linewidth=5)

plt.title("Number of Products of Customers Distribution of Churn (Exited)")

plt.show()
# Exited -- EstimatedSalary

sns.violinplot( x=dt["Exited"], y=dt["EstimatedSalary"], linewidth=5)

plt.title("Estimated Salary of Customers Distribution of Churn (Exited)")

plt.show()


# Correlation Matrix

correlationColumns = dt[["CreditScore","Age","Tenure"

    ,"Balance","NumOfProducts","EstimatedSalary"]]



sns.set()

corr = correlationColumns.corr()

ax = sns.heatmap(corr

                 ,center=0

                 ,annot=True

                 ,linewidths=.2

                 ,cmap="YlGnBu")

plt.show()
# Decomposition predictors and target

predictors = dt.iloc[:,0:10]

target = dt.iloc[:,10:]
try:

    predictors['isMale'] = predictors['Gender'].map({'Male':1, 'Female':0})

except:

    pass
try:

    # Geography one shot encoder

    predictors[['France', 'Germany', 'Spain']] = pd.get_dummies(predictors['Geography'])

    # Removal of unused columns.

    predictors = predictors.drop(columns=['Gender','Geography','Spain'])

except:

    pass
normalization = lambda x:(x-x.min()) / (x.max()-x.min())

transformColumns = predictors[["Balance","EstimatedSalary","CreditScore"]]

predictors[["Balance","EstimatedSalary","CreditScore"]] = normalization(transformColumns)



# All Predictors Columns

predictors.describe()
# Train and test splitting

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.25, random_state=0)

pd.DataFrame({"Train Row Count":[x_train.shape[0],y_train.shape[0]],

              "Test Row Count":[x_test.shape[0],y_test.shape[0]]},

             index=["X (Predictors)","Y (Target)"])
# Numpy excaptions handle

y_train = y_train.values.ravel()
# Decision Tree

dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_pred_dtc = dtc.predict(x_test)

dtc_acc = accuracy_score(y_test,y_pred_dtc)



# Logistic Regression

logr = LogisticRegression()

logr.fit(x_train,y_train)

y_pred_logr = logr.predict(x_test)

logr_acc = accuracy_score(y_test,y_pred_logr)



# Naive Bayes

gnb = GaussianNB()

gnb.fit(x_train,y_train)

y_pred_gnb = gnb.predict(x_test)

gnb_acc = accuracy_score(y_test,y_pred_gnb)



# K Neighbors Classifier

knn = KNeighborsClassifier( metric='minkowski')

knn.fit(x_train,y_train)

y_pred_knn = knn.predict(x_test)

knn_acc = accuracy_score(y_test,y_pred_knn)



# Random Forrest

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

y_pred_rfc = rfc.predict(x_test)

rfc_acc = accuracy_score(y_test,y_pred_rfc)



# Neural Network

nnc = MLPClassifier()

nnc.fit(x_train,y_train)

y_pred_nnc = nnc.predict(x_test)

nnc_acc = accuracy_score(y_test,y_pred_nnc)



# Xgboost Classifier

xgboast = xgb.XGBClassifier()

xgboast.fit(x_train, y_train)

xgboast = xgboast.score(x_test,y_test)



pd.DataFrame({"Algorithms":["Decision Tree","Logistic Regression","Naive Bayes","K Neighbors Classifier","Random Ferest","Neural Network","Xgboost Classifier"],

              "Scores":[dtc_acc,logr_acc,gnb_acc,knn_acc,rfc_acc,nnc_acc,xgboast]})

# Cross validation test

models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('RFC', RandomForestClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('xgboast', XGBClassifier()))



# evaluate each model in turning kfold results

results_boxplot = []

names = []

results_mean = []

results_std = []

p,t = predictors.values, target.values.ravel()

for name, model in models:

    cv_results = cross_val_score(model, p,t, cv=10)

    results_boxplot.append(cv_results)

    results_mean.append(cv_results.mean())

    results_std.append(cv_results.std())

    names.append(name)

pd.DataFrame({"Algorithm":names,

                                "Accuracy Mean":results_mean,

                                "Accuracy":results_std})
# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results_boxplot)

ax.set_xticklabels(names)

plt.show()
#Grid Seach for XGboast

params = {

        'min_child_weight': [1, 2, 3],

        'gamma': [1.9, 2, 2.1, 2.2],

        'subsample': [0.4,0.5,0.6],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3,4,5]

        }

gd_sr = GridSearchCV(estimator=XGBClassifier(),

                     param_grid=params,

                     scoring='accuracy',

                     cv=5,

                     )

gd_sr.fit(predictors, target.values.ravel())

best_parameters = gd_sr.best_params_

pd.DataFrame(best_parameters.values(),best_parameters.keys(),columns=["Best Parameters"])
print("Best score is: ",gd_sr.best_score_)
# If you saved model, you can use Pickle file.

# Pickle cound use "import pickle"