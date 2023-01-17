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
#Imports

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn import decomposition

from sklearn.feature_selection import RFECV

from sklearn.feature_selection import SelectKBest, f_classif

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix
df = pd.read_csv("../input/pimaindians-datasetmodified/diabetes.csv")

df.head()
X = df.drop('class',axis=1)

Y = df[['class']]
X.head()
Y.head()
#correlation matrix

corr = df.corr()

corr
sns.heatmap(corr, annot = True)
X.hist(bins=50, figsize=(20, 15))

plt.show()
# Calculating the median value of Mass

median_mass = X['mass'].median()

# Replacing in Mass column where values equal to 0

X['mass'] = X['mass'].replace(

    to_replace=0, value=median_mass)
# Calculating the median value of Pres

median_pres = X['pres'].median()

# Replacing in Pres column where values equal to 0

X['pres'] = X['pres'].replace(

    to_replace=0, value=median_pres)
# Calculating the median value of Plas

median_plas = X['plas'].median()

# Replacing in Plas column where values equal to 0

X['plas'] = X['plas'].replace(

    to_replace=0, value=median_plas)
# Calculating the median value of Skin

median_Skin = X['Skin'].median()

# Replacing in Skin column where values equal to 0

X['Skin'] = X['Skin'].replace(

    to_replace=0, value=median_Skin)
# Calculating the median value of Preg

median_preg = X['preg'].median()

# Replacing in Skin column where values equal to 0

X['preg'] = X['preg'].replace(

    to_replace=0, value=median_preg)
# Calculating the median value of test

median_test = X['test'].median()

# Replacing in Skin column where values equal to 0

X['test'] = X['test'].replace(

    to_replace=0, value=median_test)
X.head()
X.hist(bins=50, figsize=(20, 15))

plt.show()
X.info()
#Transforming into an array of values

X_values = X.values

Y_values = Y.values.reshape(-1)
X_train,X_test,y_train,y_test = train_test_split(X_values,Y_values,test_size = 0.33)
#Using standardscaler

sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
X_train_std
#Using cross_val_score

def fit_model(name,model,X_train,Y_train):

    

    scores = cross_val_score(model,X_train,Y_train,scoring='accuracy')

    hit_rate = np.mean(scores)



    msg = "hit rate of {} is: {}".format(name,hit_rate)

    print(msg)

    return hit_rate
decisionTree_model = DecisionTreeClassifier(random_state=0)

decisionTree_result = fit_model("DecisionTree",decisionTree_model,X_train_std,y_train)
#RandomForest training

randomForest_model = RandomForestClassifier(random_state=0)

randomForest_result = fit_model("RandomForest",randomForest_model,X_train_std,y_train)
#OnevsRest Training

OnevsRest_model = OneVsRestClassifier(LinearSVC(random_state=0))

OnevsRest_result = fit_model("OnevsRest",OnevsRest_model,X_train_std,y_train)
#ONEVSONE training

OnevsOne_model = OneVsOneClassifier(LinearSVC(random_state=0))

OnevsOne_result = fit_model("OnevsOne",OnevsOne_model,X_train_std,y_train)
#KNN training

Knn_model = KNeighborsClassifier(n_neighbors=2)

Knn_result = fit_model("Knn",Knn_model,X_train_std,y_train)
# AdaBOOST training

AdaBoost_model = AdaBoostClassifier()

AdaBoost_result = fit_model("AdaBoost",AdaBoost_model,

                                X_train_std,y_train)
results = pd.DataFrame({

    'Model': ['DecisionTree', 'RandomForest', 'OnevsRest', 

              'ONEVSONE', 'KNN','AdaBOOST'],

    'Score': [decisionTree_result, randomForest_result, OnevsRest_result, 

              OnevsOne_result, Knn_result, AdaBoost_result]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df
SVC_model = SVC()

LinearSVC_model = LinearSVC()
#Using gridSearchCV for the OnevsRest Model

OnevsRest_parameters = [

    {'estimator':[decisionTree_model,randomForest_model,Knn_model,SVC_model,LinearSVC_model],'n_jobs':[1,-1]}

    ]
clf = GridSearchCV(estimator=OneVsRestClassifier(SVC()), param_grid=OnevsRest_parameters, n_jobs=-1)
clf.fit(X_train_std,y_train)
print('Best score for data1:', clf.best_score_)

print('Best estimators:',clf.best_estimator_.estimator)

print('Best n_jobs:',clf.best_estimator_.n_jobs)

clf.score(X_test_std, y_test)
result_GRIDOneVsRest = OneVsRestClassifier(LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,

          intercept_scaling=1, loss='squared_hinge', max_iter=1000,

          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,

          verbose=0),n_jobs=1).fit(X_train_std,y_train).score(X_test_std, y_test)
result_GRIDOneVsRest
model = OneVsRestClassifier(LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,

          intercept_scaling=1, loss='squared_hinge', max_iter=1000,

          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,

          verbose=0),n_jobs=1)



model.fit(X_train_std,y_train)

#ROC curve for OnevsRest model

probs = model.predict(X_test_std)

fpr, tpr, threshold = metrics.roc_curve(y_test, probs)

roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Confusion Matrix

confusion_matrix(y_test,probs)
y_test.shape
from sklearn.metrics import f1_score

f1_score(y_test,probs)