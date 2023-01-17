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
# plottting lib

import seaborn as sns

import matplotlib.pyplot as plt



import xgboost

from xgboost import XGBClassifier

### pre-processing lib

from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

from sklearn. model_selection import train_test_split,GridSearchCV,KFold,cross_val_predict,RandomizedSearchCV

### classification lib required

from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier,StackingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import KernelPCA,PCA

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.svm import SVC

## different metrices

from sklearn.metrics import accuracy_score,r2_score
data=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data
data.describe()
corelation=data.corr()

plt.figure(figsize=(14,12))

sns.heatmap(corelation,annot=True)
corelation['quality'].sort_values()
corelation.columns
selected_features=['volatile acidity', 'citric acid','sulphates', 'alcohol','quality']
feat_data=data[selected_features]

pd.plotting.scatter_matrix(feat_data,alpha=0.1,figsize=(10,10))

plt.title('Scatter Matrix plot of selected features.')
plt.figure(figsize=(10,10))

pd.plotting.radviz(feat_data,'quality')
#labels=feat_data.pop('quality')

#X_train,X_test,y_train,y_test=train_test_split(feat_data,labels,test_size=0.2,random_state=42)
labels=feat_data.pop('quality')

X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)
#X_train,y_train
svm=SVC(gamma='scale', probability=True)

svm.fit(X_train,y_train)



svm_pred=svm.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,svm_pred)

print("SVM accuracy is :{}".format(score))
random_f=RandomForestClassifier(n_estimators=250)

random_f.fit(X_train,y_train)

random_f_pred=random_f.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,random_f_pred)

print("random forest accuracy is :{}".format(score))
log=LogisticRegression(solver='liblinear')

log.fit(X_train,y_train)

pred=log.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("LogisticRegression accuracy is :{}".format(score))
Decision=DecisionTreeClassifier()

Decision.fit(X_train,y_train)

pred=Decision.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("DecisionTreeClassifier accuracy is :{}".format(score))
guassian=GaussianNB()

guassian.fit(X_train,y_train)

pred=guassian.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("GaussianNB accuracy is :{}".format(score))
KNN=KNeighborsClassifier()

KNN.fit(X_train,y_train)

pred=KNN.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("KNeighborsClassifier accuracy is :{}".format(score))
Ada=AdaBoostClassifier()

Ada.fit(X_train,y_train)

pred=Ada.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("AdaBoostClassifier accuracy is :{}".format(score))
Bagging=BaggingClassifier(n_estimators=300)

Bagging.fit(X_train,y_train)

pred=Bagging.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("BaggingClassifier accuracy is :{}".format(score))
Ex_Tree=ExtraTreesClassifier(n_estimators=300)

Ex_Tree.fit(X_train,y_train)

pred=Ex_Tree.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("ExtraTreesClassifier accuracy is :{}".format(score))
XGB=XGBClassifier()

XGB.fit(X_train,y_train)

pred=XGB.predict(X_test)

print("*"* 30)

score=accuracy_score(y_test,pred)

print("XGBClassifier accuracy is :{}".format(score))
def get_models():

	models = list()

	models.append(LogisticRegression(solver='liblinear'))

	models.append(DecisionTreeClassifier())

	models.append(SVC(gamma='scale', probability=True))

	models.append(GaussianNB())

	models.append(KNeighborsClassifier())

	models.append(AdaBoostClassifier())

	models.append(BaggingClassifier(n_estimators=10))

	models.append(RandomForestClassifier(n_estimators=10))

	models.append(ExtraTreesClassifier(n_estimators=10))

	models.append(XGBClassifier())

	return models
!pip install mlens
import mlens

from mlens.ensemble import SuperLearner

def get_super_learner(X):

	ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X))

	# add base models

	models = get_models()

	ensemble.add(models)

	# add the meta model

	ensemble.add_meta(LogisticRegression(solver='lbfgs'))

	return ensemble
ensemble = get_super_learner(X_train)
# fit the super learner

ensemble.fit(X_train.values,y_train.values)

# summarize base learners

print(ensemble.data)



# make predictions on hold out set

## here i face error with an pandas dataframe input hence i convert it into numpy array 

##may be mlens lib still not support direct pipeline of pandas dataframe

## may be they will fix this issue further :)



yhat = ensemble.predict(X_test.values)



print("*"* 30)

score=accuracy_score(y_test,yhat)

print("Super Learner accuracy is :{}".format(score))
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, yhat))