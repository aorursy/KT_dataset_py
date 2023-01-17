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
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
train_df.tail()
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms= ms[ms["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    fig=sns.barplot(ms.index, ms["Percent"],color="blue",alpha=0.8)

    plt.xlabel('Independent variables', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('NaN exploration', fontsize=15)

    return ms



missing_data(train_df)
missing_data(test_df)
test_df['Age'].fillna(test_df['Age'].median(),inplace = True)

train_df['Age'].fillna(train_df['Age'].median(),inplace = True)



drop_column=['Cabin']

train_df.drop(drop_column,axis=1,inplace=True)

test_df.drop(drop_column,axis=1,inplace=True)



test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)



train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)



print(train_df.isnull().sum())

print(test_df.isnull().sum())
all_data = [train_df,test_df]
import re

# extract the second word from every name and assign it to a new column



def title_parser(name):

    # Check wheter title exists, then return it, if not return ""

    title_search = re.search('([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""

# Create new column Title

for dataset in all_data:

    dataset['Title'] = dataset['Name'].apply(title_parser)

    

for dataset in all_data:

    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'irrelevant')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
g = sns.pairplot(data = train_df, hue ='Survived',palette ='seismic',size=1.2, diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))

g.set(xticklabels=[])
# create new feature FamilySize as a combination of SibSp and Parch

for dataset in all_data:

    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1

# create bin for age features

for dataset in all_data:

    dataset['Age_bin']= pd.cut(dataset['Age'],bins=[0,12,20,40,120],labels=['Children','Teenager','Adult','Elder'])
#create bin for fare features

for dataset in all_data:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'],bins=[0,7.91,14.45,31,120],labels=['Low_fare','median_fare','Average_fare','high_fare'])
train_df.head()
for dataset in all_data:

    drop_column =['Name','Ticket']

    dataset.drop(drop_column,axis=1,inplace=True)
train_df.head()
train_df = pd.get_dummies(train_df,columns=['Sex','Embarked','Title','Age_bin','Fare_bin'],prefix=['Sex','Embarked','Title','Age_bin','Fare_bin'])

test_df = pd.get_dummies(test_df,columns=['Sex','Embarked','Title','Age_bin','Fare_bin'],prefix=['Sex','Embarked','Title','Age_bin','Fare_bin'])

train_df.head()
sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
from sklearn.preprocessing import MinMaxScaler



train_df[['Age','Fare']] = train_df[['Age','Fare']].apply(pd.to_numeric)

scaler = MinMaxScaler()

train_df[['Age','Fare']] = scaler.fit_transform(train_df[['Age','Fare']])



drop_column =['PassengerId']

train_df.drop(drop_column,axis=1,inplace=True)



train_X = train_df.drop('Survived',axis=1)

train_Y = train_df['Survived']



test_X = test_df.drop('PassengerId',axis=1).copy()



train_df.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10,random_state=22)



all_features = train_df.drop('Survived',axis=1)

targeted_feature = train_df['Survived']



X_train,X_test,y_train,y_test = train_test_split(all_features, targeted_feature,test_size=0.3,random_state=42)
train_X = train_df.drop('Survived',axis=1)

train_Y = train_df['Survived']

test_X = test_df.drop('PassengerId',axis=1).copy()



import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier()

param_grid ={'loss':['deviance'],

            'n_estimators':[100,200,300,400],

            'learning_rate':[0.1,0.05,0.01,0.001],

            'max_depth':[4,8],

            'min_samples_leaf':[100,150],

             'max_features':[0.3,0.2,0.1]}



modelf = GridSearchCV(model,param_grid=param_grid,cv=kfold,scoring="accuracy",n_jobs=4,verbose=1)

modelf.fit(train_X,train_Y)



modelf.best_estimator_
modelf.best_score_
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

n_estim = range(100,1000,100)



param_grid ={'n_estimators':n_estim}



model_rf = GridSearchCV(model,param_grid = param_grid,cv=5,scoring="accuracy",n_jobs=4,verbose=1)



model_rf.fit(train_X,train_Y)



model_rf.best_estimator_
model_rf.best_score_
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()

param_grid = {'tol':[0.001,0.01,0.1,0.2]}



modell = GridSearchCV(model,param_grid = param_grid,cv=5,scoring="accuracy",n_jobs=4,verbose=1)



modell.fit(train_X, train_Y)



modell.best_estimator_
modell.best_score_
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



model = LogisticRegression()

model.fit(X_train,y_train)

prediction_lr = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_lr,y_test)*100,2))



result_lr = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_lr.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)

from sklearn.svm import SVC, LinearSVC



model = SVC()

model.fit(X_train,y_train)

prediction_svm = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_svm,y_test)*100,2))



result_svm = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_svm.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

prediction_knn = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_knn,y_test)*100,2))



result_knn = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_knn.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(X_train,y_train)

prediction_gnb = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_gnb,y_test)*100,2))



result_gnb = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_gnb.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(criterion='gini',

                              min_samples_split = 10,

                              min_samples_leaf = 1,

                              max_features ='auto')

model.fit(X_train,y_train)

prediction_tree = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_tree,y_test)*100,2))



result_tree = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_tree.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.ensemble import AdaBoostClassifier



model = AdaBoostClassifier()

model.fit(X_train,y_train)

prediction_adb = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_adb,y_test)*100,2))



result_adb = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_adb.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



model = LinearDiscriminantAnalysis()

model.fit(X_train,y_train)

prediction_lda = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_lda,y_test)*100,2))



result_lda = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_lda.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier()

model.fit(X_train,y_train)

prediction_gbc = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_gbc,y_test)*100,2))



result_gbc = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_gbc.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.ensemble import RandomForestClassifier



model_rf = RandomForestClassifier(criterion='gini',n_estimators=700,

                                 min_samples_split = 10, min_samples_leaf=1,

                                 max_features='auto',oob_score=True,

                                 random_state=1,n_jobs=-1)

model_rf.fit(X_train,y_train)

prediction_rm = model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_rm,y_test)*100,2))



result_rm = cross_val_score(model, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_rm.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
from sklearn.ensemble import RandomForestClassifier



model1 = RandomForestClassifier(bootstrap = True, class_weight = None,

                                max_depth = None, max_leaf_nodes = None,

                                min_weight_fraction_leaf = 0.0,

                                 criterion='gini',n_estimators=100,

                                 min_samples_split = 2, min_samples_leaf=1,

                                 max_features='auto',oob_score=False,

                                 random_state=None,n_jobs=None,

                               verbose=0, warm_start = False)

model1.fit(X_train,y_train)

prediction_rm1 = model1.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_rm1,y_test)*100,2))



result_rm1 = cross_val_score(model1, all_features, targeted_feature, cv=10,scoring='accuracy')

print('The cross validated score ',round(result_rm1.mean()*100,2))



y_pred = cross_val_predict(model,all_features, targeted_feature,cv=10)

sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix',y=1.05,size=15)
models = pd.DataFrame({

    'Model':["support vector machine","KNN","Logistic Regression",

            "Random Forest","Naive Bayes","AdaBoostClassifier",

            "Gradient Decent","Linear Discriminant Analysis",

            "Decision Tree","Tuned RF"],

    "Score":[result_svm.mean(), result_knn.mean(),result_lr.mean(),

            result_rm.mean(),result_gnb.mean(),result_adb.mean(),

            result_gbc.mean(),result_lda.mean(),result_tree.mean(),

            result_rm1.mean()]

})

models.sort_values(by="Score",ascending=False)
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(bootstrap = True, class_weight = None,

                                max_depth = None, max_leaf_nodes = None,

                                min_weight_fraction_leaf = 0.0,

                                 criterion='gini',n_estimators=100,

                                 min_samples_split = 2, min_samples_leaf=1,

                                 max_features='auto',oob_score=False,

                                 random_state=None,n_jobs=None,

                               verbose=0, warm_start = False)

random_forest.fit(train_X, train_Y)

Y_pred_rf = random_forest.predict(test_X)

random_forest.score(train_X,train_Y)

acc_random_forest = round(random_forest.score(train_X, train_Y)*100,2)

print(acc_random_forest)

print("Feature selection")

pd.Series(random_forest.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8)
submission = pd.DataFrame({

    "PassengerId":test_df["PassengerId"],

    "Survived":Y_pred_rf

})
submission.head()
submission.to_csv('submission.csv',index=False)