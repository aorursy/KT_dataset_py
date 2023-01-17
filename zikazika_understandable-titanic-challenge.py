#Necessary packages

import numpy as np #linear Algebra

import pandas as pd #data manipulation

import seaborn as sns #multiple plots

import matplotlib.pyplot as plt #plotting

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline 



train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")
train_df.head()

train_df.tail()

def missingdata(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms= ms[ms["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='80')

    fig=sns.barplot(ms.index, ms["Percent"],color="red",alpha=0.8)

    plt.xlabel('Independent variables', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('NaN exploration', fontsize=15)

    return ms



missingdata(train_df)
missingdata(test_df)
fig=sns.scatterplot(train_df.index, train_df["Age"],color="blue",alpha=0.8)

plt.xlabel('Occurances', fontsize=15)

plt.ylabel('Values of Age', fontsize=15)

plt.title('Variable exporation', fontsize=15)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)





#For example when trying to exclude the cabin column we are going to receive unnecessary warnings. Exclude! (thats why we imported it)

drop_column = ['Cabin']

train_df.drop(drop_column, axis=1, inplace = True)

test_df.drop(drop_column,axis=1,inplace=True)



test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)



train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
print(train_df.isnull().sum())

print(test_df.isnull().sum())
# in order to apply the function only once

all_data=[train_df,test_df]
import re

# A way to think about textual preprocessing is: Given my character column, what are some regularities that occur often. In our case we see titles (miss,mr etc).

# Let us then extract second word from every row and assign it to a new column. Not only that let us make it categorical (so that we can one-hot encode it) where we observe the most frequent ones.





def title_parser(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # Check whether title exists, then return it, if not ""

    if title_search:

        return title_search.group(1)

    return ""

# New column named Title, containing the titles of passenger names

for dataset in all_data:

    dataset['Title'] = dataset['Name'].apply(title_parser)

# Irrelevant titles should be just called irrelevant (in sence that they do not occur often)

for dataset in all_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'irrelevant')

# Let us make sure they are categorical, where we replace similiar names

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

g = sns.pairplot(data=train_df, hue='Survived', palette = 'seismic',

                 size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])



#as a general rule with classification problems, we will always put the target variable (to be classified) in hue

all_data=[train_df,test_df]



# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in all_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    ## create bin for age features

for dataset in all_data:

    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])

    

    

  

# create bin for fare features

for dataset in all_data:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',

                                                                                      'Average_fare','high_fare'])
traindf=train_df

testdf=test_df



all_dat=[traindf,testdf]



for dataset in all_dat:

    drop_column = ['Name','Ticket']

    dataset.drop(drop_column, axis=1, inplace = True)

    

   
train_df.head()
traindf = pd.get_dummies(traindf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],

                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])



testdf = pd.get_dummies(testdf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],

                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])



traindf.head()

testdf.head()
sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
type(traindf["Age"])
from sklearn.preprocessing import MinMaxScaler





traindf[['Age','Fare']] = traindf[['Age',"Fare"]].apply(pd.to_numeric)

scaler = MinMaxScaler()

traindf[['Age','Fare']] = scaler.fit_transform(traindf[['Age',"Fare"]])



drop_column = ['PassengerId']#id of any kind will be always dropped since it does not have predictive power

traindf.drop(drop_column, axis=1, inplace = True)

train_X = traindf.drop("Survived", axis=1)#we do not need train test splitting with skicit learn (in nomral setting test_df and train_df will be concatanted and then use it)

train_Y=traindf["Survived"]

test_X  = testdf.drop("PassengerId", axis=1).copy() 

train_X.shape, train_Y.shape, test_X.shape





traindf.head()
from sklearn.model_selection import train_test_split #split the dat in test and train sets

from sklearn.model_selection import cross_val_score #score evaluation with cross validation

from sklearn.model_selection import cross_val_predict #prediction with cross validation

from sklearn.metrics import confusion_matrix #for confusion matrix (metric of succes)

from sklearn.model_selection import KFold #for K-fold cross validation

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts





all_features = traindf.drop("Survived",axis=1) #all of the independent variables are necessary for the cross_val function

Targeted_feature = traindf["Survived"]



X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)#why do we have to do it cant we just use test_df ? NO, since we do not have the predictions (that si checked internally in Kaggle) we can not have accuracy on hold-out test)
train_X = traindf.drop("Survived", axis=1)

train_Y=traindf["Survived"]

test_X  = testdf.drop("PassengerId", axis=1).copy()

train_X.shape, train_Y.shape, test_X.shape





import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier()

param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300,400],

              'learning_rate': [0.1, 0.05, 0.01,0.001],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.2,0.1] 

              }



modelf = GridSearchCV(model,param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



modelf.fit(train_X,train_Y)





# Best Estimator

modelf.best_estimator_
modelf.best_score_ #accuracy metric is straightforeward, how much did I predict corrrectly?
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

n_estim=range(100,1000,100)



#This is the grid

param_grid = {"n_estimators" :n_estim}





model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_rf.fit(train_X,train_Y)





#best estimator

model_rf.best_estimator_
model_rf.best_score_
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model =LinearDiscriminantAnalysis()

param_grid = {'tol':[0.001,0.01,.1,.2]}



modell = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



modell.fit(train_X,train_Y)



# Best Estimator

modell.best_estimator_
modell.best_score_ 


from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn.metrics import accuracy_score  #for accuracy_score





model = LogisticRegression()

model.fit(X_train,y_train)

prediction_lr=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_lr,y_test)*100,2))

result_lr=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_lr.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.svm import SVC, LinearSVC



model = SVC()

model.fit(X_train,y_train)

prediction_svm=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_svm,y_test)*100,2))

result_svm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_svm.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.neighbors import KNeighborsClassifier





model = KNeighborsClassifier(n_neighbors = 5)

model.fit(X_train,y_train)

prediction_knn=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_knn,y_test)*100,2))

result_knn=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_knn.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

model.fit(X_train,y_train)

prediction_gnb=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_gnb,y_test)*100,2))

result_gnb=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_gnb.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(criterion='gini', 

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto')

model.fit(X_train,y_train)

prediction_tree=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_tree,y_test)*100,2))

result_tree=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_tree.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.ensemble import AdaBoostClassifier

model= AdaBoostClassifier()

model.fit(X_train,y_train)

prediction_adb=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_adb,y_test)*100,2))

result_adb=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_adb.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model= LinearDiscriminantAnalysis()

model.fit(X_train,y_train)

prediction_lda=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_lda,y_test)*100,2))

result_lda=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_lda.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.ensemble import GradientBoostingClassifier

model= GradientBoostingClassifier()

model.fit(X_train,y_train)

prediction_gbc=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_gbc,y_test)*100,2))

result_gbc=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_gbc.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(criterion='gini', n_estimators=700,

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto',oob_score=True,

                             random_state=1,n_jobs=-1)

model_rf.fit(X_train,y_train)

prediction_rm=model.predict(X_test)





print('Accuracy',round(accuracy_score(prediction_rm,y_test)*100,2))

result_rm=cross_val_score(model_rf,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_rm.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

model1.fit(X_train,y_train)



prediction_rm1=model1.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_rm1,y_test)*100,2))

result_rm1=cross_val_score(model1,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score',round(result_rm.mean()*100,2))

y_pred = cross_val_predict(model1,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
models = pd.DataFrame({

    'Model': ["support vector machine",'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'AdaBoostClassifier', 

              'Gradient Decent', 'Linear Discriminant Analysis', 

              'Decision Tree',"Tuned RF"],

    'Score': [result_svm.mean(),result_knn.mean(), result_lr.mean(), 

              result_rm.mean(), result_gnb.mean(), result_adb.mean(), 

              result_gbc.mean(), result_lda.mean(), result_tree.mean(),result_rm1.mean()]})

models.sort_values(by='Score',ascending=False) #pd.DAtaFrame() is a function that takes a dictionary as an input. Within this list we determine key-values paires (column name-values within column)

# Random Forests

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

random_forest.fit(train_X, train_Y)

Y_pred_rf = random_forest.predict(test_X)

random_forest.score(train_X,train_Y)

acc_random_forest = round(random_forest.score(train_X, train_Y) * 100, 2)

print(acc_random_forest)




print("Feature selection")

pd.Series(random_forest.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8) #in a series x (theirs relative importance) and y values are taken





submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_rf})
submission.head()
submission.to_csv('submission.csv', index=False)