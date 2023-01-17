import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_selection import RFE,RFECV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

import os

from sklearn.linear_model import LogisticRegression

import seaborn as sns

from sklearn import preprocessing

import matplotlib.pyplot as plt 

from sklearn import svm

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
train_ds = pd.read_csv("../input/train.csv")
from sklearn.impute import SimpleImputer

def preprocessData(df):

    df.drop(['Name','PassengerId'], axis=1,inplace=True)

    

    encoder = preprocessing.LabelEncoder()

    df.Sex = encoder.fit_transform(df.Sex)

    df.drop(['Ticket'], axis=1,inplace=True)

    df.drop(['Fare'], axis=1,inplace=True)

    df.Embarked.fillna(df.Embarked.mode()[0])

    df = pd.get_dummies(df,prefix='embarked',columns=['Embarked'])

    df = pd.get_dummies(df,prefix='Pclass',columns=['Pclass'])



    df['mapped_cabin'] = [str(cabin)[0] for cabin in df.Cabin]

    df.drop(columns=['Cabin'],axis=1,inplace=True)

    cabin_dict = {value:cabin for cabin, value in enumerate(df.mapped_cabin.unique(),0)}

    print(cabin_dict)

    df['mapped_cabin'] = df['mapped_cabin'].map(cabin_dict)





    #imputer = SimpleImputer(missing_values='nan', strategy='mean')

    #imputer.fit_transform(train_ds)

    df.Age.fillna(np.mean(df.Age), inplace=True)

    return df

train_ds.info()

%matplotlib inline

train_ds.isnull().count()

train_ds = preprocessData(train_ds)

#train_ds.drop(['Name','PassengerId'], axis=1,inplace=True)
train_ds.info()

train_ds.head()

train_ds.Parch.unique()

train_ds.groupby(['Parch']).count()
#corr heatmap

sns.heatmap(train_ds.corr())
Y=train_ds['Survived']

train_ds_X = train_ds.drop(columns=['Survived'])

X=train_ds_X
#Logistic Regression with PCA



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, roc_auc_score



pca = PCA(n_components=2)



new_ds = pd.DataFrame(pca.fit(train_ds_X).transform(train_ds_X))

#new_ds.info()

new_ds.head()

#plt.plot(pca.explained_variance_ratio_)

#print(pca.components_)

#scaler = StandardScaler()

#scaler.fit(X)

#X.head()



X_train, X_test, y_train, y_test = train_test_split(new_ds, Y, test_size=0.33, random_state=42)

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

score = accuracy_score(y_test,y_pred)

auc = roc_auc_score(y_test,y_pred)

print("accuracy score without cv {}".format(score))

print("roc score without cv {}".format(score))



# cross validation

from sklearn.model_selection import cross_val_score

cross_val = cross_val_score(X=X,y=Y,estimator=lr,scoring='accuracy',cv=10,verbose=0)

print("score after cross validation {}".format(cross_val.mean()))

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

score = accuracy_score(y_test,y_pred)

auc = roc_auc_score(y_test,y_pred)

print("accuracy score without cv {}".format(score))

print("roc score without cv {}".format(auc))



# cross validation

from sklearn.model_selection import cross_val_score

cross_val = cross_val_score(X=X,y=Y,estimator=dt,scoring='accuracy',cv=10,verbose=0)

print("score after cross validation {}".format(cross_val.mean()))



#--Adaboost Classifier--------#

ab = AdaBoostClassifier(n_estimators=300)

ab.fit(X_train,y_train)

y_pred = ab.predict(X_test)

score = accuracy_score(y_test,y_pred)

auc = roc_auc_score(y_test,y_pred)

print("accuracy score without cv {} for Adaboost".format(score))

print("roc score without cv {} for Adaboost".format(auc))

cross_val_adaboost = cross_val_score(X=X,y=Y,estimator=dt,scoring='accuracy',cv=10,verbose=0)

print("score after cross validation {} for Adaboost".format(cross_val_adaboost.mean()))
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

#Feature Scaling is must for GridSearchCV



scaler = StandardScaler()

train_ds_X = pd.DataFrame(scaler.fit(train_ds_X).transform(train_ds_X), columns=train_ds_X.columns)

#train_ds_X

X_train, X_test, y_train, y_test = train_test_split(train_ds_X, Y, test_size=0.33, random_state=42)

vector = SVC()

parameter_grid = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.1,0.01,0.001, 0.0001], 'kernel': ['rbf']},

 ]

grid_search = GridSearchCV(estimator=vector, param_grid = parameter_grid,cv=2)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_ )

print(grid_search.best_score_)



#-----Second Run-------#

parameter_grid = [

    {'C': [98, 99, 100, 101,102], 'gamma': [0.005,0.0025,0.01,0.02, 0.03], 'kernel': ['rbf']},

 ]

grid_search = GridSearchCV(estimator=vector, param_grid = parameter_grid,cv=2)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_ )

print(grid_search.best_score_)



#-----Third Run-------#

parameter_grid = [

    {'C': [102,103,104,105,106,107,108,109,110], 'gamma': [0.01], 'kernel': ['rbf']},

 ]

grid_search = GridSearchCV(estimator=vector, param_grid = parameter_grid,cv=2)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_ )

print(grid_search.best_score_)

# we get best parameters from third run 

vector = SVC(C=102, gamma=0.01, kernel='rbf')

vector.fit(X_train,y_train)

y_pred = vector.predict(X_test)

score = accuracy_score(y_test,y_pred)

auc = roc_auc_score(y_test,y_pred)

print("accuracy score without cv {} for SVM".format(score))

print("roc score without cv {} for SVM".format(auc))

cross_val_adaboost = cross_val_score(X=X,y=Y,estimator=dt,scoring='accuracy',cv=10,verbose=0)

print("score after cross validation {} for SVM".format(cross_val_adaboost.mean()))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10)

rfc.fit(X_train,y_train)

predicted = rfc.predict(X_test)

score = accuracy_score(y_test,predicted)

auc = roc_auc_score(y_test,predicted)

print("accuracy score without cv {} for RFC".format(score))

print("roc score without cv {} for RFC".format(auc))

cross_val_adaboost = cross_val_score(X=X,y=Y,estimator=dt,scoring='accuracy',cv=10,verbose=0)

print("score after cross validation {} for RFC".format(cross_val_adaboost.mean()))
#make predictions on test set now

test_ds = pd.read_csv("../input/test.csv")

test_ds.info()
result_df = pd.DataFrame(test_ds['PassengerId'])

test_df = preprocessData(test_ds)

#test_df.info()

#test_df.head()

#result_df.info()
#Logistic Regression



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

score = accuracy_score(y_test,y_pred)

auc = roc_auc_score(y_test,y_pred)

print("accuracy score without cv {}".format(score))

print("roc score without cv {}".format(score))



# cross validation

from sklearn.model_selection import cross_val_score

cross_val = cross_val_score(X=X,y=Y,estimator=lr,scoring='accuracy',cv=10,verbose=0)

print("score after cross validation {}".format(cross_val.mean()))



y_prediction_for_test = lr.predict(test_df)

result_df['Survived'] = y_prediction_for_test

result_df.set_index(keys=['PassengerId'],inplace=True)

result_df.info()

result_df.head()
result_df.to_csv("../submission.csv")