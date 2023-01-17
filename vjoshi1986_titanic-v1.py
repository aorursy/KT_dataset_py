# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
#Load Titanic data

def load_titanic_data(filename):
    return pd.read_csv(filename)
train_data = load_titanic_data("../input/train.csv")
test_data = load_titanic_data("../input/test.csv")
train_data.head()
#data info

train_data.info()
#Embarked

train_data["Embarked"].value_counts()
#Replace NaN in Embarked by category S

train_data["Embarked"].fillna('S', inplace=True)

#Get Median Age and replace NaN with Median Age

age_median = train_data["Age"].median()
age_median

train_data["Age"].fillna(age_median, inplace=True)
#Let's describe the data now.

train_data.describe()
#A quick look at categorical features.

#Pclass

train_data["Pclass"].value_counts()
#Let's see how is the distribution for survived passengers.

train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Sex

train_data["Sex"].value_counts()

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Embarked

train_data["Embarked"].value_counts()
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data.head() 
#Drop columns Cabin, Name, PassengerId, Ticket

train_data.drop(["PassengerId","Name", "Ticket", "Cabin"], axis =1, inplace=True)
train_data.head()
from sklearn.preprocessing import LabelEncoder

le_pClass = LabelEncoder()
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
train_data['PClass_encoded'] = le_pClass.fit_transform(train_data.Pclass)
train_data['Sex_encoded'] = le_sex.fit_transform(train_data.Sex)
train_data['Embarked_encoded'] = le_embarked.fit_transform(train_data.Embarked)

train_data.head()
#One hot encoding for categorical columns (PClass, Sex, Embarked)

from sklearn.preprocessing import OneHotEncoder

pClass_ohe = OneHotEncoder()
sex_ohe = OneHotEncoder()
embarked_ohe = OneHotEncoder()

Xp =pClass_ohe.fit_transform(train_data.PClass_encoded.values.reshape(-1,1)).toarray()
Xs =sex_ohe.fit_transform(train_data.Sex_encoded.values.reshape(-1,1)).toarray()
Xe =embarked_ohe.fit_transform(train_data.Embarked_encoded.values.reshape(-1,1)).toarray()

#Add back to original dataframe

train_dataOneHot = pd.DataFrame(Xp, columns = ["PClass_"+str(int(i)) for i in range(Xp.shape[1])])
train_data = pd.concat([train_data, train_dataOneHot], axis=1)

train_dataOneHot = pd.DataFrame(Xs, columns = ["Sex_"+str(int(i)) for i in range(Xs.shape[1])])
train_data = pd.concat([train_data, train_dataOneHot], axis=1)

train_dataOneHot = pd.DataFrame(Xe, columns = ["Embarked_"+str(int(i)) for i in range(Xe.shape[1])])
train_data = pd.concat([train_data, train_dataOneHot], axis=1)
train_data.head()
#Drop unneccesary columns

train_data.drop(["Pclass","Sex", "Embarked", "PClass_encoded", "Sex_encoded", "Embarked_encoded"], axis =1, inplace=True)
train_data.head()
train_data.shape
train_data.info()
#Feature Matrix

X = train_data.drop(['Survived'], axis=1)
X.shape
#Target Vector

y = train_data['Survived']
y.shape
#Split into train test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state =42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#SGDClassifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state =42)
sgd_clf.fit(X_train, y_train)
#SupportVectorMachine

from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
fr_clf = RandomForestClassifier(random_state=42)
fr_clf.fit(X_train, y_train)
#LogisticRegression

from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X_train, y_train)
#KNearestNeighbors

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train,y_train)
#DecisionTree

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
#Measuring accuracy Using Cross Validation

from sklearn.model_selection import cross_val_score

#SGDClassifier
sgd_clf_score = cross_val_score(sgd_clf, X_train, y_train, cv=10, scoring="accuracy")
sgd_mean = sgd_clf_score.mean()
sgd_mean
#SVMClassifier
svm_clf_score = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_mean = svm_clf_score.mean()
svm_mean
#RandomForestClassifier
rf_clf_score = cross_val_score(fr_clf, X_train, y_train, cv=10)
rf_mean = rf_clf_score.mean()
rf_mean
#LogisticRegression
lr_clf_score = cross_val_score(lr_clf, X_train, y_train, cv=10)
lr_mean = lr_clf_score.mean()
lr_mean
#KNearestNeighbors
knn_clf_score = cross_val_score(knn_clf, X_train, y_train, cv=10)
knn_mean = knn_clf_score.mean()
knn_mean
#DecisionTreeClassifier
dt_clf_score = cross_val_score(dt_clf, X_train, y_train, cv=10)
dt_mean = dt_clf_score.mean()
dt_mean
#Model Evaluations

models = pd.DataFrame({'Model': ['SGDClassifier', 'SupportVectorMachine', 'RandomForestClassifier',
                                'LogisticRegression', 'KNearestNeighbors', 'DecisionTreeClassifier'], 
                       'Score':[sgd_mean, svm_mean, rf_mean, lr_mean, knn_mean, dt_mean
                    ]})
models.sort_values(by = 'Score', ascending = False)
test_data.head()
test_data.info()
#We need to do the same transformation on test data file, so we can predict.

#Replace NaN in Embarked by category S
test_data["Embarked"].fillna('S', inplace=True)


#Get Median Age and replace NaN with Median Age
age_median = test_data["Age"].median()
test_data["Age"].fillna(age_median, inplace=True)

#Drop columns Cabin, Name, Ticket
test_data.drop(["Name", "Ticket", "Cabin"], axis =1, inplace=True)
#OneHotEncoding

le_pClass = LabelEncoder()
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
test_data['PClass_encoded'] = le_pClass.fit_transform(test_data.Pclass)
test_data['Sex_encoded'] = le_sex.fit_transform(test_data.Sex)
test_data['Embarked_encoded'] = le_embarked.fit_transform(test_data.Embarked)

test_data.head()
pClass_ohe = OneHotEncoder()
sex_ohe = OneHotEncoder()
embarked_ohe = OneHotEncoder()

Xp =pClass_ohe.fit_transform(test_data.PClass_encoded.values.reshape(-1,1)).toarray()
Xs =sex_ohe.fit_transform(test_data.Sex_encoded.values.reshape(-1,1)).toarray()
Xe =embarked_ohe.fit_transform(test_data.Embarked_encoded.values.reshape(-1,1)).toarray()

test_data.head()
#Add back to original dataframe

test_dataOneHot = pd.DataFrame(Xp, columns = ["PClass_"+str(int(i)) for i in range(Xp.shape[1])])
test_data = pd.concat([test_data, test_dataOneHot], axis=1)

test_dataOneHot = pd.DataFrame(Xs, columns = ["Sex_"+str(int(i)) for i in range(Xs.shape[1])])
test_data = pd.concat([test_data, test_dataOneHot], axis=1)

test_dataOneHot = pd.DataFrame(Xe, columns = ["Embarked_"+str(int(i)) for i in range(Xe.shape[1])])
test_data = pd.concat([test_data, test_dataOneHot], axis=1)

test_data.head()
#Drop unneccesary columns

test_data.drop(["Pclass","Sex", "Embarked", "PClass_encoded", "Sex_encoded", "Embarked_encoded"], axis =1, inplace=True)
test_data.head()
#Predictions
test_data_pred = test_data.drop(["PassengerId"], axis =1)
test_data_pred.info()
#FARE is missing one entry , so will impute that by a median value.

fare_median = test_data_pred["Fare"].median()
fare_median

test_data_pred["Fare"].fillna(fare_median, inplace=True)
test_data_pred.info()
#Predict using LogisticRegression model

Y_pred = lr_clf.predict(test_data_pred)
#Make submission file version 1

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('Titanic_Prediction_v1.csv', index=False)
