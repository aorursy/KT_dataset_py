'''

author:yxf

'''
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import LabelEncoder



import xgboost as xgb



train_df = pd.read_csv("../input/train.csv", dtype={"Age":np.float64})

test_df = pd.read_csv("../input/test.csv", dtype={"Age":np.float64})







#train_df.head()

#train_df.describe()



#train_df.head()

#train_df.tail()

#train_df.shape()

#train_df.isnull.sum()

#train_df.Pclass.value_counts()

#train_df.groupby(["Survived","Pclass"]).size()







from sklearn import metrics



def harmonize_data(titanic):

    

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic["Age"].median()

    

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 1

    titanic.loc[titanic["Sex"] == "female", "Sex"] = 2

    

    titanic["Embarked"] = titanic["Embarked"].fillna("S")



    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 1

    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 2    

    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 3



    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())



    return titanic

    

train_df = harmonize_data(train_df)

test_df = harmonize_data(test_df)

    

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

target = ["Survived"]



#X_train, X_test, y_train, y_test = train_test_split(train_df[predictors], train_df[target], 

#                                                    test_size=0.2, random_state=42)

#train = train_df[predictors]

#target = train_df[target]

#test = test_df[predictors]

X_train = train_df[predictors]

y_train = train_df[target]

X_test = test_df[predictors]



logre = LogisticRegression(random_state=1, )

logre.fit(X_train, y_train)

logre_pre = logre.predict(X_test)

#logre_auc = roc_auc_score(y_test, logre_pre)



gb = GradientBoostingClassifier(n_estimators=120, learning_rate=1.0,

                                 max_depth=2, random_state=1).fit(X_train, y_train)

gb_pre = gb.predict(X_test)

#gb_auc = roc_auc_score(y_test, gb_pre)





rf = RandomForestClassifier(random_state=1, 

        n_estimators=70,

        criterion="gini",

        min_samples_leaf=2,

        min_samples_split=3,)

rf.fit(X_train, y_train)

rf_pre = rf.predict(X_test)

#rf_auc = roc_auc_score(y_test, rf_pre)



GB = GaussianNB()

GB.fit(X_train, y_train)

GB_pre = GB.predict(X_test)

#GB_auc = roc_auc_score(y_test, GB_pre)



le = LabelEncoder()

nonnumeric_columns = ['Sex', 'Embarked']

for features in nonnumeric_columns:

    X_train[features] = le.fit_transform(X_train[features])

    X_test[features] = le.fit_transform(X_test[features])

gbm = xgb.XGBClassifier(n_estimators=120,

        max_depth=5,

        min_child_weight=2,

        learning_rate=1.0)

gbm.fit(X_train, y_train)

gbm_pre = gbm.predict(X_test)

#gbm_auc = roc_auc_score(y_test, gbm_pre)



tree = DecisionTreeClassifier(criterion="entropy",

        max_depth=7,

        min_samples_split=2,

        min_samples_leaf=1,

        min_impurity_split= 0.3,

        random_state=1)

tree.fit(X_train, y_train)

tree_pre = tree.predict(X_test)

#tree_auc = roc_auc_score(y_test, tree_pre)



KNN = KNeighborsClassifier(n_neighbors=7, p=4)

KNN.fit(X_train, y_train)

KNN_pre = KNN.predict(X_test)

#KNN_auc = roc_auc_score(y_test, KNN_pre)



svc = SVC(kernel="linear",degree=3)

svc.fit(X_train, y_train)

svc_pre = svc.predict(X_test)

#svc_auc = roc_auc_score(y_test, svc_pre)





eclf = VotingClassifier(estimators=[('lr', logre), ('rf', rf), ('gb', gb), ('xgb', gbm), ('tree', tree)],voting="hard")

eclf.fit(X_train, y_train)

Y_pre = eclf.predict(X_test)

#eclf_auc = roc_auc_score(y_test, Y_pre)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pre

    })

submission.to_csv('titanic.csv', index=False)











    




