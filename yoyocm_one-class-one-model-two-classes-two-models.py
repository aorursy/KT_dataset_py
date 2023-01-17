import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import CountVectorizer

import warnings

warnings.filterwarnings('ignore')
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Concatenating data sets

data = pd.concat([train, test], axis=0)



# Generate features from categorical features

data = pd.get_dummies(data)



# Replace NAs with means

data = data.fillna(data.mean())
# Re-split dataset to X_train and X_test

train = data[:train.shape[0]]

test = data[train.shape[0]:]
# Getting output variable

#y = train['Survived']

#train = train.drop(['Survived'],axis=1)



y_c1 = train['Survived'][ train['Pclass'] == 1 ]

y_c2 = train['Survived'][ train['Pclass'] == 2 ]

y_c3 = train['Survived'][ train['Pclass'] == 3 ]




# Filtering on passenger class 

data_c1 = train[ train['Pclass'] == 1 ]

data_c2 = train[ train['Pclass'] == 2 ]

data_c3 = train[ train['Pclass'] == 3 ]

rfc1 = RandomForestClassifier(n_jobs=-1,n_estimators=1000)

rfc1.fit(data_c1,y_c1)



rfc2 = RandomForestClassifier(n_jobs=-1,n_estimators=1000)

rfc2.fit(data_c2,y_c2)



rfc3 = RandomForestClassifier(n_jobs=-1,n_estimators=1000)

rfc3.fit(data_c3,y_c3)



print(rfc1.score(data_c1,y_c1))

print(rfc2.score(data_c2,y_c2))

print(rfc3.score(data_c3,y_c3))
res = []



for index, row in test.iterrows():

    if row['Pclass'] == 1 :

        res.append(rfc1.predict(row)[0])

    elif row['Pclass'] == 2 : 

        res.append(rfc2.predict(row)[0])

    elif row['Pclass'] == 3 : 

        res.append(rfc3.predict(row)[0])



submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived":res})

submission = submission.set_index("PassengerId")



submission.to_csv('submission.csv')