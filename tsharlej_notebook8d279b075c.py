import numpy as np

import pandas as pd

from pandas import DataFrame, Series

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation



#load train and test.csv into list

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

list = [train, test]

#print(list)





#test correlation

print(train.corr().loc['Survived'])





#clean data and create an inverted index

for var in list:

    var["Age"] = var["Age"].fillna(var["Age"].median())

    var.loc[var["Sex"] == "male", "Sex"] = 0

    var.loc[var["Sex"] == "female", "Sex"] = 1

    #print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

    var["Embarked"] = var["Embarked"].fillna('S')

    var.loc[var["Embarked"] == "S", "Embarked"] = 0

    var.loc[var["Embarked"] == "C", "Embarked"] = 1

    var.loc[var["Embarked"] == "Q", "Embarked"] = 2

    var["Fare"] = var["Fare"].fillna(var["Fare"].median())

    print(var.columns)



#determine predictors

predictors = ["Pclass", "Sex", "Age", "Parch", "Fare"]



#initialize classifiers

algo = RandomForestClassifier(random_state = 1, n_estimators = 50, min_samples_split = 4\

, min_samples_leaf = 2)

print (list[0]["Survived"])

print (list[0].describe())



#score using cross validation

scores = cross_validation.cross_val_score( algo, list[0][predictors], list[0]["Survived"], cv =50)

print(scores.mean())



#fit classifiers

algo.fit(list[0][predictors], list[0]["Survived"])

results = algo.predict(list[1][predictors])

outcome = pd.DataFrame({"PassengerId" : list[1]["PassengerId"], "Survived": results})

outcome.to_csv("outcome.csv", index = False)

#print(results)