# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #Library imported from plots

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

trainDF = pd.read_csv('../input/train.csv') # Getting DB For machien Learning and Vizualisation. 

testDF = pd.read_csv('../input/test.csv')
trainDF.info()
trainDF_ = trainDF.drop("Name", axis=1)

trainDF_ = trainDF_.drop("Ticket", axis=1)

trainDF_ = trainDF_.drop("Cabin", axis=1)

trainDF_ = trainDF_.drop("Embarked", axis=1)

trainDF_['Sex'].replace(['female','male'],[0,1],inplace=True)



testDF_ = testDF.drop("Name", axis=1)

testDF_ = testDF_.drop("Ticket", axis=1)

testDF_ = testDF_.drop("Cabin", axis=1)

testDF_ = testDF_.drop("Embarked", axis=1)

testDF_['Sex'].replace(['female','male'],[0,1],inplace=True)



trainDF_.head(20)

testDF_.head(20)
# Now we can continue our process and fill these empty values

for col in trainDF_.columns.values:

    if trainDF_[col].isnull().values.any():

        print("Missing values in "+col)

#Filling Missing Values

trainDF_["Age"] = trainDF_["Age"].fillna(method="ffill")

testDF_["Age"] = testDF_["Age"].fillna(method="ffill")

testDF_["Fare"] = testDF_["Fare"].fillna(method="ffill")

trainusedFeatures = ["Pclass","Sex","Age","SibSp","Parch","Fare"]

trainX_ = trainDF_[trainusedFeatures]

trainY_ = trainDF_[["Survived"]]

testX_ = testDF_[trainusedFeatures]
trainX_.describe()
correlations = trainDF_.corr() 

correlations
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(correlations, annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.title("Correlation between Columns of dataFrame",y=1.08)

plt.show()


clf = RandomForestClassifier(max_depth=None, random_state=0)

clf.fit(trainX_, trainY_)

predictions = clf.predict(testX_)

submission = pd.DataFrame({ 'PassengerId': testDF_['PassengerId'],'Survived': predictions })

submission.to_csv("submission.csv", index=False)