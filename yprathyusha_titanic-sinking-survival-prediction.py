### Modules ##################

import csv

import pandas as pd

import numpy as np

from sklearn import tree

survival_clf = tree.DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score
training_df1 = pd.read_csv("../input/train.csv", skipinitialspace=True)

testing_df1 = pd.read_csv("../input/test.csv", skipinitialspace=True)
# print(training_df1.shape)

training_df1 = training_df1.fillna(training_df1.mode().iloc[0])

train_label = training_df1["Survived"]



training_df = training_df1.drop(['Survived', 'Name', 'Cabin','Ticket'], axis=1)

training_df.head()
training_df["Sex"] = training_df["Sex"].astype("category")

training_df["Sex"] = training_df["Sex"].cat.codes



training_df["Embarked"] = training_df["Embarked"].astype("category")

training_df["Embarked"] = training_df["Embarked"].cat.codes

training_df
survival_clf.fit(training_df.as_matrix(),train_label)

################ Training Accuracy #####################

cross_val_score(survival_clf, training_df.as_matrix(), train_label, cv=5)
testing_df1 = testing_df1.fillna(testing_df1.mode().iloc[0])

testing_df = testing_df1.drop(['Name', 'Cabin','Ticket'], axis=1)



############### Categorical to Numeric Conversion ############

testing_df["Sex"] = testing_df["Sex"].astype("category")

testing_df["Sex"] = testing_df["Sex"].cat.codes



testing_df["Embarked"] = testing_df["Embarked"].astype("category")

testing_df["Embarked"] = testing_df["Embarked"].cat.codes



############### Predicting the probability ############



predictval = survival_clf.predict(testing_df.as_matrix(), check_input=True)

predictval