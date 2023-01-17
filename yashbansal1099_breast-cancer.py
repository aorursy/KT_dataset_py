import pandas as pd

import numpy as np

import re

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB 
# Read Data

# transform Outcome Column from string to float

# Delete not needed Data
x = pd.read_excel("../input/innovacer-challenge-breast-cancer-prediction/BreastCancer_Prognostic_v1.xlsx")

outcome = x["Outcome"] 

ids1 = x["ID"]

le = LabelEncoder()

for i in range(x.shape[0]):

    if x['Lymph_Node_Status'][i] == '?':

        x['Lymph_Node_Status'][i] = 0

outcome = le.fit_transform(outcome)

x1 = x.drop(['ID','Outcome','Time'], axis = 1)

x.shape
x.head()
# Confusion Matrix Design



# [[True Negative, False Negative]

#  [False Recurrence, True Recurrence]

#Split data for Classification
from sklearn.model_selection import train_test_split

x_train, x_test, outcome_train, outcome_test, ids1_train, ids1_test= train_test_split(x1, outcome, 

                                                                ids1, test_size=0.1, random_state = 10) 
# classifier to predict the outcome of a new patient using Multinomial Naive Bayes


mnb = MultinomialNB(alpha = 4)

mnb.fit(x_train, outcome_train) 

outcome_pred1 = mnb.predict(x_test)

print(le.inverse_transform(outcome_pred1))

print(le.inverse_transform(outcome_test))

print(accuracy_score(outcome_pred1, outcome_test)*100)

print(confusion_matrix(outcome_pred1, outcome_test))
# classifier to predict the outcome of a new patient using Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(splitter = 'random',max_depth = 12)

clf.fit(x_train, outcome_train)

outcome_pred2 = clf.predict(x_test)

print(le.inverse_transform(outcome_pred2))

print(le.inverse_transform(outcome_test))

print(accuracy_score(outcome_pred2, outcome_test)*100)

print(confusion_matrix(outcome_pred2, outcome_test))
# classifier to predict the outcome of a new patient using Gradient Boosting Algorithm
from sklearn.ensemble import GradientBoostingClassifier

gbc= GradientBoostingClassifier(learning_rate = 0.1)

gbc.fit(x_train, outcome_train)

outcome_pred3 = gbc.predict(x_test)

print(le.inverse_transform(outcome_pred3))

print(le.inverse_transform(outcome_test))

print(accuracy_score(outcome_pred3, outcome_test)*100)

print(confusion_matrix(outcome_pred3, outcome_test))
# Since a person can be Treated if proved to be false Recurrsive and can be discharged once found 

# but if there is someone who can has breast Cancer but is predicted as Negative this can be dangerous to his/her health 

# So we select the outcome of the Multinomial Naive Bayes algorithm because it give the minimum False Negative
outcome_pred1 = le.inverse_transform(outcome_pred1)
out = list(zip(ids1_test, outcome_pred1))

print(out)
ids2_test = []

for i in range(len(out)):

    if out[i][1] =='R':

        ids2_test.append(out[i][0])

print(ids2_test)
# Remove the Data Having N outcome so that we are left with data having R outcome only

#make x, y and id and then split the data 
x = x.set_index("Outcome")

x2 = x.drop('N', axis = 0)

x2 = x2.reset_index("Outcome")

x2 = x2.drop("Outcome", axis = 1)

# create IDS to to calcuate the persons time of recurrence 
ids2 = x2["ID"]

ids2_train = []

for ids in ids2:

    if ids not in ids2_test:

        ids2_train.append(ids)
#create the training and test data for the time of recurrence calculation
x2_train = x2.loc[x2['ID'].isin(ids2_train)]

time_train = x2_train["Time"]

x2_test = x.loc[x['ID'].isin(ids2_test)]

x2_test = x2_test.reset_index("Outcome")

x2_test = x2_test.drop("Outcome", axis = 1)

time_test = x2_test["Time"]

# x2_train = x2_train.values

# x2_test = x2_test.values

# time_test = time_test.values

# time_train = time_train.values
x2_train.columns
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators = 1000, criterion = 'mae')

reg.fit(x2_train, time_train)

time_pred = reg.predict(x2_test)

print(time_pred)

print(time_test)
out = list(zip(ids1_test, outcome_pred1))
o = list(zip(ids2_test, time_pred))
z = np.zeros(len(out))

for i in range(len(o)):

    for j in range(len(out)):

        if o[i][0] == out[j][0]:

            z[j] = o[i][1]
output = list(zip(ids1_test, outcome_pred1, z))
df = pd.DataFrame(output, columns = ["ID", "Outcome", "Time"])
df.head()
df.to_excel("Output.xlsx", index = False)