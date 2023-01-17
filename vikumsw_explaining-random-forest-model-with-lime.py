# Importing Required Libraries

import numpy as np

import pandas as pd

import os

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
data = pd.read_csv("../input/train.csv")

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train.head()
# Dropping Features

train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)



train = train.drop(['Ticket'], axis=1)

test = test.drop(['Ticket'], axis=1)



train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)



train = train.drop(['PassengerId'], axis=1)

test = test.drop(['PassengerId'], axis=1)



# Convert categorical variables into dummy/indicator variables

train_processed = pd.get_dummies(train)

test_processed = pd.get_dummies(test)



# Filling Null Values

train_processed = train_processed.fillna(train_processed.mean())

test_processed = test_processed.fillna(test_processed.mean())



# Create X_train,Y_train,X_test

X_train = train_processed.drop(['Survived'], axis=1)

Y_train = train_processed['Survived']



X_test  = test_processed.drop(['Survived'], axis=1)

Y_test  = test_processed['Survived']



# Display

print("Processed DataFrame for Training : Survived is the Target, other columns are features.")

display(train_processed.head())
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest_preds = random_forest.predict(X_test)

print('The accuracy of the Random Forests model is :\t',metrics.accuracy_score(random_forest_preds,Y_test))
import lime

import lime.lime_tabular
predict_fn_rf = lambda x: random_forest.predict_proba(x).astype(float)

X = X_train.values

explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = X_train.columns,class_names=['Will Die','Will Survive'],kernel_width=5)
test.loc[[421]]
choosen_instance = X_test.loc[[421]].values[0]

exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)

exp.show_in_notebook(show_all=False)
test.loc[[310]]
choosen_instance = X_test.loc[[310]].values[0]

exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)

exp.show_in_notebook(show_all=False)
test.loc[[736]]
choosen_instance = X_test.loc[[736]].values[0]

exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)

exp.show_in_notebook(show_all=False)
test.loc[[788]]
choosen_instance = X_test.loc[[788]].values[0]

exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)

exp.show_in_notebook(show_all=False)