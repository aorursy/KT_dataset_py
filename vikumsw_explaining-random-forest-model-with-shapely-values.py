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
import shap 
# Create Tree Explainer object that can calculate shap values

explainer = shap.TreeExplainer(random_forest)
#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.

test.loc[[421]]
# Calculate Shap values

choosen_instance = X_test.loc[[421]]

shap_values = explainer.shap_values(choosen_instance)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.

test.loc[[310]]
# Calculate Shap values

choosen_instance = X_test.loc[[310]]

shap_values = explainer.shap_values(choosen_instance)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.

test.loc[[736]]
# Calculate Shap values

choosen_instance = X_test.loc[[736]]

shap_values = explainer.shap_values(choosen_instance)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.

test.loc[[788]]
# Calculate Shap values

choosen_instance = X_test.loc[[788]]

shap_values = explainer.shap_values(choosen_instance)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
shap.summary_plot(shap_values, X_train)