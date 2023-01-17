# Starting example using Gradient Boosting. You can add feature selection and grid search to improve it.

# Scoring 0.272
########################################################################################################

# Model building

########################################################################################################



import pandas as pd

import numpy as np

from collections import Counter

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion 

from sklearn.feature_selection import RFE

from matplotlib import pyplot

import warnings

warnings.filterwarnings('ignore')



# Load data

X_train = pd.read_csv("../input/train.csv")

print(Counter(X_train['target']))



# Extract some variables

Y_train = X_train['target']

del X_train['id']

del X_train['target']

print(X_train.columns)



# Internal variables

seed = 2017-20-11



# Create pipeline for GB

model = GradientBoostingClassifier(random_state = seed)

estimators = []

estimators.append(('model', model))

baseline = Pipeline(estimators)

baseline.fit(X_train, Y_train)
########################################################################################################

# Submit predictions

########################################################################################################



# Load data

X_test = pd.read_csv("../input/test.csv")

test_ids = X_test['id']

test_ids = pd.DataFrame(test_ids, columns=['id'])

del X_test['id']



# Predict

predictions = baseline.predict(X_test)

predictionsProb = baseline.predict_proba(X_test)



# Organize predictions

pred = pd.concat([test_ids, pd.DataFrame(predictionsProb), pd.DataFrame(predictions)], axis=1)

pred.columns = ['id','prob0', 'prob1', 'predicted']

print(pred.shape)



# Extract results

extracted = pred[['id','prob1']]

extracted.columns = ['id', 'target']

extracted.to_csv('results.csv', index = False)