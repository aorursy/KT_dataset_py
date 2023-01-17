import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.isnull().sum()
X = df.drop(['DEATH_EVENT'], axis = 1)

y = df['DEATH_EVENT']
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import GridSearchCV 
c_space = np.logspace(-5, 8, 15) 

param_grid = {'C': c_space}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)
logreg_cv.fit(X, y) 
logreg_cv.best_params_
logreg_cv.best_score_
from scipy.stats import randint 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import RandomizedSearchCV
param_dist = {"max_depth": [3, None], 

              "max_features": randint(1, 9), 

              "min_samples_leaf": randint(1, 9), 

              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv = 5)
tree_cv.fit(X, y)
tree_cv.best_params_
tree_cv.best_score_