# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
def loadData(df, test = None):
    
    dt = pd.to_datetime(df.datetime).dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Day"] = dt.day
    df["Hour"] = dt.hour
    
    df.drop("datetime", axis = 1, inplace = True)
    if not test:
        df.drop("casual", axis = 1, inplace = True)
        df.drop("registered", axis = 1, inplace = True)
    if test:
        y = None
    else:
        y = df["count"]
        df.drop("count", axis = 1, inplace = True)
        
    X = df
    
    return X, y
        
        
X, y = loadData(train)
new_y = np.log(y + 1)
# use a full grid over all parameters
'''
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False]}

clf = RandomForestRegressor(n_estimators=20)
# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
#report(grid_search.grid_scores_)
'''
### GridSearchCV test
#X_test, _ = loadData(test, test = True)
#prediction = grid_search.predict(X_test)
### RF validation
'''
X_train, X_test, y_train, y_test = train_test_split(X, new_y, test_size = 0.33, random_state = 42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
mean_squared_error(y_test, prediction)
'''
# RF
X_test, _ = loadData(test, test = True)
rf = RandomForestRegressor().fit(X, new_y)
prediction = rf.predict(X_test)

prediction = np.exp(prediction) - 1
### xgb
#import xgboost as xgb
#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
#predictions = gbm.predict(test_X)
### Get submission
sample = pd.read_csv("../input/sampleSubmission.csv")
submission = pd.DataFrame()
submission["datetime"] = sample["datetime"]
submission["count"] = pd.Series(prediction)
submission.to_csv("sub.csv", index = False)
print(check_output(["head", "../input/sampleSubmission.csv"]).decode("utf8"))