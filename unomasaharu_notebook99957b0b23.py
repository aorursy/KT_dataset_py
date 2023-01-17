# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load the data
data = pd.read_csv("/kaggle/input/drug-classification/drug200.csv")
data.head()
data.info()
y = data.Drug
X = data.drop("Drug", axis=1)
num_X = X[["Age", "Na_to_K"]]
cat_X = X[["Sex", "BP", "Cholesterol"]]
from sklearn.preprocessing import OneHotEncoder
oh_enc = OneHotEncoder(sparse=False)
oh_cat = pd.DataFrame(oh_enc.fit_transform(cat_X))
oh_cat.index = num_X.index
oh_X = pd.concat([num_X, oh_cat], axis=1)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

clf = xgb.XGBClassifier()
clf_cv = GridSearchCV(clf, 
                      {"max_depth": np.arange(2, 11, 2), 
                       "n_estimators": np.arange(20, 210, 20)}, 
                      cv=4, 
                      verbose=1)
clf_cv.fit(oh_X, y)
print(f"Best Score: {clf_cv.best_score_}")
print(f"Best Score: {clf_cv.best_params_}")
rfclf = xgb.XGBRFClassifier()
rfclf_cv = GridSearchCV(rfclf, 
                        {"max_depth": np.arange(2, 11, 2), 
                         "n_estimators": np.arange(20, 210, 20)}, 
                        cv=4, 
                        verbose=1)
rfclf_cv.fit(oh_X, y)
print(f"Best Score: {rfclf_cv.best_score_}")
print(f"Best Score: {rfclf_cv.best_params_}")
from sklearn.model_selection import train_test_split
X_tr, X_va, y_tr, y_va = train_test_split(oh_X, y, 
                                           test_size=0.2)
my_model = xgb.XGBRFClassifier(max_depth=4,
                               n_estimators=60)

my_model.fit(X_tr, y_tr)
pred = my_model.predict(X_va)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_va, pred)
print(score)
xgb.plot_importance(my_model)
