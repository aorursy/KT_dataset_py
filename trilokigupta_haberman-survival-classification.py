import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input"))



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv("../input/haberman.csv", names=['age', 'year_of_operation', 'positive_axillary_nodes', 'survival_status'])
df.columns
df.head(3)
df.dtypes
df.info()
df.describe()
X = df.drop(columns = ["survival_status"])

y = df.survival_status
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 42)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

model = SVC(kernel = "rbf")

model.fit(X_train, y_train)

predX = model.predict(X_test)

print(accuracy_score(predX, y_test))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

model = LogisticRegression(penalty="l2", solver='lbfgs')

model.fit(X_train, y_train)

predX = model.predict(X_test)

print("Accuracy is:", accuracy_score(predX, y_test))
import lightgbm as ltb

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

RF = RandomForestClassifier(random_state=1)

#parameter for RF

PRF = [{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]



#parameter tunning by GridSearchCV

GSRF = GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)



#score with cross validation 5

rf_scores = cross_val_score(GSRF,X_train,y_train,scoring='accuracy',cv=5)



print(np.mean(rf_scores)) 

# GSRF.fit(X_train, y_train)

# GSRF.best_estimator_