# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
### Try to use knn to predict ages, skipped now, try it later
'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
cols_list = ["Pclass", "SibSp", "Parch", "Fare"]
known = train[train.Age.notnull()]
X = known[cols_list]
y = known.Age
knn.fit(X,y)
'''
#df["Ticket_Value"] = df.Ticket.map(df.Ticket.value_counts())
#df.drop("Ticket", axis = 1, inplace = True)
#df.Ticket_Value
### Deleted, move to loadData function
### Pre-process
# missing Fare
#test.iloc[152, 8] = test.Fare.mean()

#missing Embark
#train.iloc[61, 11] = 'S'
#train.iloc[829, 11] = 'S'

def loadData(df, test = False):
    df.Fare[df.Fare.isnull()] = df.Fare.mean()
    df.Embarked[df.Embarked.isnull()] = 'S'
    df.Sex[df.Sex == "male"] = 1
    df.Sex[df.Sex == "female"] = 0
    df.Embarked[df.Embarked == "S"] = 0
    df.Embarked[df.Embarked == "C"] = 1
    df.Embarked[df.Embarked == "Q"] = 2

    df["Ticket_Value"] = df.Ticket.map(df.Ticket.value_counts())
    df.drop("Ticket", axis = 1, inplace = True)

    
    cols_list = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
    
    if test:
        y = None
    else:
        y = df.Survived
    X = df[cols_list]
    
    return X, y

X, y = loadData(train)
### CV for naive gbm
gbm = GradientBoostingClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 42)
gbm.fit(X_train,y_train)
pred = gbm.predict(X_test)
print(accuracy_score(y_test, pred))

### Predict result
X, _ = loadData(test, test = True)
pred = gbm.predict(X)
### GridSearchCV for gbm
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

kfold = KFold(n = len(X), n_folds = 3, random_state = 42)

gbm_grid = GridSearchCV(
  estimator = GradientBoostingClassifier(warm_start = True, random_state = 42),
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [1e-1, 1] 
    },
    cv = kfold,
    scoring = "accuracy"    
)

gbm_grid.fit(X, y)

### Predict result
X, _ = loadData(test, test = True)
pred = gbm.predict(X)
#gbm_grid.best_score_
gbm_grid.best_params_
submission = pd.DataFrame()
submission["PassengerId"] = test.PassengerId
submission["Survived"] = pred
submission.to_csv("sub.csv", index = False)