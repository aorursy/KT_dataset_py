# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Visualization



# Metrics

from sklearn.metrics import accuracy_score
# Load training data into DataFrame

raw_data = pd.read_csv("../input/train.csv").set_index("PassengerId")

print(raw_data.head())



print("\nNumber of samples: {}".format(raw_data.shape[0]))
# Preprocessing

from sklearn.preprocessing import MinMaxScaler



def preprocess(data, target, dropna=True, impute=False, testing=False):

    # Drop irrelevant features

    data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1)



    # Remove samples that have missing values

    if (dropna):

        data.dropna(inplace=True)

        

    # Impute missing values using mean (no effect if dropna=True)

    if (impute):

        data.fillna(data.mean(), inplace=True)



    # One-hot encode categorical data

    data = pd.get_dummies(data, columns=['Pclass', 'Sex'])



    # Normalize 'Age' feature

    mms = MinMaxScaler()

    data['Age'] = mms.fit_transform(data['Age'].values.reshape(-1, 1))

    

    if (testing):

        return data

    else:

        return data.drop(target, axis=1), data[target]

    

X, y = preprocess(raw_data, "Survived")

print(X.head())



n_samples = X.shape[0]

print("\nNumber of samples after preprocessing: {}".format(n_samples))
# Build a naive predictor as a baseline for gauging performance of later models

# Since most people did not survive the disaster, predict no one survived

naive_pred = np.zeros(n_samples)



print("\nAccuracy of naive predictor on entire dataset: {:.3f}".format(accuracy_score(y, naive_pred)))
# Model

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from time import time



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# Try Logistic Regression

clf = LogisticRegression(random_state=42)

params = {'max_iter': [100, 1000],

          'C': [0.001, 0.01, 0.1, 1.0]}

sss = StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=0.2)

scorer = make_scorer(accuracy_score)

cv = GridSearchCV(clf, params, cv=sss, scoring=scorer)

start_time = time()

cv.fit(X, y)

total_time = time() - start_time



print("Best params for Logistic Regression: {}".format(cv.best_params_))

print("Accuracy score: {:.3f}".format(cv.best_score_))

print("Time to train and test: {:.3f}s".format(total_time))



# Try Random Forest

clf = RandomForestClassifier(random_state=42)

params = {'n_estimators': [10, 100],

          'min_samples_split': [2, 4, 10, 20, 50, 100]}

cv2 = GridSearchCV(clf, params, cv=sss, scoring=scorer)

start_time = time()

cv2.fit(X, y)

total_time = time() - start_time



print("\nBest params for Random Forest: {}".format(cv2.best_params_))

print("Accuracy score: {:.3f}".format(cv2.best_score_))

print("Time to train and test: {:.3f}s".format(total_time))
# Run on test data



raw_test_data = pd.read_csv("../input/test.csv").set_index("PassengerId")

X_test = preprocess(raw_test_data, "Survived", dropna=False, impute=True, testing=True)



pred = cv2.predict(X_test)

print(len(pred))



submission = pd.DataFrame(np.array([X_test.index.values, pred]).T, columns=["PassengerId", "Survived"])

submission.to_csv("submission.csv", index=False)