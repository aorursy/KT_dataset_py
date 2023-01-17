import numpy as np

import pandas as pd

 

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, r2_score

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/wineQualityReds.csv')
y = data.quality

X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=123, 

                                                    stratify=y)

 
clf = GradientBoostingClassifier(n_estimators=200, 

                                 learning_rate=0.2, 

                                 max_depth=5, 

                                 random_state=5)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

clf.score(X_test, y_test)
clf1 = LogisticRegression(random_state=5)

clf1.fit(X_train, y_train)

pred = clf1.predict(X_test)

clf1.score(X_test, y_test)
clf2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=5)

clf2.fit(X_train, y_train)

pred = clf2.predict(X_test)

clf2.score(X_test, y_test)
clf3 = GaussianNB()

clf3.fit(X_train, y_train)

pred = clf3.predict(X_test)

clf3.score(X_test, y_test)
clf4 = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=5)

clf4.fit(X_train, y_train)

pred = clf4.predict(X_test)

clf4.score(X_test, y_test)
print(r2_score(y_test, pred))

print(mean_squared_error(y_test, pred))