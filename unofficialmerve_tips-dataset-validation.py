
import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/tipping/tips.csv")
df = pd.concat([df, pd.get_dummies(df["sex"],prefix="sex")], axis=1)
df = pd.concat([df, pd.get_dummies(df["day"],prefix="day")], axis=1)
df = pd.concat([df, pd.get_dummies(df["time"],prefix="time")], axis=1)
df = pd.concat([df, pd.get_dummies(df["smoker"],prefix="smoker")], axis=1)
df.head()
columns_to_scale = ['tip', 'size', 'total_bill']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_columns = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]),columns=columns_to_scale)
scaled_columns.describe()
df.drop(["total_bill", "tip", "size", "smoker", "sex","day", "time"], axis = 1, inplace = True)
df = pd.concat([df, scaled_columns], axis = 1)
df.head()
df.drop(["sex_Female","time_Dinner", "smoker_No"], axis=1, inplace=True)
Y = df.tip
X = df.loc[:,df.columns!="tip"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
from sklearn.svm import SVR
svregressor = SVR()
svregressor.fit(X_train, y_train)
predsvr = svregressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test, predsvr))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predsvr))
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, predsvr))
from sklearn.model_selection import cross_val_score
cross_val_score(svregressor, X, Y, cv=10).mean()
import sklearn
from sklearn import metrics
sklearn.metrics.SCORERS.keys()
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
param_grid = [ {"C":[0.0001, 0.01, 0.1, 1, 10], "kernel":["poly", "linear", "rbf"], "gamma":[0.0001, 0.01, 0.1, 1, 10]}]
grid_search = GridSearchCV(svregressor, param_grid, cv=5)
grid_search.fit(X, Y)
grid_search.best_estimator_
grid_search.best_score_
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
distributions = dict(C = uniform(loc = 0, scale = 4), kernel=["poly","rbf","linear"])
random = RandomizedSearchCV(svregressor, distributions, random_state=0)
search = random.fit(X, Y)
search.best_params_
df.head()
Y = df.sex_Male
X = df.loc[:,df.columns!="sex_Male"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
predsvc = svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predsvc))
from sklearn.model_selection import cross_val_score
cross_val_score(svc, X, Y, cv=10).mean()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(svc, X, Y, cv=skf)
print("skorlar:\n{}".format(scores))
print("skorların ortalaması:\n{}".format(scores.mean()))
from sklearn.model_selection import GridSearchCV
param_grid = [ {"C":[0.0001, 0.01, 0.1, 1, 10], "kernel":["poly", "linear", "rbf"], "gamma":[0.0001, 0.01, 0.1, 1, 10]}]
grid_search_c = GridSearchCV(svc, param_grid, cv = 5, scoring = "accuracy")
grid_search_c.fit(X, Y)
grid_search_c.best_estimator_
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
distributions = dict(C = uniform(loc = 0, scale = 4), kernel=["poly","rbf","linear"])
random_c = RandomizedSearchCV(svc, distributions, random_state=0)
search_c = random_c.fit(X, Y)
search_c.best_params_
search_c.best_score_