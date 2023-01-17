%load_ext autoreload
%autoreload 2

from fastai.imports import *

#from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import *
from feather import *
PATH = ""
df_raw = pd.read_csv('../input/train.csv', low_memory=False)
df_test = pd.read_csv('../input/test.csv', low_memory = False)
df_raw.head()
def display_all(df):
    with pd.option_context("display.max_rows",1000, "display.max_columns",1000):
        display(df)
display_all(df_raw.head().T)
df_train, df_valid, y_train, y_valid = train_test_split(df_raw.drop(['Id','Cover_Type'], axis = 1), df_raw.Cover_Type, test_size=0.20, random_state=42)
df_train.shape
df_valid.shape
m = RandomForestClassifier(n_jobs = -1)
m.fit(df_train, y_train)
def print_score(m):
    res = [m.score(df_train, y_train), m.score(df_valid, y_valid)]
    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)
    print (res)
m = RandomForestClassifier(n_jobs = -1)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators = 1, max_depth = 3, bootstrap = False, n_jobs = -1)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators = 1, bootstrap = False, n_jobs = -1)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier (n_jobs = -1)
%time m.fit(df_train, y_train)
print_score(m)
preds = np.stack([t.predict(df_valid) for t in m.estimators_])
preds[:,:3]
m = RandomForestClassifier (n_jobs = -1, n_estimators = 20)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier (n_jobs = -1, n_estimators = 40)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier (n_jobs = -1, n_estimators = 160)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier (n_jobs = -1, n_estimators = 40, oob_score = True)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 40, oob_score = True)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 40, oob_score = True)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 40, min_samples_leaf = 3, oob_score = True)
%time m.fit(df_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 40, min_samples_leaf = 2, oob_score = True, max_features = 0.5)
%time m.fit(df_train, y_train)
print_score(m)
df_valid.head()
df_test['Cover_Type'] = m.predict(df_test.drop('Id', axis = 1))
df_test.Cover_Type.value_counts()
df_test[['Id', 'Cover_Type']].to_csv('submission.csv', index=False)
