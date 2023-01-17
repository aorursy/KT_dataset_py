from fastai.imports import *
from fastai.structured import *
%load_ext autoreload
%autoreload 2
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
PATH = "../input/"
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory = False)
df_test = pd.read_csv(f'{PATH}test.csv', low_memory = False)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
            display(df)
display_all(df_raw.head().T)
display_all(df_raw.tail().T)
for col in df_raw.columns.tolist(): 
    if df_raw[col].dtype =='object':
        print (col, df_raw[col].dtype)
df_raw.edjefa.value_counts()
df_raw.edjefe.value_counts()
df_raw.dependency.value_counts()
df_raw[df_raw['dependency'] == 'yes'].idhogar.value_counts()
display_all(df_raw[df_raw['idhogar'] == 'ae6cf0558'])
df_raw.Target.value_counts()
display_all(df_raw[df_raw['idhogar'] == 'fd8a6d014'][df_raw['parentesco1'] == 1])
df_raw['idhogar'].nunique()
df_raw['parentesco1'].sum()
df_raw['parentesco1'].unique()
df_raw.groupby(['idhogar'])[['parentesco1']].sum().sort_values(by = 'parentesco1')
display_all(df_raw[df_raw['idhogar'].isin(['1bc617b23','03c6bdf85','61c10e099','ad687ad89','1367ab31d','f2bfa75c4','6b1b2405f','896fe6d3e','c0c8a5013','b1f4d89d7','374ca5a19','bfd5067c2','a0812ef17','d363d9183','09b195e7a'])].sort_values('idhogar'))
df_raw.head()
df_raw.edjefe.unique()
df_raw.edjefa.unique()
df_raw.edjefe.head()
df_raw.edjefe = np.where(df_raw.edjefe == 'yes', 1,
        np.where(df_raw.edjefe == 'no', 0,
                df_raw.edjefe))
df_raw.edjefa = np.where(df_raw.edjefa == 'yes', 1,
        np.where(df_raw.edjefa == 'no', 0,
                df_raw.edjefa))
df_raw.dependency = np.where(df_raw.dependency == 'yes', 1,
        np.where(df_raw.dependency == 'no', 0,
                df_raw.dependency))
df_test.edjefe = np.where(df_test.edjefe == 'yes', 1,
        np.where(df_test.edjefe == 'no', 0,
                df_test.edjefe))
df_test.edjefa = np.where(df_test.edjefa == 'yes', 1,
        np.where(df_test.edjefa == 'no', 0,
                df_test.edjefa))
df_test.dependency = np.where(df_test.dependency == 'yes', 1,
        np.where(df_test.dependency == 'no', 0,
                df_test.dependency))
df_raw = df_raw.sort_values(by = 'idhogar')
df_raw = df_raw.reset_index (drop = True)
train_cats(df_raw)
apply_cats(df_test, df_raw)
df_train, label_train, nas = proc_df(df_raw.drop(['Id', 'idhogar'], axis = 1), 'Target')
df_test, y_test, nas2 = proc_df(df_test.drop(['idhogar'], axis = 1), y_fld = None, na_dict = nas)
X_train = df_train[0:7645]
y_train = label_train[0:7645]
print (X_train.shape)
print (y_train.shape)
X_valid = df_train[7645:]
y_valid = label_train[7645:]
print (X_valid.shape)
print (y_valid.shape)
def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)
    print (res)
m = RandomForestClassifier(n_jobs = -1)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators = 3, n_jobs = -1)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators = 3, n_jobs = -1, min_samples_leaf=5)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators = 3, n_jobs = -1, min_samples_leaf=5, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 40, min_samples_leaf = 3, oob_score = True, max_features = 0.5)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 80, min_samples_leaf = 3, oob_score = True, max_features = 0.5)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 120, min_samples_leaf = 3, oob_score = True, max_features = 0.5)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs = -1, n_estimators = 120, min_samples_leaf = 4, oob_score = True, max_features = 0.5)
%time m.fit(X_train, y_train)
print_score(m)
df_test['Target'] = m.predict(df_test.drop(['Id'], axis = 1))
df_test[['Id', 'Target']].to_csv('submission.csv', index=False)
