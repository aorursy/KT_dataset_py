%load_ext autoreload
%autoreload 2

%matplotlib inline
import pandas as pd
import numpy as np
from structured import *
import warnings
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics 

warnings.filterwarnings('ignore')
! unzip ../input/bluebook-for-bulldozers/Train.zip
df_raw = pd.read_csv('./Train.csv',low_memory=False,
                    parse_dates=["saledate"])
df_raw.head()
df_raw.saledate
#since the kaggle competition evaluation metric is the RMSLE(Root mean square log error)
df_raw.SalePrice=np.log(df_raw.SalePrice)
df_raw.saledate #datatype is datetime
add_datepart(df_raw, 'saledate')
df_raw.head()
df_raw.columns
df_raw.info()
train_cats(df_raw) #converts most of these objects into categories
df_raw.info() #how most objects have been turned to category
df_raw.UsageBand
df_raw.UsageBand.cat.categories #gives you the categories for the usage band feature
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
#order it so the splitting gets the maximum benifit from it 
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.codes
df_raw.head() #usage band still says high or low but behind the scenes they've been made into numbers
df_raw.to_feather(('bulldozers-raw'))
df_raw = pd.read_feather('bulldozers-raw')
df, y, nas = proc_df(df_raw, 'SalePrice')
def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
# raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
import math
#let's track the metrics we're interested in 
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestRegressor() 
%time m.fit(X_train, y_train)
print_score(m) #training rmse, valid rmse, training accuracy and validation accuracy respectively
len(df_raw)
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=50_000) 
X_train, _ = split_vals(df_trn, 40_000) 
y_train, _ = split_vals(y_trn, 40_000) 
m = RandomForestRegressor()
%time m.fit(X_train, y_train)
print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds.shape
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m) #prev was 81
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=50, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=75, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=100, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=125, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=160, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=170, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=160, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m) #final output is oob error
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
m = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)