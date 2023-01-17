# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/incident_event_logdata.csv',delimiter=',')
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
pd.set_option('max_colwidth', None)
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
# with the initial inspection, there are a lot of ? in data 
df.replace('?', np.NaN, inplace = True)
df.info()
# drop the columns that most values is nan
df1 = df.copy()
df1.drop(columns = ['cmdb_ci','problem_id','rfc','vendor','caused_by'], inplace = True)
# remove impact and urgency, since Priority value is directly computed from them.
df1.drop(columns = ['impact','urgency'], inplace = True)
# extract the numbers from the data 
pattern = r'(\d{1,4})'
colum = ['caller_id','opened_by','sys_created_by','sys_updated_by','location','category','subcategory','u_symptom','priority','assignment_group','assigned_to', 'closed_code', 'resolved_by']
for col in colum:
    df1[col] = df1[col].str.extract(pattern)

# time    
from datetime import datetime, date
timeColum = ['opened_at', 'sys_created_at','sys_updated_at','resolved_at','closed_at']    
for col in timeColum:
    df1[col] = pd.to_datetime(df1[col], format='%d/%m/%Y %H:%M',errors='coerce')
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
for col in df1.columns:
    print(col, df1[col].nunique())
# Distribution and correlation among columns
#%matplotlib notebook
idencolum = ['opened_by','sys_created_by','sys_updated_by','assignment_group','assigned_to','resolved_by']    
df_identify = df1.loc[:, idencolum]
for col in idencolum:
    df_identify[col] = pd.to_numeric(df_identify[col], errors='coerce').fillna(0).astype(np.int64)
plt.figure()
pd.plotting.scatter_matrix(df_identify,figsize=(12,12))
plt.savefig(r"Distribution and correlation among features.png")
# continue 
othercolum = ['reassignment_count','reopen_count','made_sla','category','priority','closed_code']

df_other = df1.loc[:, othercolum]
for col in othercolum:
    df_other[col] = pd.to_numeric(df_other[col], errors='coerce').fillna(0).astype(np.int64)
plt.figure()
pd.plotting.scatter_matrix(df_other,figsize=(12,12))
plt.savefig(r"Distribution and correlation among features_2.png")
plt.figure()
bins = np.arange(0,df1.incident_state.nunique()+2,1)

ax = df1.incident_state.hist(width =0.6,bins= bins,figsize=(6,4),align='mid')
plt.xticks(rotation=45)
ax.grid(False)
ax.set_xticks(bins[:-1])
ax.set_ylabel('Numbers')
ax.set_title('Distribution of the incident_state')
sla = (df1[(df1.made_sla == True) & (df1.reopen_count>0)].groupby('number')['reopen_count'].mean()).mean()
nosla = (df1[(df1.made_sla == False) & (df1.reopen_count>0)].groupby('number')['reopen_count'].mean()).mean()
print(f'mean reopen_count for having SLA {sla} and without SLA {nosla}')
# Distribution of closed_code; relationship between close_code and reopen_count
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,4))
bins=np.arange(0,df1.closed_code.nunique()+2,1)
df1[df1.incident_state=='Closed'].sort_values('closed_code').closed_code.hist(width =0.8,bins = bins,align='left',ax=ax1)
ax1.grid(False)
ax1.set_xticks(bins[:-1])
ax1.set_xlabel('closed code')
ax1.set_ylabel('Numbers')
ax1.set_title('Distribution of closed_code')


dfclosecode = df1[(df1.reopen_count>0) & (df1.incident_state=='Closed')]
dfclose_reopen = dfclosecode.groupby('closed_code').reopen_count.mean()
dfclose_reopen.plot.bar(ax=ax2)
ax2.grid(False)
ax2.set_ylabel('mean of reopen_count')
ax2.set_xticks(bins[:-1])
ax2.set_title('closed_code vs. reopen_count')
plt.show()
df_ar = df1.loc[:,['assigned_to','resolved_by']]
df_ar['equal'] = np.where(df_ar.assigned_to == df_ar.resolved_by,1,0)
equal_num = df_ar['equal'].sum()
print(equal_num/df_ar.shape[0] * 100)
# completion time for incident resolution 
df_closed = df1[df1.incident_state=='Closed'].reset_index()
df_closed['completion_time_days'] = (df_closed.closed_at- df_closed.opened_at).dt.total_seconds()/3600/24
#print(f'The mean of completion time for incident resolution is {df_closed.completion_time_days.mean()} days.')

#plots
plt.figure()
ax = df_closed['completion_time_days'].plot(figsize=(24,4))
ax.grid(False)
ax.set_ylabel('completion time in days')
ax.set_title('Distribution of completion_time')
df_closed['completion_time_days'].describe()
# completion time vs closed code; completion time vs made_sla

plt.figure()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,4))
df_closecode_time = df_closed.groupby('closed_code')['completion_time_days'].mean()
df_closecode_time.plot.bar(ax=ax1)
ax1.grid(False)
ax1.set_ylabel(' mean completion time in days')
ax1.set_title('completion time vs closed code')

df_made_sla_time = df_closed.groupby('made_sla')['completion_time_days'].mean()
df_made_sla_time.plot.bar(ax=ax2)
ax2.grid(False)
ax2.set_ylabel('mean completion time in days')
ax2.set_title('completion time vs made_sla')
plt.show()
X = df1[['made_sla', 'caller_id', 'contact_type', 'location','category', 'subcategory','u_symptom']]
y = df1.priority
#X.head(2)
for col in ['caller_id','location','category', 'subcategory','u_symptom']:
    X.loc[:,col] = pd.to_numeric(X.loc[:,col], errors='coerce').fillna(0).astype(np.int64)

# Label Encoding
enc= LabelEncoder()
for col in ['made_sla', 'contact_type']:
    X.loc[:,col] = enc.fit_transform(X.loc[:,col])
X.head(2)
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(np.int64)
# Splitting the data into test and train 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn.metrics import roc_curve, auc
# baseline, accuracy and confusion matrix of predicting 3 for all incidents 
print(f' The baseline of accuracy is {accuracy_score(y_test, np.full(y_test.shape, 3))}')
print(classification_report(y_test,np.full(y_test.shape, 3)))
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
y_predict_xgb = model_xgb.predict(X_test)
# Predicting the model
y_predict_xgb = model_xgb.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix
f1 = f1_score(y_test,y_predict_xgb,average='macro')
print(f'The macro F1 score for initial XGB model:{f1}')
print(classification_report(y_test,y_predict_xgb))
from sklearn.utils import class_weight
class_weights = list(class_weight.compute_class_weight('balanced',np.unique(y_train), y_train))

w_array = np.ones(y_train.shape[0], dtype = 'float')
for i, val in enumerate(y_train):
    w_array[i] = class_weights[val-1]

model_xgb_weight = XGBClassifier()   
model_xgb_weight.fit(X_train, y_train,sample_weight=w_array)
# Predicting the model
y_predict_xgb_weight = model_xgb_weight.predict(X_test) 
f1 = f1_score(y_test,y_predict_xgb_weight,average='macro')
print(f'The F1 score for XGB_weighted model:{f1}')
print(classification_report(y_test,y_predict_xgb_weight))
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

para_search ={'learning_rate':[0.2, 0.6, 1.2], 'n_estimators':[600, 800, 1200]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
grid_search = GridSearchCV(model_xgb_weight, param_grid = para_search, scoring='f1_macro',cv=kfold)
grid_search.fit(X_train, y_train,sample_weight=w_array)
print(grid_search.best_params_)
y_decision_fn = grid_search.predict(X_test) 
f1 = f1_score(y_test,y_decision_fn,average='macro')
print(f'The F1 score for XGB_weighted model after 1st tuning:{f1}')
print(classification_report(y_test,y_decision_fn))
# plot feature importance
from xgboost import plot_importance
model_grid = grid_search.best_estimator_
_ = plot_importance(model_grid, height = 0.9)
print(model_grid.feature_importances_)
# Standardization technique
sc = StandardScaler()
X_train_svm = sc.fit_transform(X_train)
X_test_svm = sc.transform(X_test)
#Initial model 
from sklearn.svm import SVC
rbf_svc = SVC(kernel='rbf',C=10,gamma=0.1).fit(X_train_svm,y_train)
# create target
df_ar = df1[['assigned_to','resolved_by']]
y2 = np.where(df_ar.assigned_to == df_ar.resolved_by,1,0)
df_closed.head(2)
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
# tick on sundays every week
loc = mdates.WeekdayLocator(byweekday=SU)

plt.figure()
df_opened_at_time = df_closed.groupby('opened_at')['completion_time_days'].mean()
axt = df_opened_at_time.plot(figsize=(20, 4))
axt.xaxis.set_minor_locator(loc)
axt.set_ylabel('mean completion time in days')
axt.set_xlabel('opened_at time')
plt.show()
df_closed['open_month'] = df_closed.opened_at.dt.month
df_closed['open_year'] = df_closed.opened_at.dt.year
df_closed['open_day'] = df_closed.opened_at.dt.day
plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(15,3))
df_open_year_time = df_closed.groupby('open_year')['completion_time_days'].mean()
df_open_year_time.plot.bar(ax=ax1)
ax1.grid(False)
ax1.set_ylabel('Mean completion time in days')
ax1.set_xlabel('Year')

df_open_month_time = df_closed.groupby('open_month')['completion_time_days'].mean()
df_open_month_time.plot(ax=ax2)
ax2.grid(False)
ax2.set_xlabel('Month')

df_open_day_time = df_closed.groupby('open_day')['completion_time_days'].mean()
df_open_day_time.plot(ax=ax3)
ax3.grid(False)
ax3.set_xlabel('day')
plt.show()
# time difference between resolved_at and closed_at 
df_closed['resolved_closed'] = ((df_closed.closed_at- df_closed.resolved_at).dt.total_seconds()/3600/24).fillna(0)
print(f'The mean time difference between resolved_at and closed_at is {df_closed.resolved_closed.mean()} days.')

#plots
plt.figure()
ax = df_closed['resolved_closed'].plot(figsize=(10,4))
ax.grid(False)
ax.set_ylabel('time difference in days')
y3 = df_closed.completion_time_days
X3 = df_closed[['category','subcategory', 'priority','caller_id','made_sla']]
# Label Encoding
# enc= LabelEncoder()
# X3['incident_state'] = enc.fit_transform(X3['incident_state'])
for col in ['category','subcategory', 'priority','caller_id']:
    X3[col] = pd.to_numeric(X3[col], errors='coerce').fillna(0).astype(np.int64)
# Splitting the data into test and train 
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,test_size=0.3,random_state=10)
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

xg_reg = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 500, seed = 1)
xg_reg.fit(X3_train, y3_train, eval_metric='mae')
y3_preds = xg_reg.predict(X3_test)
y3_preds[y3_preds <0] = 0
mae = mean_absolute_error(y3_test.values, y3_preds)
print("MAE: %f" % (mae))
plt.figure(figsize=(12,4))
plt.plot(y3_test.values,label="y3_test")
plt.plot(y3_preds,label="y3_preds")
plt.legend(loc='upper left') 