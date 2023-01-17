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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics



import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

import gc



import os

import warnings

warnings.filterwarnings('ignore')
dtypes={

    'ip': 'uint16', 'app': 'uint16','device':'uint16','os':'uint16','channel':'uint16','ips_attributed':'uint16','click_id':'uint32'

}
testing=True

if testing:

    train_path="../input/talkingdata-adtracking-fraud-detection/train_sample.csv"

    skiprows=None

    nrows=None

    colnames=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']

else:

    train_path="../input/talkingdata-adtracking-fraud-detection/train.csv"

    skiprows=None

    nrows=None

    colnames=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']



train_sample = pd.read_csv(train_path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=colnames)
len(train_sample.index)
print(train_sample.memory_usage())
print('Training dataset uses {0} MB'.format(train_sample.memory_usage().sum()/1024**2))
train_sample.head()
train_sample.info()
def fraction_unique(x):

    return len(train_sample[x].unique())

number_unique_vals={x:fraction_unique(x) for x in train_sample.columns}
number_unique_vals
train_sample.dtypes
plt.figure(figsize=(14,10))

sns.countplot(x='app',data=train_sample)
plt.figure(figsize=(14, 8))

sns.countplot(x="device", data=train_sample)
plt.figure(figsize=(14, 8))

sns.countplot(x="channel", data=train_sample)
plt.figure(figsize=(14, 8))

sns.countplot(x="os", data=train_sample)
train_sample['is_attributed'].astype('object').value_counts()/len(train_sample.index)
app_target=train_sample.groupby('app').is_attributed.agg(['mean','count'])

app_target
frequent_apps=train_sample.groupby('app').size().reset_index(name='count')

frequent_apps=frequent_apps[frequent_apps['count']>frequent_apps['count'].quantile(0.80)]

frequent_apps=frequent_apps.merge(train_sample,on='app',how='inner')

frequent_apps.head()
plt.figure(figsize=(10,10))

sns.countplot(y="app", hue="is_attributed", data=frequent_apps);
def time_features(df):

    df['datetime']=pd.to_datetime(df['click_time'])

    df['day_of_week']=df['datetime'].dt.dayofweek

    df['day_of_year']=df['datetime'].dt.dayofyear

    df['month']=df['datetime'].dt.month

    df['hour']=df['datetime'].dt.hour

    return df
train_sample=time_features(train_sample)

train_sample.drop(['click_time','datetime'],axis=1,inplace=True)

train_sample.head()
train_sample.dtypes
int_vars = ['app', 'device', 'os', 'channel', 'day_of_week','day_of_year', 'month', 'hour']

train_sample[int_vars]=train_sample[int_vars].astype('uint16')
train_sample.dtypes
ip_count = train_sample.groupby('ip').size().reset_index(name='ip_count').astype('int16')

ip_count.head()
# creates groupings of IP addresses with other features and appends the new features to the df

def grouped_features(df):

    # ip_count

    ip_count = df.groupby('ip').size().reset_index(name='ip_count').astype('uint16')

    ip_day_hour = df.groupby(['ip', 'day_of_week', 'hour']).size().reset_index(name='ip_day_hour').astype('uint16')

    ip_hour_channel = df[['ip', 'hour', 'channel']].groupby(['ip', 'hour', 'channel']).size().reset_index(name='ip_hour_channel').astype('uint16')

    ip_hour_os = df.groupby(['ip', 'hour', 'os']).channel.count().reset_index(name='ip_hour_os').astype('uint16')

    ip_hour_app = df.groupby(['ip', 'hour', 'app']).channel.count().reset_index(name='ip_hour_app').astype('uint16')

    ip_hour_device = df.groupby(['ip', 'hour', 'device']).channel.count().reset_index(name='ip_hour_device').astype('uint16')

    

    # merge the new aggregated features with the df

    df = pd.merge(df, ip_count, on='ip', how='left')

    del ip_count

    df = pd.merge(df, ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left')

    del ip_day_hour

    df = pd.merge(df, ip_hour_channel, on=['ip', 'hour', 'channel'], how='left')

    del ip_hour_channel

    df = pd.merge(df, ip_hour_os, on=['ip', 'hour', 'os'], how='left')

    del ip_hour_os

    df = pd.merge(df, ip_hour_app, on=['ip', 'hour', 'app'], how='left')

    del ip_hour_app

    df = pd.merge(df, ip_hour_device, on=['ip', 'hour', 'device'], how='left')

    del ip_hour_device

    

    return df
train_sample = grouped_features(train_sample)

train_sample.head()
gc.collect()
X = train_sample.drop('is_attributed', axis=1)

y = train_sample[['is_attributed']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
print(y_train.mean())

print(y_test.mean())
tree=DecisionTreeClassifier(max_depth=2)



adaboost_model_1=AdaBoostClassifier(

    base_estimator=tree,

    n_estimators=600,

    learning_rate=1.54,

    algorithm="SAMME")
adaboost_model_1.fit(X_train,y_train)
predictions=adaboost_model_1.predict_proba(X_test)

predictions[:10]
metrics.roc_auc_score(y_test,predictions[:,1])
param_grid={"base_estimator__max_depth":[2,5],

           "n_estimators":[200,400,600]

           }
tree=DecisionTreeClassifier()

ABC=AdaBoostClassifier(

    base_estimator=tree,

    learning_rate=0.6,

    algorithm="SAMME")
folds=3

grid_search_ABC=GridSearchCV(ABC,

                            cv=folds,

                            param_grid=param_grid,

                            scoring='roc_auc',

                            return_train_score=True,

                            verbose=1)
grid_search_ABC.fit(X_train,y_train)
cv_results = pd.DataFrame(grid_search_ABC.cv_results_)

cv_results
# plotting AUC with hyperparameter combinations



plt.figure(figsize=(16,6))

for n, depth in enumerate(param_grid['base_estimator__max_depth']):

    



    # subplot 1/n

    plt.subplot(1,3, n+1)

    depth_df = cv_results[cv_results['param_base_estimator__max_depth']==depth]



    plt.plot(depth_df["param_n_estimators"], depth_df["mean_test_score"])

    plt.plot(depth_df["param_n_estimators"], depth_df["mean_train_score"])

    plt.xlabel('n_estimators')

    plt.ylabel('AUC')

    plt.title("max_depth={0}".format(depth))

    plt.ylim([0.60, 1])

    plt.legend(['test score', 'train score'], loc='lower left')

    plt.xscale('log')



    

tree = DecisionTreeClassifier(max_depth=2)

ABC = AdaBoostClassifier(

    base_estimator=tree,

    learning_rate=0.6,

    n_estimators=200,

    algorithm="SAMME")



ABC.fit(X_train, y_train)
predictions = ABC.predict_proba(X_test)

predictions[:10]
metrics.roc_auc_score(y_test, predictions[:, 1])
param_grid = {"learning_rate": [0.2, 0.6, 0.9],

              "subsample": [0.3, 0.6, 0.9]

             }
GBC = GradientBoostingClassifier(max_depth=2, n_estimators=200)
folds = 3

grid_search_GBC = GridSearchCV(GBC, 

                               cv = folds,

                               param_grid=param_grid, 

                               scoring = 'roc_auc', 

                               return_train_score=True,                         

                               verbose = 1)



grid_search_GBC.fit(X_train, y_train)
cv_results = pd.DataFrame(grid_search_GBC.cv_results_)

cv_results.head()
plt.figure(figsize=(16,6))





for n, subsample in enumerate(param_grid['subsample']):

    



    # subplot 1/n

    plt.subplot(1,len(param_grid['subsample']), n+1)

    df = cv_results[cv_results['param_subsample']==subsample]



    plt.plot(df["param_learning_rate"], df["mean_test_score"])

    plt.plot(df["param_learning_rate"], df["mean_train_score"])

    plt.xlabel('learning_rate')

    plt.ylabel('AUC')

    plt.title("subsample={0}".format(subsample))

    plt.ylim([0.60, 1])

    plt.legend(['test score', 'train score'], loc='upper left')

    plt.xscale('log')

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)

y_pred[:10]
roc = metrics.roc_auc_score(y_test, y_pred[:, 1])

print("AUC: %.2f%%" % (roc * 100.0))
folds = 3



# specify range of hyperparameters

param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]}          





# specify model

xgb_model = XGBClassifier(max_depth=2, n_estimators=200)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring= 'roc_auc', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



model_cv.fit(X_train, y_train)       
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')

cv_results.head()
plt.figure(figsize=(16,6))



param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]} 





for n, subsample in enumerate(param_grid['subsample']):

    



    # subplot 1/n

    plt.subplot(1,len(param_grid['subsample']), n+1)

    df = cv_results[cv_results['param_subsample']==subsample]



    plt.plot(df["param_learning_rate"], df["mean_test_score"])

    plt.plot(df["param_learning_rate"], df["mean_train_score"])

    plt.xlabel('learning_rate')

    plt.ylabel('AUC')

    plt.title("subsample={0}".format(subsample))

    plt.ylim([0.60, 1])

    plt.legend(['test score', 'train score'], loc='upper left')

    plt.xscale('log')
params = {'learning_rate': 0.2,

          'max_depth': 2, 

          'n_estimators':200,

          'subsample':0.6,

         'objective':'binary:logistic'}



# fit model on training data

model = XGBClassifier(params = params)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)

y_pred[:10]
auc = sklearn.metrics.roc_auc_score(y_test, y_pred[:, 1])

auc
results = pd.concat([ pd.Series(y_pred[:, 1], name="is_attributed")], axis=1)

results.head()
test_path="../input/talkingdata-adtracking-fraud-detection/test.csv"

skiprows = None

nrows = None

colnames=['click_id']

test_sample = pd.read_csv(test_path,usecols=colnames)



test_sample.head()
results = pd.concat([test_sample, pd.Series(y_pred[:, 1], name="is_attributed")], axis=1)

results.head()
results = results.sort_values(by="click_id", axis=0).reset_index().drop("index", axis=1)

results.head()
results.to_csv("submission_file.csv", sep=',', index=False)
results.isna()
results=results.dropna()
results.info()
results.to_csv("submission_file.csv", sep=',', index=False)