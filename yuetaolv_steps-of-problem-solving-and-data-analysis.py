import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import datetime
from datetime import date
import calendar
import dateutil
import os
## Check files
print(os.listdir("../input/elo-merchant-category-recommendation"))
file_path = '../input/elo-merchant-category-recommendation/train.csv'
train = pd.read_csv(file_path)
train.head()
## check the correlation of each feature and target
train.corr()
train.target.describe()
file_path = '../input/elo-merchant-category-recommendation/test.csv'
test = pd.read_csv(file_path)
test.head()
history = pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv')
history.head()
## Month_lag
fig = plt.figure(figsize=(14,6))
ax = sns.distplot(history.month_lag)
ax.set_xlabel('Month_lag', size=12)
ax.set_ylabel('Frequency', size=12)
ax.set_title('Month_lag', size=15)
history['month_lag'].describe()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
temp1 = history.groupby('city_id')['purchase_amount'].mean()
temp2 = history.groupby('state_id')['purchase_amount'].mean()

ax = [ax1, ax2]
temp = [temp1, temp2]
x_labels = ['city_id', 'state_id']
titles = ['Average City purchase amount', 'Average State purchase amount']
scale = [100, 10000]

for i in range(2):
    ax[i].scatter(x=temp[i].index, y=temp[i].values, c=temp[i].values, s=temp[i].values*scale[i], alpha=0.8)
    ax[i].set_xlabel(x_labels[i], size=12)
    ax[i].set_ylabel('Purchase amount', size=12)
    ax[i].set_title(titles[i], size=15)
## parse purchase_date and divide it into day, week, time session and see how important they will be
history['purchase_weekday'] = pd.to_datetime(history['purchase_date']).dt.day_name()
history['purchase_month'] = pd.to_datetime(history['purchase_date']).dt.month_name()
## Define a day session
## Morning: 5am to 12pm (05:00 to 11:59)
## Afternoon: 12pm to 5pm (12:00 to 16:59)
## Evening: 5pm to 9pm (17:00 to 20:59)
## Night: 9pm to 5am (21:00 to 04:59)

def time_session(time):
    
    if time >= 5 and time < 12:
        return 'Morning'
    elif time >=12 and time < 17:
        return 'Afternoon'
    elif time >=17 and time < 21:
        return 'Evening'
    else:
        return 'Night'
history['temp'] = pd.to_datetime(history['purchase_date']).dt.hour
history['purchase_time_session'] = history['temp'].apply(lambda x : time_session(x))
## Make categorical data has specific order
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
history['purchase_weekday'] = pd.Categorical(history['purchase_weekday'], categories=weekday_labels, ordered=True)

month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December']
history['purchase_month'] = pd.Categorical(history['purchase_month'], categories=month_labels, ordered=True)

session_labels = ['Morning', 'Afternoon', 'Evening', 'Night']
history['purchase_time_session'] = pd.Categorical(history['purchase_time_session'], categories=session_labels, ordered=True)
f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
temp1 = history.groupby('purchase_weekday')['purchase_amount'].mean()
temp2 = history.groupby('purchase_month')['purchase_amount'].mean()
temp3 = history.groupby('purchase_time_session')['purchase_amount'].mean()

a = sns.lineplot(x=temp1.index, y=temp1.values, data=history, ax=ax2)
b = sns.lineplot(x=temp2.index, y=temp2.values, data=history, ax=ax1)
c = sns.lineplot(x=temp3.index, y=temp3.values, data=history, ax=ax3)

plt.xlabel('Purchase time', size=12)
plt.ylabel('Purchase amount', size=12)
f.suptitle('Time Series Analysis', size=15)
mer = pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv')
mer.head()
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16,8))
temp1 = mer.groupby('city_id')['avg_sales_lag3'].sum()
temp2 = mer.groupby('city_id')['avg_sales_lag6'].sum()
temp3 = mer.groupby('city_id')['avg_sales_lag12'].sum()

ax = [ax1, ax2, ax3]
temp = [temp1, temp2, temp3]
y_labels = ['Total avg sales lag3', 'Total avg sales lag6', 'Total avg sales lag12']

for i in range(3):
    ax[i].scatter(x=temp[i].index, y=temp[i].values, s=temp[i].values/100, c=temp[i].values, alpha=0.8)
    ax[i].set_xlabel('City id', fontsize=12)
    ax[i].set_ylabel(y_labels[i])
    ax[i].set_title(y_labels[i] + ' in each city')
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16,8))
temp1 = mer.groupby('city_id')['avg_purchases_lag3'].sum()
temp2 = mer.groupby('city_id')['avg_purchases_lag6'].sum()
temp3 = mer.groupby('city_id')['avg_purchases_lag12'].sum()

ax = [ax1, ax2, ax3]
temp = [temp1, temp2, temp3]
y_labels = ['Total avg purchases lag3', 'Total avg purchases lag6', 'Total avg purchases lag12']

for i in range(3):
    ax[i].scatter(x=temp[i].index, y=temp[i].values, s=temp[i].values/100, c=temp[i].values, alpha=0.8)
    ax[i].set_xlabel('City id', fontsize=12)
    ax[i].set_ylabel(y_labels[i])
    ax[i].set_title(y_labels[i] + ' in each city')
new_mer = pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv')
new_mer.head()
train = pd.read_csv('../input/elo-combined-data/X.csv')
train.drop('Unnamed: 0', axis=1, inplace=True)
test = pd.read_csv('../input/elo-combined-data/X_test.csv')
test.drop('Unnamed: 0', axis=1, inplace=True)
y = pd.read_csv('../input/elo-combined-data/y.csv', header=None)
y.drop(0, axis=1, inplace=True)
y.rename({1: 'target'}, axis=1, inplace=True)
not_use_col = ['first_active_month', 'card_id']
use_cols = [col for col in train.columns if col not in not_use_col]
X = train[use_cols]
X_test = test[use_cols]
features = list(train[use_cols].columns)
categorical_feat = [col for col in features if 'feature_' in col]
def model():
    lgb_params = {
              'objective': 'regression',
              'metric': 'rmse',
              'max_depth': 11,
              'min_chil_samples': 20,
              'min_data_in_leaf': 200,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'num_leaves': 140,
              'learning_rate': 0.07,
              'subsample': 0.8,
              'colsample_bytress': 0.9,
              'verbosity': -1}
    
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    oof = np.zeros(len(X))
    predictions = np.zeros(len(X_test))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values, y.values)):
        print("LGB" + str(fold_) + '*' * 50)
        trn_data = lgb.Dataset(X.iloc[trn_idx][use_cols], label=y.iloc[trn_idx], categorical_feature=categorical_feat)
        val_data = lgb.Dataset(X.iloc[val_idx][use_cols], label=y.iloc[val_idx], categorical_feature=categorical_feat)

        num_round=1000

        clf = lgb.train(lgb_params, trn_data, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=600)
        oof[val_idx] = clf.predict(X.iloc[val_idx][use_cols], num_iteration = clf.best_iteration)
        predictions += clf.predict(X_test[use_cols], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, y)**0.5))
model()