import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

plt.style.use('seaborn')
sns.set(font_scale=2)

%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from scipy import stats

from sklearn.model_selection import KFold
import time

import warnings
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print('Train (rows, columns)', df_train.shape)
print('test (rows, columns)', df_test.shape)
columns = df_train.columns

r_value_array = []
for num in range(2, df_train.shape[1]):
    x = df_train[columns[num]]
    y = df_train['target']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    r_value_array.append(r_value)
df_r_values = pd.DataFrame({'column': columns[2:], 'r_value': r_value_array})
plt.figure(figsize=(8, 8))
df_r_values['r_value'].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('r_value')
high_r_value_dataframe = df_r_values.loc[df_r_values.r_value.abs() > 0.05, :]
high_r_value_dataframe['column'].shape
high_r_value_dataframe.head()
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

num = np.random.randint(754)
temp_col = high_r_value_dataframe['column'].values[num]
temp_r_value = high_r_value_dataframe.loc[high_r_value_dataframe.column == temp_col, 'r_value'].values[0]
x = df_train[temp_col].values
y = df_train['target'].values
ax[0].scatter(x, y)
ax[0].set_yscale('log')
ax[0].set_title('{} {} \nr-value: {:.4f}'.format(num, columns[num], temp_r_value), fontsize=15)

x = df_train[df_train.columns[5]].values
y = df_train['target'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
ax[1].scatter(x, y)
ax[1].set_yscale('log')
ax[1].set_title('{} {} \nr-value: {:.4f}'.format(num, columns[num], temp_r_value), fontsize=15)
new_df_train = df_train[high_r_value_dataframe['column']]
new_df_test = df_test[high_r_value_dataframe['column']]

Y = np.log(df_train.target+1).values
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 8,
    'num_leaves': 32,  # 63, 127, 255
    'feature_fraction': 0.8, # 0.1, 0.01
    'bagging_fraction': 0.8,
    'learning_rate': 0.01, #0.00625,#125,#0.025,#05,
    'verbose': 1
}
kf = KFold(n_splits=10, shuffle=True, random_state=1989)
Y_target = []
for fold_id,(train_idx, val_idx) in enumerate(kf.split(new_df_train)):
    print('FOLD:',fold_id)
    X_train = new_df_train.values[train_idx]
    y_train = Y[train_idx]
    X_valid = new_df_train.values[val_idx]
    y_valid =  Y[val_idx]
    
    
    lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=new_df_train.columns.tolist(),
    #             categorical_feature = categorical
                         )

    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=new_df_train.columns.tolist(),
    #             categorical_feature = categorical
                         )

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=30000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    test_pred = lgb_clf.predict(new_df_test.values)
    Y_target.append(np.exp(test_pred)-1)
    print('fold finish after', time.time()-modelstart)
Y_target = np.array(Y_target)

#submit
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = Y_target.mean(axis=0)
sub.to_csv('simple_r_value_filter_result.csv', index=False)
### Feature Importance ###
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(lgb_clf, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()






