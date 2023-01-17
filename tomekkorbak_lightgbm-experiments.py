import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

!ls -lh ../input
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
}
df_train = pd.read_csv('../input/train.csv', nrows=10**7, dtype=dtypes)
df_test = pd.read_csv('../input/test.csv', dtype=dtypes)
df_train, df_val = train_test_split(df_train, train_size=.95, shuffle=False)
def do_feature_engineering(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    #df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['ip', 'click_time'], axis=1, inplace=True)
    return df
    
df_train = do_feature_engineering(df_train)
df_val = do_feature_engineering(df_val)
df_test = do_feature_engineering(df_test)
target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour']
xgtrain = lgb.Dataset(df_train[predictors].values, label=df_train[target].values,
                      feature_name=predictors,
                      categorical_feature=predictors
)
xgvalid = lgb.Dataset(df_val[predictors].values, label=df_val[target].values,
                     feature_name=predictors,
                     categorical_feature=predictors
)
evals_results = {}
lgb_params = {  # credit for https://www.kaggle.com/aharless/try-pranav-s-r-lgbm-in-python
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
}

bst = lgb.train(lgb_params, 
                xgtrain, 
                valid_sets= [xgvalid], 
                valid_names=['valid'], 
                evals_result=evals_results, 
                num_boost_round=1000,
                early_stopping_rounds=50,
                verbose_eval=10, 
                feval=None
)
n_estimators = bst.best_iteration
print("n_estimators: ", n_estimators)
print("best auc: ", evals_results['valid']['auc'][n_estimators-1])
df_output = pd.DataFrame()
df_output['click_id'] = df_test['click_id']
df_output['is_attributed'] = bst.predict(df_test[predictors])
df_output.to_csv('output.csv', index=False, float_format='%.9f')