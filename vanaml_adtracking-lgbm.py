import gc
import numpy as np #For numerical computations
import pandas as pd #For data wrangling
import matplotlib.pyplot as plt #For basic plotting
%matplotlib inline  
import os
inputpath = os.listdir("../input")
#It is important to define the new datatypes becuase of the space each of the default ones occupy
define_dtypes = {
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'is_attributed':'uint16',
    'click_id':'uint32'
}
train_data = pd.read_csv(inputpath+'train.csv', nrows = 1000000,usecols = ['ip','app','device','os','channel','click_time','is_attributed'], dtype = define_dtypes)
test_data = pd.read_csv(inputpath+'test.csv',   usecols = ['ip','app','device','os','channel','click_time','click_id'], dtype = define_dtypes)
print(train_data.isnull().sum(axis =0))
print(test_data.isnull().sum(axis =0))
print ('The max and the min days for which the data is collected in train dataset is %s %s' %(train_data['click_time'].min(), train_data['click_time'].max()))
print ('The max and the min days for which the data is collected in test dataset is %s %s' %(test_data['click_time'].min(), test_data['click_time'].max()))
nunique = train_data.nunique(dropna = False)
train_data['click_day'] = pd.to_datetime(train_data['click_time']).dt.day.astype('uint8')
train_data['click_hour'] = pd.to_datetime(train_data['click_time']).dt.hour.astype('uint8')
train_data['click_minute'] = pd.to_datetime(train_data['click_time']).dt.minute.astype('uint8')
train_data = train_data.drop(['click_time'], axis = 1)
print(train_data['is_attributed'].value_counts())
plt.hist(train_data['is_attributed']);
plt.title("Histogram of Target variable")
plt.xlabel = "Target variable"
plt.ylabel = "Frequency percentage"
plt.show()
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    #print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    model = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    n_estimators = model.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    return model 
#Importing lightgbm
import lightgbm as lgb
import sklearn as sk #For shuffling data
sk.utils.shuffle(train_data) #Shuffling train data to split into train and validation sets
#Fitting the  lgbm model to train dataset in order to find the important features
predictors = ['ip','app','device','os','channel','click_day','click_hour']
target = 'is_attributed'
train_df, val_df = np.split(train_data, [int(.95*(len(train_data)))]) 
params = {
        'learning_rate': 0.15,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':99 # because training data is extremely unbalanced 
    }
check_model = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30
                        #verbose_eval=True, 
                        #num_boost_round=500 
                        #categorical_features=categorical
                        )
del train_data
gc.collect()
print("Feature gain/importance...")
gain = check_model.feature_importance('gain')
ft = pd.DataFrame({'feature':check_model.feature_name(), 
                   'split':check_model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(25))

plt.figure()
ft[['feature','split']].head(25).plot(kind='bar', x='feature', y='split', legend=False, figsize=(10, 20))
#plt.gcf().savefig('feature_importance.png')
plt.show()
train_data = pd.read_csv(inputpath+'train.csv', nrows = 20000000,usecols = ['ip','app','device','os','channel','click_time','is_attributed'], dtype = define_dtypes)
test_data = pd.read_csv(inputpath+'test.csv',   usecols = ['ip','app','device','os','channel','click_time','click_id'], dtype = define_dtypes)
Combined_data = train_data.append(test_data)
del train_data
del test_data
gc.collect()
Combined_data.head()
Combined_data['hour'] = pd.to_datetime(Combined_data.click_time).dt.hour.astype('uint8')
Combined_data['day'] = pd.to_datetime(Combined_data.click_time).dt.day.astype('uint8')
Combined_data = Combined_data.drop(['click_time'], axis = 1)
gc.collect()
print('Grouping Combined data by ip-day-hour combination...')
gp = Combined_data[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
Combined_data = Combined_data.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()
print('Grouping Combined data by ip-app combination...')
gp =Combined_data[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
Combined_data = Combined_data.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()
print('Grouping Combined data by ip-app-os combination')
gp =  Combined_data[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
Combined_data = Combined_data.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()
#print("vars and data type: ")
Combined_data.info()
Combined_data['ip_tcount'] = Combined_data['ip_tcount'].astype('uint16')
Combined_data['ip_app_count'] = Combined_data['ip_app_count'].astype('uint16')
Combined_data['ip_app_os_count'] = Combined_data['ip_app_os_count'].astype('uint16')
gc.collect()
test_df = Combined_data[Combined_data['click_id'].isnull() ==False] 
train_df, val_df = np.split(Combined_data[Combined_data['click_id'].isnull() ==True], [int(.95*(len(Combined_data[Combined_data['click_id'].isnull() ==True])))]) 
#Checking the counts to make sure the train, test and validaiton splits are right
print(Combined_data.shape)
print(train_df.shape)
print(test_df.shape)
print(val_df.shape)
#Fitting the  lgbm final model to train and validation datasets
output = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'day', 'ip_app_os_count','ip_app_count','ip_tcount']
#train_df, val_df = np.split(train_data, [int(.95*(len(train_data)))]) #Splitting into train and validation datasets np.split(train_data, [int(.95*(len(train_data)))]) #Splitting into train and validation datasets
train_df, val_df = np.split(Combined_data[Combined_data['click_id'].isnull() ==True], [int(.95*(len(Combined_data[Combined_data['click_id'].isnull() ==True])))]) 
#train_lgb(predictors,output)
params = {
        'learning_rate': 0.15,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':99 # because training data is extremely unbalanced 
    }
check_model = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30
                        #verbose_eval=True, 
                        #num_boost_round=500 
                        #categorical_features=categorical
                        )
Submission = pd.DataFrame()

#Submission['click_id'] = test_df['click_id'].astype('int')
Submission['is_attributed'] = check_model.predict(test_df[predictors])
#Submission.head()
Submission.to_csv('submission.csv',index=False)
#print("done...")