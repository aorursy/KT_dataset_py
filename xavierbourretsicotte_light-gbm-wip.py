import os
import numpy as np
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

#Light GBM
import lightgbm as lgb
os.listdir('../input/kernel-for-file-processing-2')
# Extract target values and Ids
cat_cols = ['channelGrouping',
            'device.browser',
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent','trafficSource.adContent',
       #'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.gclId',
       #'trafficSource.adwordsClickInfo.isVideoAd',
       #'trafficSource.adwordsClickInfo.page',
       #'trafficSource.adwordsClickInfo.slot', #Drop as only 3 values and always poor
       'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source',
        #Interaction columns
        'geoNetwork.city+geoNetwork.networkDomain',
        'device.operatingSystem+geoNetwork.networkDomain',
        'device.operatingSystem+geoNetwork.city', 
        'channelGrouping+geoNetwork.networkDomain',
        'geoNetwork.city+trafficSource.source',
        'geoNetwork.networkDomain+trafficSource.source',
        'geoNetwork.networkDomain+trafficSource.referralPath',
        'geoNetwork.networkDomain+trafficSource.medium',
        'geoNetwork.city+trafficSource.medium',
        'geoNetwork.city+geoNetwork.country',
        #Time columns (categorical)
        # '_dayofweek', '_year', '_local_hourofday' #These are time related but actually integer  categories
           ]

to_drop = ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.isVideoAd',
          'trafficSource.adwordsClickInfo.page','trafficSource.adwordsClickInfo.slot']

num_cols = ['visitNumber', 'totals.bounces', 'totals.hits',
            'totals.newVisits', 'totals.pageviews', 
            '_prev_totals.bounces_1', '_prev_totals.bounces_2',
           '_next_totals.bounces_1', '_next_totals.bounces_2',
           '_prev_totals.hits_1', '_prev_totals.hits_2', '_next_totals.hits_1',
           '_next_totals.hits_2', '_prev_totals.pageviews_1',
           '_prev_totals.pageviews_2', '_next_totals.pageviews_1',
           '_next_totals.pageviews_2', '_prev__local_hourofday_1',
           '_prev__local_hourofday_2', '_next__local_hourofday_1',
           '_next__local_hourofday_2', '_difference_first_last',
           '_time_since_first_visit', 'visitNumber_12H', 'visitNumber_7D',
           'visitNumber_30D' ]


visitStartTime = ['visitStartTime']

time_shift_cols = ['_time_since_last_visit', '_time_since_last_visit_2',
       '_time_to_next_visit', '_time_to_next_visit_2']
        #[ '_time_since_first_visit' ] # '_difference_first_last'] #These are timedelta format which require additional steps

time_cols = [ '_local_hourofday' , '_dayofweek', '_year']


ID_cols = ['date', 'fullVisitorId', 'sessionId', 'visitId']

target_col = ['totals.transactionRevenue']


%%time
#del train_df
#del test_df

train_df = pd.read_pickle('../input/kernel-for-file-processing-2/train_flat_FE_CAT_LE.pkl')
test_df = pd.read_pickle('../input/kernel-for-file-processing-2/test_flat_FE_CAT_LE.pkl')

#train_df['_time_since_last_visit'] = pd.to_numeric(train_df['_time_since_last_visit'])
#test_df['_time_since_last_visit'] = pd.to_numeric(test_df['_time_since_last_visit'])

train_df.drop(to_drop, axis = 1, inplace = True)
test_df.drop(to_drop, axis = 1, inplace = True)
#Time features
train_df['_dayofweek'] = train_df['visitStartTime'].dt.dayofweek
train_df['_year'] = train_df['visitStartTime'].dt.year

test_df['_dayofweek'] = test_df['visitStartTime'].dt.dayofweek
test_df['_year'] = test_df['visitStartTime'].dt.year
train_df.groupby('fullVisitorId').count().info()
%%time
#Numeric as float
for n in [num_cols + time_cols]:
    train_df[n] = train_df[n].fillna(0).astype('int')
    test_df[n] = test_df[n].fillna(0).astype('int')

#train totals.transactionRevenue
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0).astype('float')

#visitStartTime
for v in time_shift_cols:
    train_df[v] = pd.to_numeric(train_df[v]) / 1e9 # in seconds
    test_df[v] = pd.to_numeric(test_df[v]) / 1e9

#Index
train_idx = train_df['fullVisitorId']
test_idx = test_df['fullVisitorId']

#Targets
train_target = np.log1p(train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum())
train_y = np.log1p(train_df["totals.transactionRevenue"])

#Datasets
train_X = train_df[cat_cols + num_cols + time_cols + time_shift_cols]
test_X = test_df[cat_cols + num_cols + time_cols + time_shift_cols]


print(train_X.shape)
print(test_X.shape)
train_X.info()
predictions_train = pd.DataFrame(data = {'fullVisitorId':train_df['fullVisitorId'], 
                                         'sessionId':train_df['sessionId'], 
                                         'visitId':train_df['visitId'],
                                         'index':train_df.index,
                                         'totals.transactionRevenue':np.log1p(train_df['totals.transactionRevenue']),
                                         'predictedRevenue':np.nan})

predictions_test = pd.DataFrame(data = {'fullVisitorId':test_df['fullVisitorId'], 
                                         'sessionId':test_df['sessionId'], 
                                         'visitId':test_df['visitId'],
                                         'index':test_df.index, 
                                         'predictedRevenue':np.nan})

from sklearn.model_selection import GroupKFold

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
from lightgbm import LGBMRegressor

#Initialize LGBM
gbm = LGBMRegressor(objective = 'regression', 
                     boosting_type = 'gbdt', 
                     metric = 'rmse',
                     n_estimators = 10000, #10000
                     num_leaves = 30,
                     learning_rate = 0.01, #0.01
                     bagging_fraction = 0.9,#0.8
                     feature_fraction = 0.3,#.3
                     #bagging_seed = 42,
                     #max_depth = 12, #-1 
                     #categorical_feature = [train_df[cat_cols].columns.get_loc(c) for c in cat_cols ] ,
                     #cat_smooth = 20
                    )
#predictions_train.info()
predictions_train.groupby('fullVisitorId').count().info()
%%time
#Initilization
all_K_fold_results = []
#kf = KFold(n_splits=5, shuffle = True)
folds = get_folds(df=train_df, n_splits=5)
k = 0


for fold_, (d, v) in enumerate(folds):
    dev_index, val_index  = train_X.index[d], train_X.index[v]
    X_dev, X_val = train_X.loc[dev_index], train_X.loc[val_index]
    y_dev, y_val = train_y[dev_index], train_y[val_index]
    
    #Fit the model
    model = gbm.fit(X_dev,y_dev, eval_set=[(X_val, y_val)],verbose = 100, 
                    eval_metric = 'rmse', early_stopping_rounds = 100) #100
    
    #Predict out of fold 
    predictions_train.loc[val_index, 'predictedRevenue'] = gbm.predict(X_val, num_iteration= model.best_iteration_).copy()
    predictions_train[predictions_train['predictedRevenue'] < 0]['predictedRevenue'] = 0
    print(predictions_train.groupby('fullVisitorId').count().info())
    
    #Predict on train using all train for each fold
    sub_prediction_train = pd.Series(data = gbm.predict(train_X, num_iteration= model.best_iteration_)).copy()
    sub_prediction_train[sub_prediction_train<0] = 0
    predictions_train['Predictions_{}'.format(k)] = sub_prediction_train.values.copy()
    print(predictions_train.groupby('fullVisitorId').count().info())
    
    #Predict on test set based on current fold model. Average results
    sub_prediction = pd.Series(data = gbm.predict(test_X, num_iteration= model.best_iteration_))
    sub_prediction[sub_prediction<0] = 0
    predictions_test['Predictions_{}'.format(k)] = sub_prediction.copy()
    k += 1 #increase by 1
    
    #Save current fold values
    fold_results = {'best_iteration_' : model.best_iteration_, 
                   'best_score_' : model.best_score_['valid_0']['rmse'], 
                   'evals_result_': model.evals_result_['valid_0']['rmse'],
                   'feature_importances_' : model.feature_importances_}

    all_K_fold_results.append(fold_results.copy())
    

#Save results
results = pd.DataFrame(all_K_fold_results)
predictions_test['average_predictions'] = predictions_test.iloc[:,-5:].mean(axis = 1).copy()


def RMSE_log_sum(pred, df):
    #set negative values to zero
    pred[pred < 0] = 0
    
    #Build new dataframe
    pred_df = pd.DataFrame(data = {'fullVisitorId': df['fullVisitorId'].values, 
                                       'transactionRevenue': df['totals.transactionRevenue'].values,
                                      'predictedRevenue':np.expm1(pred) })
    #Compute sum
    pred_df = pred_df.groupby('fullVisitorId').sum().reset_index()

    mse_log_sum = mean_squared_error( np.log1p(pred_df['transactionRevenue'].values), 
                             np.log1p(pred_df['predictedRevenue'].values)  )

    #print('log (sum + 1): ',np.sqrt(mse_log_sum))
    return np.sqrt(mse_log_sum)


def save_submission(pred_test, test_df, file_name):
    #Zero negative predictions
    pred_test[pred_test < 0] = 0
    
    #Create temporary dataframe
    sub_df = pd.DataFrame(data = {'fullVisitorId':test_df['fullVisitorId'], 
                             'predictedRevenue':np.expm1(pred_test)})
    sub_df = sub_df.groupby('fullVisitorId').sum().reset_index()
    sub_df.columns = ['fullVisitorId', 'predictedLogRevenue']
    sub_df['predictedLogRevenue'] = np.log1p(sub_df['predictedLogRevenue'])
    sub_df.to_csv(file_name, index = False)

    
def visualize_results(results):
#Utility function to plot fold loss and best model feature importance
    plt.figure(figsize=(16, 12))

    #----------------------------------------
    # Plot validation loss
    plt.subplot(2,2,1)

    for K in range(results.shape[0]):
        plt.plot(np.arange(len(results.evals_result_[K])), results.evals_result_[K], label = 'fold {}'.format(K))

    plt.xlabel('Boosting iterations')
    plt.ylabel('RMSE')
    plt.title('Validation loss vs boosting iterations')
    plt.legend()

    #----------------------------------------
    # Plot box plot of RMSE
    plt.subplot(2, 2, 2)    
    scores = results.best_score_
    plt.boxplot(scores)
    rmse_mean = np.mean(scores)
    rmse_std = np.std(scores)
    plt.title('RMSE Mean:{:.3f} Std: {:.4f}'.format(rmse_mean,rmse_std ))
    
    #----------------------------------------
    # Plot feature importance
    #feature_importance = results.sort_values('best_score_').feature_importances_[0]
    df_feature_importance = pd.DataFrame.from_records(results.feature_importances_)
    feature_importance = df_feature_importance.mean()
    std_feature_importance = df_feature_importance.std()
    
    # make importances relative to max importance
    #feature_importance = 100.0 * (mean_feature_importance / mean_feature_importance.sum())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(2, 1, 2)
    plt.bar(pos, feature_importance[sorted_idx], align='center', yerr = std_feature_importance)
    xlabels = [ test_X.columns.values[i] for i in sorted_idx]
    plt.xticks(pos, xlabels, rotation = 90)
    plt.xlabel('Feature')
    plt.ylabel('Avg Importance score')
    plt.title('Mean Feature Importance over K folds') 
    
    plt.show()
print('Session level CV score: ', np.mean(results.best_score_))
#print('Session level CV score (all data):', np.sqrt(mean_squared_error(predictions_train['totals.transactionRevenue'], predictions_train['predictedRevenue'])))
print('User level CV score: ', RMSE_log_sum(predictions_train['predictedRevenue'],train_df))
print(predictions_train.groupby('fullVisitorId').count().info())
visualize_results(results)
train_df['visitStartTime'] = pd.to_datetime(train_df['visitStartTime'])

error_df = pd.DataFrame(data = {'visitStartTime':train_df['visitStartTime'],'fullVisitorId':train_df['sessionId'], 
                                'True_log_revenue' : np.log1p(train_df['totals.transactionRevenue']), 
                                'Predicted_log_revenue':predictions_train['predictedRevenue']  })

error_df['Difference'] = error_df['True_log_revenue'] - error_df['Predicted_log_revenue']
error_df['True_is_non_zero'] = error_df['True_log_revenue'] > 0
#temp_df.columns = ['fullVisitorId', 'predictedLogRevenue']
#sub_df['predictedLogRevenue'] = np.log1p(sub_df['predictedLogRevenue'])
#sub_df.to_csv(file_name, index = False)
error_df.head(5).sort_values('True_log_revenue')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (20,7))


sns.distplot(error_df[error_df['True_is_non_zero'] == False]['True_log_revenue'], ax = ax1, label = 'true')
sns.distplot(error_df[error_df['True_is_non_zero'] == False ]['Predicted_log_revenue'], ax = ax1, label = 'pred')
ax1.legend()
ax1.set_ylim(0,.1)
ax1.set_xlabel('Log revenue (session)')
ax1.set_title('Distribution of log revenues for sessions with zero true revenue ')

sns.distplot(error_df[error_df['True_is_non_zero'] == True]['True_log_revenue'], ax = ax2, label = 'true')
sns.distplot(error_df[error_df['True_is_non_zero'] == True ]['Predicted_log_revenue'], ax = ax2, label = 'pred')
ax2.legend()
ax2.set_ylim(0,.5)
ax2.set_xlabel('Log revenue (session)')
ax2.set_title('Distribution of log revenues for sessions with non zero true revenue ')

plt.show()
sorted_non_zero = error_df[error_df['True_is_non_zero'] == True].sort_values('visitStartTime')


plt.figure(figsize = (20,15))
plt.subplot(2,2,1)
plt.plot(sorted_non_zero.visitStartTime, sorted_non_zero.True_log_revenue , label = 'True')
plt.plot(sorted_non_zero.visitStartTime, sorted_non_zero.Predicted_log_revenue , alpha = .5, label = 'Pred')
plt.title('Log revenue over time (non zero true sessions only)')
plt.legend()
plt.xlabel('Time: sessions')

plt.subplot(2,2,2)
daily_error_non_zero_df = sorted_non_zero.set_index('visitStartTime', drop = True).resample('D').mean()
plt.plot(daily_error_non_zero_df.index, daily_error_non_zero_df.True_log_revenue , label = 'True')
plt.plot(daily_error_non_zero_df.index, daily_error_non_zero_df.Predicted_log_revenue , label = 'Pred')
plt.title('Daily average log revenue (non zero true sessions only)')

plt.subplot(2,2,3)
weekly_error_df = error_df.set_index('visitStartTime', drop = True).resample('W').mean()
plt.plot(weekly_error_df.index, weekly_error_df.True_log_revenue , label = 'True')
plt.plot(weekly_error_df.index, weekly_error_df.Predicted_log_revenue , label = 'Pred')
plt.title('Weekly average log revenue (all session)')


plt.subplot(2,2,4)
daily_error_df = error_df.set_index('visitStartTime', drop = True).resample('D').mean()
plt.plot(daily_error_df.index, daily_error_df.True_log_revenue , label = 'True')
plt.plot(daily_error_df.index, daily_error_df.Predicted_log_revenue , label = 'Pred')
plt.title('Daily average log revenue (all session)')

plt.legend()
plt.show()
sorted_non_zero = error_df[error_df['True_is_non_zero'] == True].sort_values('visitStartTime')
sorted_zero = error_df[error_df['True_is_non_zero'] == False].sort_values('visitStartTime')


plt.figure(figsize = (20,5))
plt.subplot(1,3,1)
ts_error_df = error_df.set_index('visitStartTime', drop = True)
difference_rev_df = error_df.sort_values('visitStartTime')
plt.plot(error_df.visitStartTime, error_df.Difference , label = 'True - predicted', color = 'grey')
plt.title('Train - Pred (log rev) for all sessions')

plt.subplot(1,3,2)
plt.plot(sorted_non_zero.visitStartTime, sorted_non_zero.Difference , label = 'True - predicted',
         color = 'grey')
plt.title('Train - Pred for non zero sessions only')

plt.subplot(1,3,3)
plt.plot(sorted_zero.visitStartTime, sorted_zero.Difference,
         color = 'grey')
plt.title('Train - Pred for zero sessions only')

plt.legend()
plt.show()
sns.jointplot(x="True_log_revenue", y="Predicted_log_revenue", data=sorted_non_zero)
display('Joint distribution of log rev for non zero sessions only')

plt.show()
save_submission(predictions_test['average_predictions'], test_df, 'submission.csv')
predictions_train.to_csv('level_1_train_output.csv')
predictions_test.to_csv('level_1_test_output.csv')
feature_importance = results.feature_importances_.mean()
sorted_idx = np.argsort(feature_importance)
feature_importance[sorted_idx]
feature_names = [ test_X.columns.values[i] for i in sorted_idx]
importance_df = pd.DataFrame({'features_names': feature_names, 'importance':feature_importance[sorted_idx] })
importance_df.sort_values('importance', ascending = False)['features_names'][0:20].values