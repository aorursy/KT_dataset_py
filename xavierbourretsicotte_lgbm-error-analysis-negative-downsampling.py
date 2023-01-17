import os
import numpy as np
import pandas as pd
import time
import warnings
import datetime

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#lgm and graph viz
import graphviz 
import lightgbm as lgb

warnings.filterwarnings('ignore')

os.listdir('../input/kernel-for-file-processing-2')
# Extract target values and Ids
cat_cols = ['channelGrouping','device.browser',
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
       'trafficSource.source'  ]

to_drop = ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.isVideoAd',
          'trafficSource.adwordsClickInfo.page','trafficSource.adwordsClickInfo.slot']

num_cols = ['visitNumber', 'totals.bounces', 'totals.hits',
            'totals.newVisits', 'totals.pageviews', ]

#interaction_cols = ['totals.hits / totals.pageviews']

visitStartTime = ['visitStartTime']

time_cols = ['_dayofweek', '_monthofyear', '_dayofyear', '_local_hourofday', '_time_since_last_visit']

ID_cols = ['date', 'fullVisitorId', 'sessionId', 'visitId']

target_col = ['totals.transactionRevenue']


%%time

train_df = pd.read_pickle('../input/kernel-for-file-processing-2/train_flat_FE.pkl')
test_df = pd.read_pickle('../input/kernel-for-file-processing-2/test_flat_FE.pkl')

train_df.drop(to_drop, axis = 1, inplace = True)
test_df.drop(to_drop, axis = 1, inplace = True)
#Time features
train_df['_dayofweek'] = train_df['visitStartTime'].dt.dayofweek
train_df['_monthofyear'] = train_df['visitStartTime'].dt.month
train_df['_dayofyear'] = train_df['visitStartTime'].dt.dayofyear
#train_df['_dayofmonth'] = train_df['visitStartTime'].dt.day

test_df['_dayofweek'] = test_df['visitStartTime'].dt.dayofweek
test_df['_monthofyear'] = test_df['visitStartTime'].dt.month
test_df['_dayofyear'] = test_df['visitStartTime'].dt.dayofyear
#test_df['_dayofmonth'] = test_df['visitStartTime'].dt.day
%%time

#Numeric as float ##
for n in [num_cols + time_cols]:
    train_df[n] = train_df[n].fillna(0).astype('int')
    test_df[n] = test_df[n].fillna(0).astype('int')
    
#Time as float
train_df['_time_since_last_visit'] = pd.to_numeric(train_df['_time_since_last_visit'])
test_df['_time_since_last_visit'] = pd.to_numeric(test_df['_time_since_last_visit'])
%%time

train_df[cat_cols] = train_df[cat_cols].fillna('unknown')
test_df[cat_cols] = test_df[cat_cols].fillna('unknown')
test_df['device.isMobile'] = test_df['device.isMobile'].astype('int')

#Factorize cats
for f in cat_cols:
    train_df[f], indexer = pd.factorize(train_df[f])
    test_df[f] = indexer.get_indexer(test_df[f])
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0).astype('float')

#Index
train_idx = train_df['fullVisitorId']
test_idx = test_df['fullVisitorId']

#Targets
train_target = np.log1p(train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum())
train_y = np.log1p(train_df["totals.transactionRevenue"])

#Datasets
train_X = train_df[cat_cols + num_cols + time_cols].copy()
test_X = test_df[cat_cols + num_cols + time_cols ].copy()

print(train_X.shape)
print(test_X.shape)
from lightgbm import LGBMRegressor

#Initialize LGBM
gbm = LGBMRegressor(objective = 'regression', 
                     boosting_type = 'gbdt', 
                     metric = 'rmse',
                     n_estimators = 10000, #10000
                     num_leaves = 10, #10
                     learning_rate = 0.08, #0.01
                     bagging_fraction = 0.9,
                     feature_fraction = 0.3,
                     bagging_seed = 0,
                     max_depth = 10,
                                         )

%%time
#Initilization
all_K_fold_results = []
kf = KFold(n_splits=5, shuffle = True)
oof_preds = np.zeros(train_X.shape[0])
sub_preds = np.zeros(test_X.shape[0])


for dev_index, val_index in kf.split(train_X):
    X_dev, X_val = train_X.iloc[dev_index], train_X.iloc[val_index]
    y_dev, y_val = train_y[dev_index], train_y[val_index]

    #Fit the model
    model = gbm.fit(X_dev,y_dev, eval_set=[(X_val, y_val)],verbose = 100, 
                    eval_metric = 'rmse', early_stopping_rounds = 100) ##100
    
    #Predict out of fold 
    oof_preds[val_index] = gbm.predict(X_val, num_iteration= model.best_iteration_)
    
    oof_preds[oof_preds < 0] = 0
    
    #Predict on test set based on current fold model. Average results
    sub_prediction = gbm.predict(test_X, num_iteration= model.best_iteration_) / kf.n_splits
    sub_prediction[sub_prediction<0] = 0
    sub_preds = sub_preds + sub_prediction
    
    #Save current fold values
    fold_results = {'best_iteration_' : model.best_iteration_, 
                   'best_score_' : model.best_score_['valid_0']['rmse'], 
                   'evals_result_': model.evals_result_['valid_0']['rmse'],
                   'feature_importances_' : model.feature_importances_}

    all_K_fold_results.append(fold_results.copy())
    

results = pd.DataFrame(all_K_fold_results)



def RMSE_log_sum(pred_val, val_df):
    #set negative values to zero
    pred_val[pred_val < 0] = 0
    
    #Build new dataframe
    val_pred_df = pd.DataFrame(data = {'fullVisitorId': val_df['fullVisitorId'].values, 
                                       'transactionRevenue': val_df['totals.transactionRevenue'].values,
                                      'predictedRevenue':np.expm1(pred_val) })
    #Compute sum
    val_pred_df = val_pred_df.groupby('fullVisitorId').sum().reset_index()

    mse_log_sum = mean_squared_error( np.log1p(val_pred_df['transactionRevenue'].values), 
                             np.log1p(val_pred_df['predictedRevenue'].values)  )

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
    xlabels = [ train_X.columns.values[i] for i in sorted_idx]
    plt.xticks(pos, xlabels, rotation = 90)
    plt.xlabel('Feature')
    plt.ylabel('Avg Importance score')
    plt.title('Mean Feature Importance over K folds') 
    
    plt.show()
print('Session level CV score: ', np.mean(results.best_score_))
print('User level CV score: ', RMSE_log_sum(oof_preds, train_df))
import graphviz 
dot_data = lgb.create_tree_digraph(model, tree_index = 1,show_info=['split_gain'])

graph = graphviz.Source(dot_data)  
graph 
error_df = pd.DataFrame(data = {'visitStartTime':train_df['visitStartTime'],'fullVisitorId':train_df['sessionId'], 
                                'True_log_revenue' : np.log1p(train_df['totals.transactionRevenue']), 
                                'Predicted_log_revenue':oof_preds  })

error_df['Difference'] = error_df['True_log_revenue'] - error_df['Predicted_log_revenue']
error_df['True_is_non_zero'] = error_df['True_log_revenue'] > 0
#temp_df.columns = ['fullVisitorId', 'predictedLogRevenue']
#sub_df['predictedLogRevenue'] = np.log1p(sub_df['predictedLogRevenue'])
#sub_df.to_csv(file_name, index = False)
error_df.sort_values('visitStartTime').head(10)
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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,7))

sns.distplot(error_df['True_log_revenue'], ax = ax, label = 'true')
sns.distplot(error_df['Predicted_log_revenue'], ax = ax, label = 'pred')
ax.legend()
ax.set_ylim(0,.04)
ax.set_xlabel('Log revenue (session)')
ax.set_title('Distribution of log revenues for all sessions')

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

%%time
all_K_fold_results = []

#Downsampling preparation
idx_pos = train_df[train_df['totals.transactionRevenue'] > 0].index
idx_neg = train_df[train_df['totals.transactionRevenue'] == 0].index
train_pos_X, train_pos_y = train_X.loc[idx_pos], train_y[idx_pos]
train_neg_X, train_neg_y = train_X.loc[idx_neg], train_y[idx_neg]

oof_preds = [] 
sub_preds = np.zeros(test_X.shape[0])
size_dev = round( (4/5) * len(idx_pos) )  #4/5 for dev and 1/5 for val

#Indices
idx_dev_pos = np.random.choice(idx_pos, size = size_dev, replace = False)
idx_val_pos = np.setdiff1d(idx_pos,idx_dev_pos)

idx_dev_neg = np.random.choice(idx_neg, size = size_dev, replace = False)
idx_val_neg = np.random.choice(idx_neg, size = len(idx_val_pos), replace = False)

#Datasets
X_dev_pos, X_val_pos = train_pos_X.loc[idx_dev_pos], train_pos_X.loc[idx_val_pos]
y_dev_pos, y_val_pos = train_pos_y[idx_dev_pos], train_pos_y[idx_val_pos]

X_dev_neg, X_val_neg = train_neg_X.loc[idx_dev_neg], train_neg_X.loc[idx_val_neg]
y_dev_neg, y_val_neg = train_neg_y[idx_dev_neg], train_neg_y[idx_val_neg]

#Concatenate
X_dev, X_val = pd.concat([X_dev_pos, X_dev_neg], axis = 0), pd.concat([X_val_pos, X_val_neg], axis = 0)
y_dev, y_val = pd.concat([y_dev_pos, y_dev_neg]), pd.concat([y_val_pos, y_val_neg])

#Fit the model
model = gbm.fit(X_dev,y_dev, eval_set=[(X_val, y_val)],verbose = 100, 
                eval_metric = 'rmse', early_stopping_rounds = 100) ##100

#Predict on val
oof_preds.append( gbm.predict(X_val, num_iteration= model.best_iteration_).copy() )

#Predict on test set based on current fold model. 
sub_prediction = gbm.predict(test_X, num_iteration= model.best_iteration_)
sub_prediction[sub_prediction<0] = 0
sub_preds = sub_preds + sub_prediction #not needed

#Save current fold values
fold_results = {'best_iteration_' : model.best_iteration_, 
               'best_score_' : model.best_score_['valid_0']['rmse'], 
               'evals_result_': model.evals_result_['valid_0']['rmse'],
               'feature_importances_' : model.feature_importances_}

all_K_fold_results.append(fold_results.copy())


results2 = pd.DataFrame(all_K_fold_results)

#Save as array and flatten
oof_preds = np.asarray(oof_preds).flatten()
oof_preds[oof_preds < 0] = 0

print('Session level validation score: ', np.mean(results2.best_score_))
print('User level validation score: ', RMSE_log_sum(oof_preds, train_df.iloc[X_val.index]))
X_val.index
visualize_results(results2)
idx_oof_pred= X_val.index

error_df = pd.DataFrame(data = {'visitStartTime':train_df.loc[idx_oof_pred]['visitStartTime'],
                                'fullVisitorId':train_df.loc[idx_oof_pred]['sessionId'], 
                                'True_log_revenue' : np.log1p(train_df.loc[idx_oof_pred]['totals.transactionRevenue']), 
                                'Predicted_log_revenue':oof_preds  })

error_df['Difference'] = error_df['True_log_revenue'] - error_df['Predicted_log_revenue']
error_df['True_is_non_zero'] = error_df['True_log_revenue'] > 0
error_df.head(10)
Total_SSE = np.square(error_df.Difference).sum()
Zero_SSE = np.square(error_df[error_df['True_is_non_zero'] == False].Difference).sum()
Non_zero_SSE = np.square(error_df[error_df['True_is_non_zero'] == True].Difference).sum() 

print('Total SSE: ',round(Total_SSE), ' 100%'  )
print('Zero true revenue SSE: ', round(Zero_SSE), round(Zero_SSE / Total_SSE , 3) * 100, '%'  )
print('Non zero true revenue SSE: ',round(Non_zero_SSE), round(Non_zero_SSE / Total_SSE, 3) * 100, '%' )
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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,7))

sns.distplot(error_df['True_log_revenue'], ax = ax, label = 'true', bins = 100)
sns.distplot(error_df['Predicted_log_revenue'], ax = ax, label = 'pred', bins = 100)
ax.legend()
ax.set_ylim(0,.1)
ax.set_xlabel('Log revenue (session)')
ax.set_title('Distribution of log revenues for all sessions')

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
sorted_all = error_df.sort_values('visitStartTime')


plt.figure(figsize = (20,5))
plt.subplot(1,3,1)
plt.plot(sorted_all.visitStartTime, sorted_all.Difference , label = 'True - predicted', color = 'grey')
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

#Save and submit
save_submission(sub_preds, test_df, 'submission.csv')

'''%%time

#Downsampling preparation
idx_pos = train_df[train_df['totals.transactionRevenue'] > 0].index
idx_neg = train_df[train_df['totals.transactionRevenue'] == 0].index
train_pos_X, train_pos_y = train_X.loc[idx_pos], train_y[idx_pos]
train_neg_X, train_neg_y = train_X.loc[idx_neg], train_y[idx_neg]

#Initilization
all_K_fold_results = []
kf = KFold(n_splits=5, shuffle = False)

oof_preds = [] #np.zeros(train_pos_X.shape[0] * 2)
sub_preds = np.zeros(test_X.shape[0])


for i, v in kf.split(train_pos_X):
    #Positive samples
    idx_dev_pos, idx_val_pos  = train_pos_X.index[i], train_pos_X.index[v]
    X_dev_pos, X_val_pos = train_pos_X.loc[idx_dev_pos], train_pos_X.loc[idx_val_pos]
    y_dev_pos, y_val_pos = train_pos_y[idx_dev_pos], train_pos_y[idx_val_pos]

    #Negative samples
    idx_dev_neg = np.random.choice(idx_neg, size = len(idx_dev_pos), replace = False)
    idx_val_neg = np.random.choice(idx_neg, size = len(idx_val_pos), replace = False)
    
    X_dev_neg, X_val_neg = train_neg_X.loc[idx_dev_neg], train_neg_X.loc[idx_val_neg]
    y_dev_neg, y_val_neg = train_neg_y[idx_dev_neg], train_neg_y[idx_val_neg]

    #Concatenate
    X_dev, X_val = pd.concat([X_dev_pos, X_dev_neg], axis = 0), pd.concat([X_val_pos, X_val_neg], axis = 0)
    y_dev, y_val = pd.concat([y_dev_pos, y_dev_neg]), pd.concat([y_val_pos, y_val_neg])
    
     #Fit the model
    model = gbm.fit(X_dev,y_dev, eval_set=[(X_val, y_val)],verbose = 100, 
                    eval_metric = 'rmse', early_stopping_rounds = 100) ##100
    
    #Predict out of fold 
    oof_preds.append( gbm.predict(X_val, num_iteration= model.best_iteration_).copy() )
    
    #Predict on test set based on current fold model. Average results
    sub_prediction = gbm.predict(test_X, num_iteration= model.best_iteration_) / kf.n_splits
    sub_prediction[sub_prediction<0] = 0
    sub_preds = sub_preds + sub_prediction

    #Save current fold values
    fold_results = {'best_iteration_' : model.best_iteration_, 
                   'best_score_' : model.best_score_['valid_0']['rmse'], 
                   'evals_result_': model.evals_result_['valid_0']['rmse'],
                   'feature_importances_' : model.feature_importances_}

    all_K_fold_results.append(fold_results.copy())


results = pd.DataFrame(all_K_fold_results)

#Save as array and flatten
oof_preds = np.asarray(oof_preds).flatten()
oof_preds[oof_preds < 0] = 0 '''
