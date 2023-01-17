# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import time

from sklearn.model_selection import train_test_split

import xgboost



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



from sklearn.metrics import log_loss

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def str_to_num(string):

    return int(string.split(" ")[1])



train=pd.read_csv('../input/train.csv', converters={'location':str_to_num})

test=pd.read_csv('../input/test.csv', converters={'location':str_to_num})

event=pd.read_csv('../input/event_type.csv', converters={'event_type':str_to_num})

log_feature=pd.read_csv('../input/log_feature.csv', converters={'log_feature':str_to_num})

severity=pd.read_csv('../input/severity_type.csv', converters={'severity_type':str_to_num})

resource=pd.read_csv('../input/resource_type.csv', converters={'resource_type':str_to_num})



sample=pd.read_csv('../input/sample_submission.csv')
# merge train and test set for now

traintest=train.append(test)



# create resource one-hot data per id

resource_by_id=pd.get_dummies(resource,columns=['resource_type'])

resource_by_id=resource_by_id.groupby(['id']).sum().reset_index(drop=False)



# create event one-hot data per id

event_by_id=pd.get_dummies(event,columns=['event_type'])

event_by_id=event_by_id.groupby(['id']).sum().reset_index(drop=False)
log_feature_dict={}



for row in log_feature.itertuples():

    if row.id not in log_feature_dict:

        log_feature_dict[row.id]={}

    if row.log_feature not in log_feature_dict[row.id]:

        log_feature_dict[row.id][row.log_feature]=row.volume



colnames=['id']

for i in range(1,387):

    colnames.append('log_feature_'+str(i))



log_feature_by_id_np=np.zeros((18552,387))

count=0

for key, feature_dict in log_feature_dict.items():

    log_feature_by_id_np[count, 0]=np.int(key)

    for feature, volume in feature_dict.items():

        log_feature_by_id_np[count, feature]=np.int(volume)

    count+=1

log_feature_by_id=pd.DataFrame(data=log_feature_by_id_np, columns=colnames, dtype=np.int)
# Merge datasets together for ml input dataframe



traintest=traintest.merge(right=severity, on='id')

print(traintest.shape)



traintest=traintest.merge(right=resource_by_id, on='id')

print(traintest.shape)



traintest=traintest.merge(right=event_by_id, on='id')

print(traintest.shape)



traintest=traintest.merge(right=log_feature_by_id, on='id')

print(traintest.shape)
# Seperate the traintest dataframe into train and test input dataframes

train_input=traintest.loc[0:train.shape[0]-1].copy()

print("train_input shape is", train_input.shape)



test_input=traintest.loc[train.shape[0]::].copy()

print("test_input shape is", test_input.shape)
y=train_input.fault_severity

train_input.drop(['fault_severity'], axis=1, inplace=True)

test_input.drop(['fault_severity'], axis=1, inplace=True)
# sum feature for resource and event

resource_cols=train_input.columns[train_input.columns.str.find('resource')==0].tolist()

event_cols=train_input.columns[train_input.columns.str.find('event')==0].tolist()



train_input['resource_sum']=train_input[resource_cols].sum(axis=1)

train_input['event_sum']=train_input[event_cols].sum(axis=1)



test_input['resource_sum']=test_input[resource_cols].sum(axis=1)

test_input['event_sum']=test_input[event_cols].sum(axis=1)
# frequency feature for location



# create a dataframe with with the location and the location frequency as features

location_frequency=traintest.location.value_counts()

location_frequency.name='location_frequency'

location_frequency=pd.DataFrame(location_frequency).reset_index()

location_frequency.rename(columns={'index':'location'}, inplace=True)



# merge this location frequency dataframe with the training and testing ML input data on location

train_input=train_input.merge(right=location_frequency, on='location', how='left')

test_input=test_input.merge(right=location_frequency, on='location', how='left')
# pattern feature for log feature

log_feature_cols=traintest.columns[traintest.columns.str.find('log_feature')==0].tolist()

traintest_log_feature=traintest[log_feature_cols].copy()



mask=(traintest_log_feature>0)

traintest_log_feature.where(mask, other=0, inplace=True)



mask=(traintest_log_feature<1)

traintest_log_feature.where(mask, other=1, inplace=True)



traintest_log_feature['log_feature_pattern_raw']= traintest_log_feature.apply(lambda x: ''.join(x.astype(str)), axis=1)





log_feature_pattern_df=pd.DataFrame(traintest_log_feature.log_feature_pattern_raw.drop_duplicates())

log_feature_pattern_df.reset_index(inplace=True)

log_feature_pattern_df.rename(columns={'index':'log_feature_pattern_id'}, inplace=True)



# merge log_feature_pattern_df back to traintest_log_feature on log_feature_pattern_raw

traintest_log_feature=traintest_log_feature.merge(right=log_feature_pattern_df, on='log_feature_pattern_raw', how='left')



# finally insert the log_feature_pattern_id column into input dataframes as new feature

train_input['log_feature_pattern_id']=traintest_log_feature.loc[0:train.shape[0]-1, 'log_feature_pattern_id'].values

test_input['log_feature_pattern_id']=traintest_log_feature.loc[train.shape[0]::]['log_feature_pattern_id'].values
# remove id column

train_input.drop(['id'], axis=1, inplace=True)

test_input.drop(['id'], axis=1, inplace=True)



train_input_fe1=train_input.copy()

test_input_fe1=test_input.copy()



train_input.shape
train_input=train_input_fe1.copy()

test_input=test_input_fe1.copy()

print(train_input.shape)
# pattern feature for resource

resource_cols=traintest.columns[traintest.columns.str.find('resource')==0].tolist()

traintest_resource=traintest[resource_cols].copy()



traintest_resource['resource_pattern_raw']= traintest_resource.apply(lambda x: ''.join(x.astype(str)), axis=1)





resource_pattern_df=pd.DataFrame(traintest_resource.resource_pattern_raw.drop_duplicates())

resource_pattern_df.reset_index(inplace=True)

resource_pattern_df.rename(columns={'index':'resource_pattern_id'}, inplace=True)



# merge resource_pattern_df back to traintest_resource on resource_pattern_raw

traintest_resource=traintest_resource.merge(right=resource_pattern_df, on='resource_pattern_raw', how='left')



# finally insert the resource_pattern_id column into input dataframes as new feature

train_input['resource_pattern_id']=traintest_resource.loc[0:train.shape[0]-1, 'resource_pattern_id'].values

test_input['resource_pattern_id']=traintest_resource.loc[train.shape[0]::]['resource_pattern_id'].values
# log feature indicator sum

# improve private LB, decrease public LB 

traintest_log_feature['log_feat_ind_sum']=traintest_log_feature[log_feature_cols].sum(axis=1)

train_input['log_feat_ind_sum']=traintest_log_feature.loc[0:train.shape[0]-1, 'log_feat_ind_sum'].values

test_input['log_feat_ind_sum']=traintest_log_feature.loc[train.shape[0]::]['log_feat_ind_sum'].values
traintest_input=train_input.append(test_input)



# create a dataframe with with the logfeat_pat_freq and the logfeat_pat_freq frequency as features

logfeat_pat_freq=traintest_input.log_feature_pattern_id.value_counts()

logfeat_pat_freq.name='logfeat_pat_freq'

logfeat_pat_freq=pd.DataFrame(logfeat_pat_freq).reset_index()

logfeat_pat_freq.rename(columns={'index':'log_feature_pattern_id'}, inplace=True)



# merge this logfeat_pat_freq frequency dataframe with the training and testing ML input data on logfeat_pat_freq



traintest_input=traintest_input.merge(right=logfeat_pat_freq, on='log_feature_pattern_id', how='left')



train_input=traintest_input.loc[0:train.shape[0]-1].copy()

test_input=traintest_input.loc[train.shape[0]::].copy()
severity_frequency=traintest.severity_type.value_counts()

severity_frequency.name='severity_frequency'

severity_frequency=pd.DataFrame(severity_frequency).reset_index()

severity_frequency.rename(columns={'index':'severity_type'}, inplace=True)



# merge this severity frequency dataframe with the training and testing ML input data on severity

train_input=train_input.merge(right=severity_frequency, on='severity_type', how='left')

test_input=test_input.merge(right=severity_frequency, on='severity_type', how='left')
train_input.head(5)
def cross_validate_xgb(param, x_train, y_train, kf, verbose=True, verbose_eval=50):

    start_time=time.time()

    nround=[]

    # the prediction matrix need to contains 3 columns, one for the probability of each class

    train_pred = np.zeros((x_train.shape[0],3))

    

    # use the k-fold object to enumerate indexes for each training and validation fold

    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)):

        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]

        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

        

        d_train = xgboost.DMatrix(x_train_kf, y_train_kf)

        d_val=xgboost.DMatrix(x_val_kf, y_val_kf)



        watchlist= [(d_train, "train"), (d_val, 'val')]

        bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=3000, early_stopping_rounds=100,

                            evals=watchlist, verbose_eval=verbose_eval)        

        

        y_val_kf_preds=bst.predict(d_val, ntree_limit=bst.best_ntree_limit)

        nround.append(bst.best_ntree_limit)

        

        train_pred[val_index] += y_val_kf_preds

        

        fold_cv = log_loss(y_val_kf.values, y_val_kf_preds)

        if verbose:

            print('fold cv {} log_loss score is {:.6f}'.format(i, fold_cv))

        

    cv_score = log_loss(y_train, train_pred)

    

    if verbose:

        print('cv log_loss score is {:.6f}'.format(cv_score))

    

    end_time = time.time()

    print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

    return bst, np.array(nround).mean()
xgb_params = {

    "objective" : "multi:softprob",

    "num_class" : 3,

    "tree_method" : "hist",

    "eval_metric" : "mlogloss",

    "nthread": 4,

    "seed" : 0,

    'silent': 1,



    "eta":0.05,  # default 0.3

    "max_depth" : 5, # default 6

    "subsample" : 0.8, # default 1

    "colsample_bytree" : 0.6, # default 1

    "gamma": 0.5

}



kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)





print('Start training...')

bst, n_round=cross_validate_xgb(xgb_params, train_input, y, kf)

d_train=xgboost.DMatrix(train_input, y)

full_n_round=int(n_round*1.2)

watchlist=[(d_train, "train")]

bst = xgboost.train(params=xgb_params, dtrain=d_train, evals=watchlist, num_boost_round=full_n_round, verbose_eval=50) 
d_test=xgboost.DMatrix(test_input)

test_xgb_preds=bst.predict(d_test)



xgb_submission=sample.copy()

xgb_submission.predict_0=test_xgb_preds[:,0]

xgb_submission.predict_1=test_xgb_preds[:,1]

xgb_submission.predict_2=test_xgb_preds[:,2]

xgb_submission.to_csv('xgb_submission_feateng_add_full1.gz', compression='gzip', index=False)