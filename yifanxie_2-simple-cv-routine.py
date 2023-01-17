# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import log_loss

from sklearn.metrics import make_scorer

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
start_time=time.time()

log_loss_scorer=make_scorer(log_loss)

rf=RandomForestClassifier(n_estimators=100, random_state=2017, n_jobs=4)



scores=cross_val_score(rf, X=train_input, y=y, cv=3, scoring=log_loss_scorer, verbose=3)



print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))
def cross_validate_sklearn(clf, x_train, y_train, kf, verbose=True):

    start_time=time.time()

    

    # the prediction matrix need to contains 3 columns, one for the probability of each class

    train_pred = np.zeros((x_train.shape[0],3))

    

    # use the k-fold object to enumerate indexes for each training and validation fold

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):

        x_train_kf, x_test_kf = x_train.loc[train_index, :], x_train.loc[test_index, :]

        y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

        

        clf.fit(x_train_kf, y_train_kf)

        y_test_kf_preds=clf.predict_proba(x_test_kf)

        train_pred[test_index] += y_test_kf_preds

        

        fold_cv = log_loss(y_test_kf.values, y_test_kf_preds)

        if verbose:

            print('fold cv {} log_loss score is {:.6f}'.format(i, fold_cv))

        

    cv_score = log_loss(y_train, train_pred)

    

    if verbose:

        print('cv log_loss score is {:.6f}'.format(cv_score))

    

    end_time = time.time()

    print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

    return cv_score
rf=RandomForestClassifier(n_estimators=100, random_state=2017, n_jobs=4)



kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)



cv_score=cross_validate_sklearn(rf, train_input, y, kf)
rf=RandomForestClassifier(n_estimators=200, random_state=2017, n_jobs=4)



kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)



cv_score=cross_validate_sklearn(rf, train_input, y, kf)
start_time=time.time()

rf=RandomForestClassifier(n_estimators=200, random_state=2017, n_jobs=4)



# let's split the data as we did in the simple ML exercise



x_train, x_val, y_train, y_val=train_test_split(train_input, y, test_size=0.2, random_state=2017)



rf.fit(x_train, y_train)

rf_preds=rf.predict_proba(x_val)

print(log_loss(y_val, rf_preds))



end_time = time.time()

print("it takes %.3f seconds to train and predict" % (end_time - start_time))

# create submission with Random Forest model



test_rf_preds=rf.predict_proba(test_input)

rf_submission=sample.copy()

rf_submission.predict_0=test_rf_preds[:,0]

rf_submission.predict_1=test_rf_preds[:,1]

rf_submission.predict_2=test_rf_preds[:,2]

rf_submission.to_csv('rf_submission_simpleCV.csv.gz', compression='gzip', index=False)
start_time=time.time()

rf=RandomForestClassifier(n_estimators=200, random_state=2017, n_jobs=4)



rf.fit(train_input, y)

test_rf_preds=rf.predict_proba(test_input)



end_time = time.time()

print("it takes %.3f seconds to train" % (end_time - start_time))





rf_submission=sample.copy()

rf_submission.predict_0=test_rf_preds[:,0]

rf_submission.predict_1=test_rf_preds[:,1]

rf_submission.predict_2=test_rf_preds[:,2]

rf_submission.to_csv('rf_submission_simpleCV_alldata.csv.gz', compression='gzip', index=False)