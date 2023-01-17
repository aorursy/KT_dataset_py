import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")
%matplotlib inline
test = pd.read_csv('../input/msxcase-xw/MSX_test_data.csv')

train = pd.read_csv('../input/msxcase-xw/MSX_train_data.csv')

sample_sub = pd.read_csv('../input/msxcase-xw/sample_submission.csv')

train.shape,test.shape
train.head()
test.head()
sample_sub.head()
train['target'].value_counts()
train_pos = train[train['target']=='yes'].copy()

train_neg = train[train['target']=='no'].copy()

train_pos.shape,train_neg.shape
# let's firstly check how many unique values does each feature have

for col in train.columns:

    print('There are {} unique values for :   {}'.format(len(train[col].unique()),col))
cat_feats = ['day','month','sex','type_of_employment','marital_status','education',

             'credit_default','house_owner','credit','customer_approach','result_last_campaign']

num_feats = ['duration','age','account_balance','number_of_approaches','days_since_last_campaign','number_of_approaches_last_campaign']
# data coverage for each feature

1 - pd.isnull(train).sum(axis=0) / float(train.shape[0])
# data coverage for each sample

pd.isnull(train).sum(axis=1).hist() 
cat_feats = ['day','month','sex','type_of_employment','marital_status','education',

             'credit_default','house_owner','credit','customer_approach','result_last_campaign']

num_feats = ['duration','age','account_balance','number_of_approaches','number_of_approaches_last_campaign']
# check for each cat feat, if there is any value in test but not in train

for feat in cat_feats:

    print('for featture {} the unique value only in test set are : {}'.format(feat,[i for i in list(test[feat].unique()) if i not in list(train[feat].unique())]))
train.dtypes
def plot_cat_feat(df,title,color):

    plt.figure()

    df.value_counts().plot(kind='bar',color=color)

    plt.title(title)
# check if all values present in test are in train

for feat in cat_feats:

    plot_cat_feat(train_pos[feat],feat+'_train_pos','red')

    plot_cat_feat(train_neg[feat],feat+'_train_neg','blue')

  
def plot_num_feat(df,title,color):

    plt.figure()

    df.plot(kind='hist',color=color,bins=30)

    plt.title(title)
# now check the differences of numerical features distribution

for feat in num_feats:

    plot_num_feat(train_pos[feat],feat+'_train_pos','red')

    plot_num_feat(train_neg[feat],feat+'_train_neg','blue')
cat_feats
num_feats
for f in cat_feats:

    le = LabelEncoder()

    le.fit(train[f])

    train[f+'_encoded'] = le.transform(train[f])

    test[f+'_encoded'] = le.transform(test[f])
train.head()
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

feats_to_use = [i+'_encoded' for i in cat_feats] + num_feats
mapping = {'yes':1,'no':0}
X, y = train[feats_to_use].values, train['target'].map(mapping).values
for train_index, test_index in skf.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model = lgb.LGBMClassifier()

    model.fit(X_train,y_train)

    y_hat = model.predict_proba(X_test)

    pos_loc = np.where(model.classes_==1)[0]

    auc = roc_auc_score(y_test,y_hat[:,pos_loc])

    print('The auc score is :',auc)
model.feature_importances_
feats_to_test = [i+'_encoded' for i in ['day', 'month', 'type_of_employment', 'house_owner', 

                                        'result_last_campaign']] + ['duration', 'account_balance', 'number_of_approaches']

mapping = {'yes':1,'no':0}

X_less_feat, y_less_feat = train[feats_to_test].values, train['target'].map(mapping).values

for train_index, test_index in skf.split(X_less_feat, y_less_feat):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X_less_feat[train_index], X_less_feat[test_index]

    y_train, y_test = y_less_feat[train_index], y_less_feat[test_index]

    model = lgb.LGBMClassifier()

    model.fit(X_train,y_train)

    y_hat = model.predict_proba(X_test)

    pos_loc = np.where(model.classes_==1)[0]

    auc = roc_auc_score(y_test,y_hat[:,pos_loc])

    print('The auc score is :',auc)
def cross_val(**kwargs):

    auc = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        model = lgb.LGBMClassifier(**kwargs)

        model.fit(X_train,y_train)

        y_hat = model.predict_proba(X_test)

        pos_loc = np.where(model.classes_==1)[0]

        auc.append(roc_auc_score(y_test,y_hat[:,pos_loc]))

    return np.mean(auc)

        
best_auc = -np.inf

for num_leaves in [11,21,31]:

    for max_depth in [-1,4,8,16]:

        for learning_rate in [0.1,0.01,0.001]:

            for is_unbalance in [True,False]:

                args = {'num_leaves':num_leaves,'max_depth':max_depth,'learning_rate':learning_rate,'is_unbalance':is_unbalance}

                mean_auc = cross_val(**args)

                print('-----------------------------------------')

                print('performance with param: ',args)

                print(mean_auc)

                if mean_auc > best_auc:

                    best_auc = mean_auc

                    best_param = args

                    best_model = model
best_auc
best_param
sample_sub.head()
test.head()
X_test_to_pred = test[feats_to_use].values
def cross_predict(**args):

    pred = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        model = lgb.LGBMClassifier(**args)

        model.fit(X_train,y_train)

        y_hat = model.predict_proba(X_test_to_pred)

        pos_loc = np.where(model.classes_==1)[0]

        y_prob = y_hat[:,pos_loc]

        pred.append(y_prob)

    return np.mean(pred,axis=0)
pred = cross_predict(**best_param)
test['Predicted'] = np.round(pred,6)
test[['id','Predicted']].to_csv('submission_xingchen.csv')