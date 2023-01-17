# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.preprocessing import LabelEncoder

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../TransactionFraudDetection/train.csv')

df_test = pd.read_csv('../TransactionFraudDetection/test.csv')
transids = df_test['TransactionID']
df_train['Transaction_day_of_week'] = np.floor((df_train['TransactionDT'] / (3600 * 24) - 1) % 7)

df_test['Transaction_day_of_week'] = np.floor((df_test['TransactionDT'] / (3600 * 24) - 1) % 7)



df_train['Transaction_hour_of_day'] = np.floor(df_train['TransactionDT'] / 3600) % 24

df_test['Transaction_hour_of_day'] = np.floor(df_test['TransactionDT'] / 3600) % 24



df_train['TransactionAmt_decimal'] = ((df_train['TransactionAmt'] - df_train['TransactionAmt'].astype(int)) * 1000).astype(int)

df_test['TransactionAmt_decimal'] = ((df_test['TransactionAmt'] - df_test['TransactionAmt'].astype(int)) * 1000).astype(int)
y = df_train['isFraud']
len_train = df_train.shape[0]

combined = pd.concat([df_train,df_test],sort=False)

del df_train, df_test
combined['D2/D1'] = combined['D2'] / combined['D1']

combined['D2+D1'] = combined['D2'] + combined['D1']

combined['DevTypeTransAmt'] = (combined['DeviceType'] == 1) & (combined['TransactionAmt'] > 1000)

combined['C11_12relation'] = (combined['C11'] < 200) & (combined['C12'] > 40)

combined['V292_V242relation'] = (combined['V292'] < 10) & (combined['V242'] > 2.5)



combined['TransactionAmt'].fillna('mean')


V_features = ['V202', 'V211', 'V212', 'V213', 'V218', 'V263', 'V264', 'V265', 'V273', 'V274', 'V275', 'V279', 'V280', 'V293', 'V295', 'V306']

combined.drop(V_features, inplace=True, axis=1)

del V_features

df_train=combined[:len_train]

df_test=combined[len_train:]

lst = [combined]

del lst
def browser(df):

    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1

    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1

    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1

    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1

    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1

    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df



df_train = browser(df_train)

df_test = browser(df_test)

features_to_use = ['card1', 'card2', 'addr1', 'TransactionAmt', 'D15', 'Transaction_hour_of_day', 'dist1', 'card5', 'D4', 'P_emaildomain', 'D10'

                   , 'C13', 'TransactionAmt_decimal', 'Transaction_day_of_week', 'id_02', 'id_20', 'id_19', 'D8', 'D11', 'D1', 'C1', 'D2', 'id_31', 'C2', 'DeviceInfo', 'D5', 

                  'D3', 'C6', 'C11', 'C14', 'dist2', 'D2+D1', 'id_06', 'id_13', 'R_emaildomain', 'id_05', 'C9', 'V307', 'V313', 'D9', 'V310', 'id_01', 'M4', 'M6', 'D14'

                  , 'id_33', 'card4', 'M5', 'card6', 'V130', 'C5', 'V315', 'id_30', 'V314', 'D2/D1', 'V127', 'D6', 'card3', 'V308', 'id_14', 'C10', 'C8', 'D12', 'id_18', 'V83', 

                  'V312', 'D13', 'V87', 'V45', 'V82', 'V317', 'C12', 'V75', 'V76', 'V53', 'V62', 'V38', 'V54', 'V320'

                  , 'V20', 'id_38', 'V285', 'ProductCD', 'V61', 'V309', 'V78', 'V131', 'V283', 'V291', 'V203', 'V35', 'V311', 'V13', 'V77', 'V294',

                   'V165', 'V44', 'V12', 'M3', 'V36', 'V133', 'V282', 'V281', 'V5', 'V19', 'id_17', 'V56', 'V267', 

                   'V86', 'M8', 'V277', 'V55', 'V136', 'M7', 'V37', 'V318', 'V129', 'DeviceType', 'V67', 'M9', 'C4', 'V99', 

                   'V316', 'V266', 'V48', 'V128', 'V160', 'V207', 'V4', 'V221', 'D7', 'V90', 'V24', 'V296', 'V258', 'M2', 'V139',

                   'id_09', 'V261', 'V204', 'V270', 'V268', 'V271', 'V209', 'V208', 'V143', 'id_32', 'V10', 'V81', 'V66', 'V321', 

                   'V164', 'V96', 'V152', 'V23', 'V70', 'V220', 'V215', 'V187', 'V69', 'V289', 'V229','TransactionID', 'id_03', 'id_11'] #id_11, id_03

def label_encode_data(_df_train, _df_test):

    catcols=[]

    for col in list(_df_train):

        if _df_train[col].dtype=='object':

            print(col)

            catcols.append(col)

            _df_train[col] = _df_train[col].fillna('unseen_before_label')

            _df_test[col]  = _df_test[col].fillna('unseen_before_label')

            le = LabelEncoder()

            le.fit(list(_df_train[col])+list(_df_test[col]))

            _df_train[col+'label'] = le.transform(_df_train[col])

            _df_test[col+'label']  = le.transform(_df_test[col])

        

            _df_train[col+'label'] = _df_train[col+'label']

            _df_test[col+'label'] = _df_test[col+'label']

            del le

            

    return _df_train, _df_test, catcols

    
params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 0.9,

        }

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold
x_predict = df_test[features_to_use]

features_to_use.append('isFraud')

x = df_train[features_to_use]

cat_cols = []

x, x_predict, cat_cols = label_encode_data(x, x_predict)



y_predict = np.zeros(len(x_predict))

num_folds = 5

kfolds = KFold(n_splits=num_folds, shuffle=True)

i = 1

for train_index, test_index in kfolds.split(x):

        x_train = x.loc[train_index]

        x_valid = x.loc[test_index]

        

        

        cat_cols.append('isFraud')

        print()

        print()

        print("FOLD: " + str(i))

        print()

        i += 1

        

        d_train = lgb.Dataset(x_train.drop(cat_cols, axis=1), label=y[train_index])

        model = lgb.train(params, d_train, 1000)

        print(roc_auc_score(y[test_index], model.predict(x_valid.drop(cat_cols, axis=1))))

        cat_cols.remove('isFraud')

        y_predict += (model.predict(x_predict.drop(cat_cols, axis=1)) / num_folds)

my_sub = pd.DataFrame({'TransactionID':transids, 'isFraud':y_predict})

my_sub.to_csv('submission.csv', index = False)