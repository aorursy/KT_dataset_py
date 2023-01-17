import pandas as pd
import numpy as np


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression


import warnings
warnings.filterwarnings("ignore")
print('Importing data...')

data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')
buro_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card  = pd.read_csv('../input/credit_card_balance.csv')
POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')
payments = pd.read_csv('../input/installments_payments.csv')
lgbm_submission = pd.read_csv('../input/sample_submission.csv')

y = data['TARGET']

del data['TARGET']

data['loan_to_income'] = data.AMT_ANNUITY/data.AMT_INCOME_TOTAL
test['loan_to_income'] = test.AMT_ANNUITY/test.AMT_INCOME_TOTAL

#One-hot encoding of categorical features in data and test sets

categorical_features = [col for col in data.columns if data[col].dtype == 'object']


one_hot_df = pd.concat([data,test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)


data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]

#Pre-processing buro_balance

print('Pre-processing buro_balance...')

buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max

buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')

#Pre-processing previous_application

print('Pre-processing previous_application...')
#One-hot encoding of categorical features in previous application data set
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_features)
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']

del avg_prev['SK_ID_PREV']

print('Pre-processing buro...')

#One-hot encoding of categorical features in buro data set

buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
buro = pd.get_dummies(buro, columns=buro_cat_features)
avg_buro = buro.groupby('SK_ID_CURR').mean()

avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

del avg_buro['SK_ID_BUREAU']

#Pre-processing POS_CASH

print('Pre-processing POS_CASH...')

le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

print('Pre-processing credit_card...')
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing payments

print('Pre-processing payments...')

avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()

del avg_payments['SK_ID_PREV']

data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')


data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
print('data: ', data.shape, 'test: ', test.shape)
print('Removing features with more than 80% missing...')

test = test[test.columns[data.isnull().mean() < 0.85]]
data = data[data.columns[data.isnull().mean() < 0.85]]
print('data: ', data.shape, 'test: ', test.shape)
train_length = data.shape[0]
df_train_ID = data['SK_ID_CURR']
df_test_ID = test['SK_ID_CURR']

data.drop(['SK_ID_CURR'], axis=1, inplace=True)
test.drop('SK_ID_CURR', axis=1, inplace=True)
df_all = pd.concat([data, test]).reset_index(drop=True)
all_with_null = df_all.loc[:, df_all.isnull().any()]
all_without_null = df_all.loc[:, df_all.notnull().all()]
features_with_null = all_with_null.columns
for i, temp_feature in enumerate(features_with_null):
    print('For now, {} features have null data'.format(df_all.isnull().any().sum()))
    print('{} have {} null data'.format(temp_feature, df_all[temp_feature].isnull().sum()))
    temp_train = all_without_null.copy()
    temp_train[temp_feature] = all_with_null[temp_feature]

    new_train = temp_train.loc[temp_train[temp_feature].notnull(), :]
    new_test = temp_train.loc[temp_train[temp_feature].isnull(), :]

    temp_target = new_train[temp_feature].values

    new_train.drop([temp_feature], axis=1, inplace=True)
    new_test.drop([temp_feature], axis=1, inplace=True)
    
    # you can add gridsearch or randomsearch for parameter tunning of linear regression model
#     x_tr, x_vld, y_tr, y_vld = train_test_split(new_train, temp_target, test_size=0.2, random_state=1989)
    print('-'*30,  '{} : Start Linear regression'.format(i), '-'*30)
    lr = LinearRegression()
    lr.fit(new_train, temp_target)

    temp_pred = lr.predict(new_test)

    new_train[temp_feature] = temp_target
    new_test[temp_feature] = temp_pred
    print('Prediction and concat')
    foo = pd.concat([new_train, new_test]).sort_index()
    
    df_all[temp_feature] = foo[temp_feature]
    del foo
df_train_filled = df_all[:train_length]
df_test_filled = df_all[train_length:]
