# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
POS_CASH_balance=pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv',encoding='cp932')

application_test=pd.read_csv('../input/home-credit-default-risk/application_test.csv',encoding='cp932')

application_train=pd.read_csv('../input/home-credit-default-risk/application_train.csv',encoding='cp932')

bureau=pd.read_csv('../input/home-credit-default-risk/bureau.csv',encoding='cp932')

bureau_balance=pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv',encoding='cp932')

credit_card_balance=pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv',encoding='cp932')

installments_payments=pd.read_csv('../input/home-credit-default-risk/installments_payments.csv',encoding='cp932')

previous_application=pd.read_csv('../input/home-credit-default-risk/previous_application.csv',encoding='cp932')

sample_submission=pd.read_csv('../input/home-credit-default-risk/sample_submission.csv',encoding='cp932')
df_list = [application_test, application_train, bureau, bureau_balance, POS_CASH_balance, credit_card_balance, previous_application, installments_payments, sample_submission]

application_train['DAYS_EMPLOYED_ANOM'] = application_train["DAYS_EMPLOYED"] == 365243

application_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
application_test['DAYS_EMPLOYED_ANOM'] = application_test["DAYS_EMPLOYED"] == 365243

application_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values = missing_values_table(application_train)

missing_values.head(20)
delete_columns = ['COMMONAREA_MODE'

, 'COMMONAREA_AVG'

, 'COMMONAREA_MEDI'

, 'NONLIVINGAPARTMENTS_MEDI'

,'NONLIVINGAPARTMENTS_MODE'

,'NONLIVINGAPARTMENTS_AVG'

,'FONDKAPREMONT_MODE'

,'LIVINGAPARTMENTS_MEDI'

,'LIVINGAPARTMENTS_AVG'

,'LIVINGAPARTMENTS_MODE'

,'FLOORSMIN_AVG'

,'FLOORSMIN_MEDI'

,'FLOORSMIN_MODE'

,'YEARS_BUILD_MEDI'

,'YEARS_BUILD_MODE'

,'YEARS_BUILD_AVG'

,'OWN_CAR_AGE']



application_train.drop(delete_columns, axis=1, inplace=True)
# Find correlations with the target and sort

correlations = application_train.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(6))

print('\nMost Negative Correlations:\n', correlations.head(5))
features=['DAYS_LAST_PHONE_CHANGE'

,'REGION_RATING_CLIENT' 

,'REGION_RATING_CLIENT_W_CITY'

,'DAYS_BIRTH'

,'EXT_SOURCE_3'

,'EXT_SOURCE_2'

,'EXT_SOURCE_1'

,'DAYS_EMPLOYED'

,'FLOORSMAX_AVG'

,'AMT_INCOME_TOTAL'

,'NAME_HOUSING_TYPE']
train = application_train[features]

test = application_test[features]

data = pd.concat([train, test], sort=False)
data.head()
def get_df_name(df):

    name =[x for x in globals() if globals()[x] is df][0]

    return name
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le_count = 0

for col in data:

    if data[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(data[col].unique())) <= 2:

            # Train on the training data

            le.fit(data[col])

            # Transform both training and testing data

            data[col] = le.transform(data[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
data = pd.get_dummies(data)



print('Training Features shape: ', data.shape)
data['EXT_SOURCE_1'].fillna(data['EXT_SOURCE_1'].median(), inplace=True)

data['EXT_SOURCE_3'].fillna(data['EXT_SOURCE_3'].median(), inplace=True)

data['FLOORSMAX_AVG'].fillna(data['FLOORSMAX_AVG'].median(), inplace=True)
train=data[:len(train)]

test=data[len(train):]



y_train = application_train['TARGET']

X_train = train

X_test = test
X_train.head()
X_test.head()
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb



y_preds = []

models = []



#初期値が0で長さがトレーニングデータ分の長さの配列

oof_train = np.zeros((len(X_train),))

#データを5個に分ける。random_state＝0は初期seed、shuffle=Trueは要素がシャッフルする

cv = KFold(n_splits=10, shuffle=True, random_state=0)







params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}

#enumerateはfold_id をindex番号としてそのまま使える

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

#     print(train_index, valid_index)

    # : は全てのラベル、つまり、train_index行の全ての列ラベルを選択している

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]

    

    #ここから機械学習に通す

    lgb_train = lgb.Dataset(X_tr, y_tr)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)



    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=10,

        num_boost_round=1000,

        early_stopping_rounds=10

    )



    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    #複数回やってるから分割ごとのモデルと予測結果保存するよ

    y_preds.append(y_pred)

    models.append(model)
y_pred_oof = (oof_train > 0.5).astype(int)

roc_auc_score(y_train, y_pred_oof)
sub = sample_submission



y_pred = (y_pred > 0.5).astype(int)



sub['TARGET'] = y_pred

sub.to_csv("submission_lightgbm_optuna.csv", index=False)



sub.head()