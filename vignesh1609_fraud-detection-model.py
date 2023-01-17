# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas_profiling as pdp

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 300)

pd.set_option('display.max_colwidth', 5000)

pd.options.display.float_format = '{:.3f}'.format

%matplotlib inline

plt.style.use('fivethirtyeight')
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RepeatedKFold, GridSearchCV

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc



import xgboost as xgb
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')



sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
pd.set_option('display.max_columns', 500)

train_transaction.head(5)
test_transaction.head(5)
train_identity.head(5)
test_identity.head(5)
train = train_transaction.merge(train_identity , how = 'left' , on = 'TransactionID')

test = test_transaction.merge(test_identity , how = 'left' , on = 'TransactionID')

print('Train dataset has {} rows and {} columns.'.format(train.shape[0], train.shape[1]))

print('Test dataset has {} rows and {} columns.'.format(test.shape[0], test.shape[1]))
del train_transaction, train_identity, test_transaction, test_identity
def is_integer_num(n):

    if isinstance(n, int):

        return True

    if isinstance(n, float):

        return n.is_integer()

    return False



def missing_values_table_specified_value(df, value=0.5): 

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum()/len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(

    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    

    if is_integer_num(value):

        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns['Missing Values'] >= value]

        print('The number of columns with {} counts missing values is {}.'.format(value, len(mis_val_table_ren_columns)))

    else:

        value = value * 100

        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns['% of Total Values'] >= value]

        print('The number of columns with {}% missing values is {}.'.format(value, len(mis_val_table_ren_columns)))

    return mis_val_table_ren_columns 



def missing_values_table(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_values_table_specified_value(train, 0.5).head()
missing_values_table_specified_value(test, 0.5).head()
display(missing_values_table(train), missing_values_table(test))

train['isFraud'].value_counts()
sns.countplot(train['isFraud'])
f,ax=plt.subplots(1,2,figsize=(18,8))

train['isFraud'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('isFraud')

ax[0].set_ylabel('')

sns.countplot('isFraud',data=train,ax=ax[1])

ax[1].set_title('isFraud')

plt.show()
train=train[train.columns[train.isnull().mean() <= 0.70]] 

test=test[test.columns[test.isnull().mean() <= 0.70]] 
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

print(quantitative)

print('Counts: {}'.format(len(quantitative)))
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

print(qualitative)

print('Counts: {}'.format(len(qualitative)))
for column in qualitative:

    train[column].fillna(train[column].mode()[0], inplace=True)
qualitative_test = [f for f in test.columns if train.dtypes[f] == 'object']

print(qualitative_test)

print('Counts: {}'.format(len(qualitative_test)))
for column in qualitative_test:

    test[column].fillna(test[column].mode()[0], inplace=True)
for column in quantitative:

    train[column].fillna(train[column].mean(), inplace=True)
quantitative_test = [f for f in train.columns if train.dtypes[f] != 'object']

print(quantitative_test)

print('Counts: {}'.format(len(quantitative_test)))
del quantitative_test[1]
quantitative_test[1]


for column in quantitative_test:

    test[column].fillna(test[column].mean(), inplace=True)
print(train.shape)

print(test.shape)
X_train = train.drop('isFraud', axis=1)

y_train = train['isFraud'].copy()

X_test = test.copy()
X_train.shape, X_test.shape
# Label Encoding

for f in qualitative:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(X_train[f].values) + list(X_test[f].values))

    X_train[f] = lbl.transform(list(X_train[f].values))

    X_test[f] = lbl.transform(list(X_test[f].values)) 
# Check if it is encoded

print(len(X_train.select_dtypes(include='object').columns))

print(len(X_test.select_dtypes(include='object').columns))

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

score = cross_val_score(LogisticRegression(),X_train,y_train).mean()
print(score)
#decisiontree

from sklearn.tree import DecisionTreeClassifier

decision_score = cross_val_score(DecisionTreeClassifier(),X_train,y_train).mean()

print(decision_score)
#randomforest

from sklearn.ensemble import RandomForestClassifier

random_score = cross_val_score(RandomForestClassifier(),X_train,y_train).mean()

print(random_score)

rand_model=RandomForestClassifier()

rand_model.fit(X_train,y_train)

rand_pred=rand_model.predict(X_test)
sample_submission['isFraud'] = rand_pred

sample_submission.to_csv('IEEE_SUBMISSION.csv',index=False)

sample_submission.columns
sample_submission['isFraud'].value_counts()
sample_submission['isFraud'].value_counts()
sample_submission.head()