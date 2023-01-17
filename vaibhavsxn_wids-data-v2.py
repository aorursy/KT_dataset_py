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



import warnings

warnings.filterwarnings("ignore")



import seaborn as sns

import math

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline

import scipy.stats as sps

import re
train = pd.read_csv('../input/widsdatathon2020/training_v2.csv')

test = pd.read_csv('../input/widsdatathon2020/unlabeled.csv')

st = pd.read_csv('../input/widsdatathon2020/solution_template.csv')

ss = pd.read_csv('../input/widsdatathon2020/samplesubmission.csv')

dictionary = pd.read_csv('../input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv')



pd.set_option('display.max_columns', 500)

print('solution template shape', st.shape)

display(st.head())

print('dictionary shape', dictionary.shape)

display(dictionary.T.head())

print('train shape', train.shape)

display(train.head())

print('test shape', test.shape)

display(test.head())
# Dropping patient_id for now

train = train.copy().drop('patient_id', axis = 1)

test = test.copy().drop('patient_id', axis = 1)
from sklearn.model_selection import train_test_split



Train, Validation = train_test_split(train, test_size = 0.3)
X_train = Train.copy().drop('hospital_death', axis = 1)

y_train = Train[['encounter_id','hospital_death']]

X_val = Validation.copy().drop('hospital_death', axis = 1)

y_val = Validation[['encounter_id','hospital_death']]
X_test = test.copy().drop('hospital_death', axis = 1)

y_test = test[['encounter_id','hospital_death']]
sns.catplot('hospital_death', data= train, kind='count', alpha=0.7, height=6, aspect=1)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = train['hospital_death'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of Hospital Deaths', fontsize = 20, color = 'black')

plt.show()
plt.figure(figsize=(30,15))

ethnicity_vs_death = sns.catplot(x='ethnicity', col='hospital_death', kind='count', data=train, 

                                 order = train['ethnicity'].value_counts().index, height = 7, aspect = 1);

ethnicity_vs_death.set_xticklabels(rotation=90);
plt.figure(figsize=(30,15))

has_vs_death = sns.catplot(x='hospital_admit_source', col='hospital_death', kind='count', data=train, 

                           order = train['hospital_admit_source'].value_counts().index, height = 7, aspect = 1.5);

has_vs_death.set_xticklabels(rotation=90);
plt.figure(figsize=(30,15))

ias_vs_death = sns.catplot(x='icu_admit_source', col='hospital_death', kind='count', data=train, 

                           order = train['icu_admit_source'].value_counts().index, height = 7, aspect = 1.5);

ias_vs_death.set_xticklabels(rotation=90);
# Freq plot of Hospital ID for hospital_death = 0

train_hID_0 = train[train['hospital_death'] == 0]

plt.figure(figsize=(30,15))

hID_vs_death = sns.catplot(y='hospital_id',  orient = "v", kind='count', data=train_hID_0, order = train_hID_0['hospital_id'].value_counts().index, 

                           height = 30, aspect = 1)
# Freq plot of Hospital ID for hospital_death = 1

train_hID_1 = train[train['hospital_death'] != 0]

plt.figure(figsize=(30,20))

hID_vs_death = sns.catplot(y='hospital_id',  orient = "v", kind='count', data=train_hID_1, order = train_hID_1['hospital_id'].value_counts().index, 

                           height = 30, aspect = 1);
# Freq plot of Hospital ID for hospital_death = 0 & 1

plt.figure(figsize=(30,40))

hID_vs_death = sns.catplot(x = 'hospital_id', col='hospital_death', kind='count', data=train, order = train['hospital_id'].value_counts().index, 

                           height = 5, aspect = 2.8);

hID_vs_death.set_xticklabels(rotation=90);
plt.figure(figsize = (15,5))

sns.kdeplot(train_hID_1['age'], shade=True, color="r")

sns.kdeplot(train_hID_0['age'], shade=True, color="b")
sns.jointplot(x="age", y="bmi", data=train, kind = "kde")

sns.jointplot(x="age", y="height", data=train, kind = "kde")

sns.jointplot(x="age", y="weight", data=train, kind = "kde")
dataset = pd.concat(objs=[X_train, X_val], axis=0)
col_1 = dataset.columns
for i in col_1:

    if X_train[i].nunique() == 1:

        print('in Train', i)

    if X_val[i].nunique() == 1:

        print('in Val', i)

    if X_test[i].nunique() == 1:

        print('in Test', i)

    
# Dropping 'readmission_status'

X_train = X_train.drop(['readmission_status'], axis=1)

X_val = X_val.drop(['readmission_status'], axis=1)

X_test = X_test.drop(['readmission_status'], axis=1)
print('For Train')

d1 = X_train.nunique()

print(sorted(d1))

print("==============================")

print('For Validation')

d2 = X_val.nunique()

print(sorted(d2))



# Considering columns with <= 15 unique values for conversion
d = pd.concat(objs=[X_train, X_val], axis=0)
col = d.columns 
# For Train data

l1 = []

for i in col:

    if X_train[i].nunique() <= 15:

        l1.append(i)

        

l1
# For Val data

l2 = []

for i in col:

    if X_val[i].nunique() <= 15:

        l2.append(i)

        

l2
# For Test data

l3 = []

for i in col:

    if X_test[i].nunique() <= 15:

        l3.append(i)

        

l3
# Checking for columns in X_train and X_validation

set(l1) & set(l2)
# Checking for columns in X_train and X_test

set(l1) & set(l3)
print('Train', len(l1))

print('Validation', len(l2))

print('Common', len(set(l1) & set(l2)))
print('Train', len(l1))

print('Test', len(l3))

print('Common', len(set(l1) & set(l3)))
X_train[l1].dtypes
X_val[l2].dtypes

# Not a necessary step since we already confirmed the common columns. Included just for reference. 
X_train[l1] = pd.Categorical(X_train[l1])

X_val[l2] = pd.Categorical(X_val[l2])

X_test[l3] = pd.Categorical(X_test[l3])

print('Train dtypes:')

print(X_train[l1].dtypes)

print('======================================')

print('Validation dtypes:')

print(X_val[l2].dtypes)

print('======================================')

print('Test dtypes:')

print(X_test[l3].dtypes)
# On train data

pd.set_option('display.max_rows', 500)

NA_col = pd.DataFrame(X_train.isna().sum(), columns = ['NA_Count'])

NA_col['% of NA'] = (NA_col.NA_Count/len(X_train))*100

NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
# On val data

pd.set_option('display.max_rows', 500)

NA_col = pd.DataFrame(X_val.isna().sum(), columns = ['NA_Count'])

NA_col['% of NA'] = (NA_col.NA_Count/len(X_val))*100

NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
# On test data

pd.set_option('display.max_rows', 500)

NA_col = pd.DataFrame(X_test.isna().sum(), columns = ['NA_Count'])

NA_col['% of NA'] = (NA_col.NA_Count/len(X_test))*100

NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
cols = X_train.columns

num_cols = X_train._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))

cat_cols
# Courtesy: https://www.kaggle.com/jayjay75/wids2020-lgb-starter-script

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

for usecol in cat_cols:

    X_train[usecol] = X_train[usecol].astype('str')

    X_val[usecol] = X_val[usecol].astype('str')

    X_test[usecol] = X_test[usecol].astype('str')

    

    #Fit LabelEncoder

    le = LabelEncoder().fit(

            np.unique(X_train[usecol].unique().tolist()+

                      X_val[usecol].unique().tolist()+

                     X_test[usecol].unique().tolist()))



    #At the end 0 will be used for dropped values

    X_train[usecol] = le.transform(X_train[usecol])+1

    X_val[usecol]  = le.transform(X_val[usecol])+1

    X_test[usecol]  = le.transform(X_test[usecol])+1

    

    X_train[usecol] = X_train[usecol].replace(np.nan, 0).astype('int').astype('category')

    X_val[usecol]  = X_val[usecol].replace(np.nan, 0).astype('int').astype('category')

    X_test[usecol]  = X_test[usecol].replace(np.nan, 0).astype('int').astype('category')
X_train.set_index('encounter_id', inplace = True)

y_train.set_index('encounter_id', inplace = True)

X_val.set_index('encounter_id', inplace = True)

y_val.set_index('encounter_id', inplace = True)

X_test.set_index('encounter_id', inplace = True)

y_test.set_index('encounter_id', inplace = True)
# y_test.hospital_death = y_test.hospital_death.fillna(0)
# y_train['hospital_death'] = pd.Categorical(y_train['hospital_death'])

# y_train.dtypes
# y_test['hospital_death'] = pd.Categorical(y_test['hospital_death'])

# y_test.dtypes
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# from pandas import Series

# l=LabelEncoder() 

# l.fit(y_train['hospital_death']) 

# l.classes_ 

# y_train['hospital_death']=Series(l.transform(y_train['hospital_death']))  #label encoding our target variable 

# y_train['hospital_death'].value_counts() 
# l.fit(y_test['hospital_death']) 

# l.classes_ 

#y_test['hospital_death'].fillna(0.0, inplace = True)

# y_test['hospital_death']=Series(l.transform(y_test['hospital_death']))  #label encoding our target variable 

# y_test['hospital_death'].value_counts() 
import lightgbm as lgbm



lgbm_train = lgbm.Dataset(X_train, y_train, categorical_feature=cat_cols)

# lgbm_test = lgbm.Dataset(X_test, y_test, categorical_feature=cat_cols)

lgbm_val = lgbm.Dataset(X_val, y_val, reference = lgbm_train)
params = {'feature_fraction': 0.9,

          'lambda_l1': 1,

          'lambda_l2': 1,

          'learning_rate': 0.01,

          'max_depth': 10,

          'metric': 'auc',

          'num_leaves': 500,

          'min_data_in_leaf': 100,

          'subsample_freq': 1,

          'scale_pos_weight':1,

          'metric': 'auc',

          'is_unbalance': 'true',

          'boosting': 'gbdt',

          'bagging_fraction': 0.5,

          'bagging_freq': 10,}
evals_result = {}  # to record eval results for plotting

model_lgbm = lgbm.train(params,

                lgbm_train,

                num_boost_round=100,

                valid_sets=[lgbm_train, lgbm_val],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [182],

                evals_result=evals_result,

                verbose_eval=10)
ax = lgbm.plot_metric(evals_result, metric='auc', figsize=(15, 8))

plt.show()
test["hospital_death"] = model_lgbm.predict(X_test, predition_type = 'Probability')

test[["encounter_id","hospital_death"]].to_csv("submission_lgbm.csv",index=False)
test[["encounter_id","hospital_death"]].head()