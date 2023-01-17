# Imports

import pandas as pd

import category_encoders as ce

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

import xgboost as xgb

import numpy as np



# Pandas options to make outputs more readable

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.float_format', '{:.2f}'.format)
# Create pandas dataframes from csv files

train = pd.merge(pd.read_csv('https://raw.githubusercontent.com/WillHK/DS-Unit-2-Classification-1/master/module1-logistic-regression/train_features.csv')

                ,pd.read_csv('https://raw.githubusercontent.com/WillHK/DS-Unit-2-Classification-1/master/module1-logistic-regression/train_labels.csv'))

test = pd.read_csv('https://raw.githubusercontent.com/WillHK/DS-Unit-2-Classification-1/master/module1-logistic-regression/test_features.csv')

sample_submission = pd.read_csv('https://raw.githubusercontent.com/WillHK/DS-Unit-2-Classification-1/master/module1-logistic-regression/sample_submission.csv')
# Create Training and Validation Sets from the train dataframe



# Stratify keeps the proportions of our target approximately the same in

# the traina and val sets



train, val = train_test_split(train, train_size=0.80, test_size=0.20, 

                             stratify=train['status_group'])
# Lets look at the train dataframe and see what our data looks like

train.describe(include='all')
# payment and payment_type



print('payment\n')

print(train['payment'].value_counts())

print('\npayment_type\n')

print(train['payment_type'].value_counts())
#water_quality and quality_group



print('water_quality\n')

print(train['water_quality'].value_counts())

print('\nquality_group\n')

print(train['quality_group'].value_counts())
train = train.drop(columns=['id', 'extraction_type_group', 'extraction_type_class',

                    'payment_type', 'quality_group', 'quantity_group',

                    'source_type', 'source_class', 'waterpoint_type_group', 'management_group', 'recorded_by'])

val = val.drop(columns=['id', 'extraction_type_group', 'extraction_type_class',

                    'payment_type', 'quality_group', 'quantity_group',

                    'source_type', 'source_class', 'waterpoint_type_group', 'management_group', 'recorded_by'])

test = test.drop(columns=['id', 'extraction_type_group', 'extraction_type_class',

                    'payment_type', 'quality_group', 'quantity_group',

                    'source_type', 'source_class', 'waterpoint_type_group', 'management_group', 'recorded_by'])
train.describe(include='all')
train.dtypes
# For many numeric entries 0 or a float close enough to 0 is equivalent to nan

def clean_zeros(df):

    df_c = df.copy()

    df_c['latitude'] = df_c['latitude'].replace(-2e-08, np.nan)

    zero_cols = ['construction_year', 'longitude', 'latitude', 'gps_height',

                 'population']

    for col in zero_cols:

      df_c[col] = df_c[col].replace(0, np.nan)

    return df_c



train = clean_zeros(train)

val = clean_zeros(val)

test = clean_zeros(test)



train.isnull().sum()
# train.iloc[1]

train[train['region'] == train.iloc[1]['region']]['gps_height'].mean()
categoricals = train.select_dtypes(exclude='number').columns.tolist()

for col in categoricals:

  train[col] = train[col].fillna('MISSING')

  val[col] = val[col].fillna('MISSING')

  if col != 'status_group':

    test[col] = test[col].fillna('MISSING')

# This took too long to run so I'm using SimpleImputer below in the pipeline

# numerical_features = train.select_dtypes(include='number').columns.tolist()



# def regional_mean(df, features):

#   df_c = df.copy()

#   for i in range(len(df_c)):

#     for feature in features:

#       df_c = df_c[feature].fillna(value=df_c[df_c['region'] == df_c.iloc[i]['region']][feature].mean())



#   return df_c





# train = regional_mean(train, numerical_features)

# val = regional_mean(val, numerical_features)

  

# train.isnull().sum()
train['date_recorded'] = pd.to_datetime(train['date_recorded'], infer_datetime_format=True)

train['year_recorded'] = train['date_recorded'].dt.year

train['month_recorded'] = train['date_recorded'].dt.month

train['day_recorded'] = train['date_recorded'].dt.day

train = train.drop(columns='date_recorded')



val['date_recorded'] = pd.to_datetime(val['date_recorded'], infer_datetime_format=True)

val['year_recorded'] = val['date_recorded'].dt.year

val['month_recorded'] = val['date_recorded'].dt.month

val['day_recorded'] = val['date_recorded'].dt.day

val = val.drop(columns='date_recorded')



test['date_recorded'] = pd.to_datetime(test['date_recorded'], infer_datetime_format=True)

test['year_recorded'] = test['date_recorded'].dt.year

test['month_recorded'] = test['date_recorded'].dt.month

test['day_recorded'] = test['date_recorded'].dt.day

test = test.drop(columns='date_recorded')
target = 'status_group'



X_train = train.drop(columns=[target])

y_train = train[target]

X_val = val.drop(columns=[target])

y_val = val[target]

X_test = test
pipeline = make_pipeline(

    ce.OrdinalEncoder(),

    SimpleImputer(strategy='median'),

    RandomForestClassifier(n_estimators=1000, max_depth=40, n_jobs=-1)

)
pipeline.fit(X_train, y_train)

print(pipeline.score(X_val, y_val))
y_pred = pipeline.predict(X_test)

submission = pd.read_csv('https://raw.githubusercontent.com/WillHK/DS-Unit-2-Classification-1/master/module1-logistic-regression/sample_submission.csv')

submission['status_group'] = y_pred

submission.to_csv('submission-08.csv', index=False)