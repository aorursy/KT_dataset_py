# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import lognorm, gamma
import collections
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/ecommerce-analytics-ml-hackathon/train_8wry4cB.csv')
test = pd.read_csv('/kaggle/input/ecommerce-analytics-ml-hackathon/test_Yix80N0.csv')
sample_submission = pd.read_csv('/kaggle/input/ecommerce-analytics-ml-hackathon/sample_submission_opxHi4g.csv')
print("Train shape: ", train.shape, 'Test shape: ', test.shape)
test.head()
sample_submission.head()
# plot target histogram
plt.hist(train['gender'])
plt.show()
def overall_session_time(dataset):
    '''Retruns difference between startTime and endTime in minutes'''
    dataset['startTime'] = pd.to_datetime(dataset['startTime'], format='%d/%m/%y %H:%M')
    dataset['endTime'] = pd.to_datetime(dataset['endTime'], format='%d/%m/%y %H:%M')
    dataset['session_time'] = pd.to_timedelta(dataset['endTime']) - pd.to_timedelta(dataset['startTime'])
    dataset['session_time'] = dataset['session_time'].apply(lambda x: x.total_seconds() / 60)
    
overall_session_time(train)
overall_session_time(test)
def convert_dates_to_dmY(dataset):
    dataset['start_day'] = dataset['startTime'].apply(lambda x: x.date().day) 
    dataset['start_month'] = dataset['startTime'].apply(lambda x: x.date().month) 
    dataset['start_year'] = dataset['startTime'].apply(lambda x: x.date().year) 
    
    dataset['end_day'] = dataset['endTime'].apply(lambda x: x.date().day) 
    dataset['end_month'] = dataset['endTime'].apply(lambda x: x.date().month) 
    dataset['end_year'] = dataset['endTime'].apply(lambda x: x.date().year)
    
convert_dates_to_dmY(train)
convert_dates_to_dmY(test)
plt.scatter(train['startTime'], train['session_time'])
def transform_start_hour(dataset):
    dataset['start_hour_min'] = dataset['startTime'].dt.hour * 60 
    dataset['start_min'] = dataset['startTime'].dt.minute 
    
transform_start_hour(train)
transform_start_hour(test)
def start_day(row):
    p = pd.Period(row.startTime.date(), freq='H')
    return p.dayofyear

def week_day(row):
    p = pd.Period(row.startTime.date(), freq='H')
    return p.dayofweek

def day_of_year(dataset):
    dataset['day_of_year'] = dataset.apply(lambda row: start_day(row), axis=1)
    
def day_of_week(dataset):
    dataset['day_of_week'] = dataset.apply(lambda row: week_day(row), axis=1)

day_of_year(train)
day_of_week(train)

day_of_year(test)
day_of_week(test)
def product_count(dataset):
    "Returns new column with count of viewed products"
    dataset['product_count'] = dataset['ProductList'].apply(lambda x: len(x.split(";")))
    
# product_count(train)
# product_count(test)
def encode_gender(dataset):
    le = preprocessing.LabelEncoder()
    le.fit(dataset['gender'].unique())
    dataset['gender'] = le.transform(dataset['gender'])
    return le

le = encode_gender(train)
def transform_product_list(dataset):
    dataset['ProductList'] = dataset['ProductList'].apply(lambda x: x.split(";"))
    
transform_product_list(train)
transform_product_list(test)
def extract_categories(dataset):
    dataset['category'] = ""
    dataset['sub_category'] = ""
    dataset['sub_sub_category'] = ""
    dataset['product_name'] = ""
    
    dataset['category_count'] = ""
    dataset['sub_category_count'] = ""
    dataset['sub_sub_category_count'] = ""
    dataset['product_name_count'] = ""
    
    for idx in dataset.index:
        cat = set()
        sub_cat = set()
        sub_sub_cat = set()
        prod_name = set() 
        for product_code in dataset['ProductList'].iloc[idx]:
            _prod = product_code.split('/')
            _cat = _prod[0]
            _sub_cat = _prod[1]
            _sub_sub_cat = _prod[2]
            _prod_name = _prod[3]
            cat.add(_cat)
            sub_cat.add(_sub_cat)
            sub_sub_cat.add(_sub_sub_cat)
            prod_name.add(_prod_name)
        dataset['category'].iloc[idx] = list(cat)
        dataset['sub_category'].iloc[idx] = list(sub_cat)
        dataset['sub_sub_category'].iloc[idx] = list(sub_sub_cat)
        dataset['product_name'].iloc[idx] = list(prod_name)
    
        dataset['category_count'].iloc[idx] = len(list(cat))
        dataset['sub_category_count'].iloc[idx] = len(list(sub_cat))
        dataset['sub_sub_category_count'].iloc[idx] = len(list(sub_sub_cat))
        dataset['product_name_count'].iloc[idx] = len(list(prod_name))
        
extract_categories(train)
extract_categories(test)

#unique categories:  11 
#unique sub-categories:  85 
#unique sub-sub-categories:  360 
#unique product names:  16503
# CATEGORY
df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
train_category_df = pd.DataFrame(mlb.fit_transform(train['category']),columns=mlb.classes_)

df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
test_category_df = pd.DataFrame(mlb.fit_transform(test['category']),columns=mlb.classes_)

# SUB-CATEGORY
df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
train_sub_category_df = pd.DataFrame(mlb.fit_transform(train['sub_category']),columns=mlb.classes_)

df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
test_sub_category_df = pd.DataFrame(mlb.fit_transform(test['sub_category']),columns=mlb.classes_)

cols_train = train_sub_category_df.columns
cols_test = test_sub_category_df.columns

def intersection(cols_train, cols_test): 
    return list(set(cols_train) & set(cols_test)) 

sub_category_cols = intersection(cols_train, cols_test)

# SUB-SUB-CATEGORY
df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
train_sub_sub_category_df = pd.DataFrame(mlb.fit_transform(train['sub_sub_category']),columns=mlb.classes_)

df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
test_sub_sub_category_df = pd.DataFrame(mlb.fit_transform(test['sub_sub_category']),columns=mlb.classes_)

cols_train = train_sub_sub_category_df.columns
cols_test = test_sub_sub_category_df.columns

def intersection(cols_train, cols_test): 
    return list(set(cols_train) & set(cols_test)) 

sub_sub_category_cols = intersection(cols_train, cols_test)

# PRODUCT NAME
df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
train_product_name_df = pd.DataFrame(mlb.fit_transform(train['product_name']),columns=mlb.classes_)

df = pd.DataFrame()
mlb = preprocessing.MultiLabelBinarizer()
test_product_name_df = pd.DataFrame(mlb.fit_transform(test['product_name']),columns=mlb.classes_)

cols_train = train_product_name_df.columns
cols_test = test_product_name_df.columns

def intersection(cols_train, cols_test): 
    return list(set(cols_train) & set(cols_test)) 

product_name_cols = intersection(cols_train, cols_test)

# CONCAT ALL
train_sub_category_df = train_sub_category_df[sub_category_cols]
test_sub_category_df = test_sub_category_df[sub_category_cols] 

train_sub_sub_category_df = train_sub_sub_category_df[sub_sub_category_cols]
test_sub_sub_category_df = test_sub_sub_category_df[sub_sub_category_cols] 

train_product_name_df = train_product_name_df[product_name_cols]
test_product_name_df = test_product_name_df[product_name_cols]

train = pd.concat([train, train_category_df, train_sub_category_df, train_sub_sub_category_df, train_product_name_df], axis=1)
test = pd.concat([test, test_category_df, test_sub_category_df, test_sub_sub_category_df, test_product_name_df], axis=1)
# train.corr().apply(lambda x: x > 0.1).query('gender == True').index

# 'gender', 'A00001', 'A00004', 'B00031', 'B00015', 'B00009', 'B00027',
# 'B00001', 'C00019', 'C00032', 'C00001', 'C00028', 'C00066', 'C00186',
# 'C00043', 'C00044', 'C00082', 'C00042', 'C00182'
# train.corr().apply(lambda x: x < -0.1).query('gender == True').index

# 'A00002', 'A00003', 'B00003', 'B00012', 'B00002', 'C00007'
train = train[['session_id', 'startTime', 'endTime', 'ProductList','category_count', 'sub_category_count', 'sub_sub_category_count',
               'product_name_count', 'session_time', 'start_day','start_month','day_of_year', 'end_day', 'day_of_week', 'start_hour_min', 'start_min',
                'A00001', 'A00004', 'B00031', 'B00015', 'B00009', 'B00001', 'C00019', 'C00032', 'C00001', 'C00028', 'C00066', 'C00186',
                'C00043', 'C00044', 'C00082', 'A00002', 'A00003', 'B00003', 'B00012', 'B00002', 'C00007', 'gender']]

test = test[['session_id', 'startTime', 'endTime', 'ProductList','category_count', 'sub_category_count', 'sub_sub_category_count',
               'product_name_count', 'session_time', 'start_day','start_month', 'day_of_year','end_day', 'day_of_week', 'start_hour_min', 'start_min',
             'A00001', 'A00004', 'B00031', 'B00015', 'B00009', 'B00001', 'C00019', 'C00032', 'C00001', 'C00028', 'C00066', 'C00186',
                'C00043', 'C00044', 'C00082', 'A00002', 'A00003', 'B00003', 'B00012', 'B00002', 'C00007']]

plt.figure(figsize=(15, 15))
sns.heatmap(train.corr(), vmin=-1, center=0, vmax=1, square=True)
plt.show()
X_train = train.drop(['session_id', 'startTime', 'endTime', 'ProductList','gender'], axis=1)
y_train = train['gender'].values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 890)

X_test = test.drop(['session_id', 'startTime', 'endTime', 'ProductList'], axis=1)

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val)
evals_result = {} 

params = {
        'task': 'train',
        'objective': 'binary',
        'metric' : 'None',
        'boosting': 'gbdt',
        'learning_rate': 0.001,
        'num_leaves': 100,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'n_estimators': 10000,
    }


def accuracy(preds, lgb_train):
    preds = np.round(preds).astype(int)
    acc = accuracy_score(preds, lgb_train.get_label())
    return ('accuracy', acc, True)

cv_results = lgb.cv(params, lgb_train, num_boost_round = 10000, nfold = 10, feval = accuracy, early_stopping_rounds = 100, verbose_eval = 100, seed = 50)

lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, feval = accuracy,  evals_result = evals_result, verbose_eval = 100)
# plot feature importance
ax = lgb.plot_importance(lgbm_model)
fig = ax.figure
fig.set_size_inches(20, 20)

# plot cv metric
od = collections.OrderedDict()
d = {}
results = cv_results['accuracy-mean']
od['accuracy'] = results
d['cv'] = od

ax = lgb.plot_metric(d,title='Metric during cross-validation', metric='accuracy')
plt.show()

print("CV best score: " + str(max(cv_results['accuracy-mean'])))

# plot train metric
ax = lgb.plot_metric(evals_result, metric='accuracy')
plt.show()

print("Train best score: " + str(max(evals_result['valid_0']['accuracy'])))

predictions = lgbm_model.predict(X_test)

# Writing output to file
subm = pd.DataFrame()
subm['session_id'] = test['session_id']

predictions = np.round(predictions).astype(int)
subm['gender'] = predictions
subm['gender'] = le.inverse_transform(subm['gender'])

# plot predictions
plt.hist(subm['gender'])
plt.show()

subm.to_csv("/kaggle/working/" + 'submission.csv', index=False)
subm