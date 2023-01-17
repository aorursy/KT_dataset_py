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
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_formats = {'png', 'retina'}
import seaborn as sns
train = pd.read_csv('../input/property-price-prediction-challenge-2nd/DC_train.csv')
test = pd.read_csv('../input/property-price-prediction-challenge-2nd/DC_test.csv')
print('train shape : ', train.shape)
print('test shape : ', test.shape)
train['PRICE'] = np.log1p(train['PRICE'])
train = train[(train['PRICE']>=8.0) & (train['PRICE']<=17.0)]
print('train shape : ', train.shape)
print('test shape : ', test.shape)
train.isnull().sum()
test.isnull().sum()
train.dtypes
no_mv_columns = []    # features without missing value
with_mv_columns = []  # features with missing value
for col in train.columns:
    if train[col].isnull().any()==False:
        no_mv_columns.append(col)
    else:
        with_mv_columns.append(col)
        
print('no_mv_columns')
print(no_mv_columns)
print('with_mv_columns')
print(with_mv_columns)
for elem in no_mv_columns:
    print(elem)
    print(train[elem].value_counts(), '\n')
for elem in with_mv_columns:
    print(elem)
    print('Number of NaN : ', train[elem].isnull().sum())
    print(train[elem].value_counts(), '\n')
for elem in with_mv_columns:
    print(elem)
    print('Number of NaN : ', test[elem].isnull().sum())
    print(test[elem].value_counts(), '\n')
# for feature in no_mv_columns:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     plt.hist(train[feature], bins=50, rwidth=0.8)
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()

# for feature in no_mv_columns:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     plt.scatter(train[feature], train['PRICE'], alpha=0.3, color='blue')
#     plt.xlabel(feature)
#     plt.ylabel('PRICE')
#     plt.show()
# corr = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
# AC
train['AC'].value_counts()
# FIREPLACES
# train['FIREPLACES']=1601　→　0
train['FIREPLACES'].replace(1601, 0, inplace=True)
train['FIREPLACES'].value_counts()
# zipcode
train['ZIPCODE']
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.scatter(train['ZIPCODE'], train['PRICE'], alpha=0.3)
# plt.xlabel('ZIPCODE')
# plt.ylabel('PRICE')
# plt.show()
# AYB
mean = round(train['AYB'].mean())
train['AYB'] = train['AYB'].fillna(mean)
mean = round(test['AYB'].mean())
test['AYB'] = test['AYB'].fillna(mean)

# YR_RMDL
# train['YR_RMDL']=20.0 → 2000.0
train['YR_RMDL'].replace(20.0, 2000.0, inplace=True)
# mean = train['YR_RMDL'].mean()
# train['YR_RMDL'] = train['YR_RMDL'].fillna(mean)
# train['YR_RMDL'].isnull().sum()
train['YR_RMDL'].describe()

train[train['AYB']>train['YR_RMDL']][['AYB', 'YR_RMDL']]
# GBA
train['GBA'] = train['GBA'].fillna(0.0)
test['GBA'] = test['GBA'].fillna(0.0)
# NUM_UNITS
train['NUM_UNITS'] = train['NUM_UNITS'].fillna(0.0)
test['NUM_UNITS'] = test['NUM_UNITS'].fillna(0.0)
# KITCHENS
train['KITCHENS'] = train['KITCHENS'].fillna(0.0)
test['KITCHENS'] = test['KITCHENS'].fillna(0.0)
# STYLE
train['STYLE'] = train['STYLE'].fillna('Vacant')
test['STYLE'] = test['STYLE'].fillna('Vacant')
# GRADE
train['GRADE'] = train['GRADE'].fillna('No Data')
test['GRADE'] = test['GRADE'].fillna('No Data')
# ROOF
train['ROOF'] = train['ROOF'].fillna('No Roof')
test['ROOF'] = test['ROOF'].fillna('No Roof')
# INTWALL
train['INTWALL'] = train['INTWALL'].fillna('No IntWall')
test['INTWALL'] = test['INTWALL'].fillna('No IntWall')
# EXTWALL
train['EXTWALL'] = train['EXTWALL'].fillna('No ExtWall')
test['EXTWALL'] = test['EXTWALL'].fillna('No ExtWall')
# CNDTN
train['CNDTN'] = train['CNDTN'].fillna('No Data')
test['CNDTN'] = test['CNDTN'].fillna('No Data')
# STORIES
index_list1 = train[train['ROOF']=='No Roof'].index
index_list2 = test[test['ROOF']=='No Roof'].index

for i in index_list1:
    train['STORIES'][i] = 0.00

for i in index_list2:
    test['STORIES'][i] = 0.00
train[train['STORIES'].isnull()==True]['STYLE']
test[test['STORIES'].isnull()==True]['STYLE']
for elem in with_mv_columns:
    print(elem)
    print('Number of Nan : ', train[elem].isnull().sum())
    print(train[elem].value_counts(), '\n')


for elem in with_mv_columns:
    print(elem)
    print('Number of Nan : ', test[elem].isnull().sum())
    print(test[elem].value_counts(), '\n')
    
train_labels = train['PRICE'].reset_index(drop=True) # PRICE
train_features = train.drop(['PRICE'], axis=1)
test_features = test

print('train_features shape : ', train_features.shape)
print('test_features shape : ', test_features.shape)

# train_labels = train['PRICE'].reset_index(drop=True) # PRICE
# train_features = train.drop(['PRICE'], axis=1)
# test_features = test

# all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
# print('all_features shape : ', all_features.shape)
# print('train_features shape : ', train_features.shape)
# print('test_features shape : ', test_features.shape)
# all_features.head(7)
# Visualize missing values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
train['AGE'] = 2018 - train['AYB']
test['AGE'] = 2018 - test['AYB']
# all_features['AGE'] = 2018 - all_features['AYB']
train['AGE'].describe()
train['IMPAGE'] = 2018-train['EYB']
test['IMPAGE'] = 2018-test['EYB']
# all_features['IMPAGE'] = 2018-all_features['EYB']
train['IMPAGE'].describe()
# BATH ROOM
train['ALLBATHRM'] = train['BATHRM'] + train['HF_BATHRM']
test['ALLBATHRM'] = test['BATHRM'] + test['HF_BATHRM']
# all_features['ALLBATHRM'] = all_features['BATHRM'] + all_features['HF_BATHRM']
train['ALLBATHRM'].describe()

# for elem in ['BATHRM', 'HF_BATHRM', 'ALLBATHRM']:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     plt.scatter(train[elem], train['PRICE'], alpha=0.3, color='blue')
#     plt.xlabel(elem)
#     plt.ylabel('PRICE')
#     plt.show()
# OTHER ROOMS 
train['OTHROOMS'] = train['ROOMS'] - train['BEDRM']
test['OTHROOMS'] = test['ROOMS'] - test['BEDRM']
# all_features['OTHROOMS'] = all_features['ROOMS'] - all_features['BEDRM']
train['OTHROOMS'].describe()
# for elem in ['ROOMS', 'BEDRM', 'OTHROOMS']:
#     fig, ax = plt.subplots(figsize=(8, 6))
#     plt.scatter(train[elem], train['PRICE'], alpha=0.3, color='blue')
#     plt.xlabel(elem)
#     plt.ylabel('PRICE')
#     plt.show()
del_feat = ['HEAT', 'AC', 'YR_RMDL', 'EYB', 'STORIES', 'SALEDATE',\
            'SALE_NUM', 'STRUCT', 'USECODE', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM',\
            'LIVING_GBA', 'FULLADDRESS', 'CITY', 'STATE', 'ZIPCODE', 'NATIONALGRID',\
            'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD',\
            'CENSUS_TRACT', 'CENSUS_BLOCK', 'SQUARE', 'X', 'Y', 'QUADRANT']
for elem in del_feat:
    del train_features[elem]
    del test_features[elem]
    
# Label Encoding 
from sklearn.preprocessing import OrdinalEncoder # 複数変換
from sklearn.preprocessing import LabelEncoder
oe = OrdinalEncoder()
le = LabelEncoder()
label_list = ['QUALIFIED', 'STYLE', 'STRUCT', 'GRADE', 'CNDTN', 'EXTWALL', 'ROOF', 'INTWALL', 'WARD']
print(label_list)
train_features = pd.get_dummies(train_features)
test_features = pd.get_dummies(test_features)
print(train_features.shape)
print(test_features.shape)
# train_features[label_list] = le.fit_transform(train_features[label_list].values)
print('train_features.columns')
print(train_features.columns)
print('test_features.columns')
print(test_features.columns)

# X = all_features.iloc[:len(train_labels), :]
# X_test = all_features.iloc[len(train_labels):, :]
# print(X.shape, train_labels.shape, X_test.shape)
X = train_features
X_test = test_features
common = set(X.columns) & set(X_test.columns)
for elem in X.columns:
    if not elem in common:
        del X[elem]
for elem in X_test.columns:
    if not elem in common:
        del X_test[elem]
print(X.shape, train_labels.shape, X_test.shape)
X.isnull().sum()
X.to_csv('X.csv', encoding='shift-jis')
X_test.to_csv('X_test.csv', encoding='shift-jis')
train_labels.to_csv('train_labels.csv', encoding='shift-jis')
