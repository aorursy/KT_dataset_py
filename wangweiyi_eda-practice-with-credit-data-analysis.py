# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/application_train.csv')
train.head()
test=pd.read_csv('../input/application_test.csv')
test.head()
print('train: ',train.shape)
print('test: ',test.shape)
# examin the sample distribution
train['TARGET'].astype(int).plot.hist() #典型不均衡分类
len(train.index)
# examin missing values
def missing_vals(df):
    mis_val=df.isnull().sum()
    mis_per=100*mis_val/len(df)
    mis_table=pd.concat([mis_val,mis_per],axis=1)
    mis_table=mis_table.rename(columns={0:'missing values',1:'missing percent'})
    mis_table=mis_table[mis_table.iloc[:,1] != 0].sort_values('missing percent',ascending=False).round(1)
    return mis_table
missing_vals(train).head()
# columns types
train.dtypes.value_counts()
# number of unique classes in each column
train.select_dtypes('object').apply(pd.Series.nunique,axis=0)
# 将类别向量化 类别大于2时使用one-hot
le=LabelEncoder()
le_count=0
for col in train:
    if train[col].dtype=='object':
        if len(list(train[col].unique()))<=2:
            le.fit(train[col])
            train[col]=le.transform(train[col])
            test[col]=le.transform(test[col])
            le_count+=1
print('%d columns has been encoded'%le_count)
# pd_get_dummies on-hot
train=pd.get_dummies(train)
test=pd.get_dummies(test)
print('train: ',train.shape)
print('test; ',test.shape)
# align test and train
target=train.TARGET
train,test=train.align(test,join='inner',axis=1)
train['TARGET']=target
print('train: ',train.shape)
print('test; ',test.shape)
# check anomalies
(train['DAYS_BIRTH']/-365).describe() # normal
train['DAYS_EMPLOYED'].describe()
(train['DAYS_EMPLOYED']/365).describe()  # not normal
train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
train['DAYS_EMPLOYED'].plot.hist(title='employed days')
plt.xlabel('days')
test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
test['DAYS_EMPLOYED'].plot.hist(title='employed days')
plt.xlabel('days')
# correlations between fetures and the target
"""
.00-.19 “very weak”
.20-.39 “weak”
.40-.59 “moderate”
.60-.79 “strong”
.80-1.0 “very strong”
"""
corr=train.corr()['TARGET'].sort_values()
print(corr.head(15))
print('-------------------------------------------------')
print(corr.tail(15))
train['DAYS_BIRTH']=abs(train['DAYS_BIRTH'])
train['DAYS_BIRTH'].corr(train['TARGET']) # older people tend to pay credit
plt.figure(figsize=(10,8))
sns.kdeplot(train.loc[train['TARGET']==0,'DAYS_BIRTH']/365,label='target==0')
sns.kdeplot(train.loc[train['TARGET']==1,'DAYS_BIRTH']/365,label='target==1')
plt.xlabel('age(years)')
plt.ylabel('density')
plt.title('age distribution')
age_data=train[['DAYS_BIRTH','TARGET']]
age_data['YEAR_BIRTH']=age_data['DAYS_BIRTH']/365
age_data['YEAR_BIND']=pd.cut(age_data['YEAR_BIRTH'],bins=np.linspace(20,70,num=11))
age_data.head()
age_group=age_data.groupby('YEAR_BIND').mean()
age_group
plt.figure(figsize=(8,8))
plt.bar(age_group.index.astype(str),100*age_group.TARGET)
plt.xticks(rotation=75)
plt.xlabel('age')
plt.ylabel('target')
plt.title('Failure to Repay by Age Group')
# exterior sources
ext_data=train[['TARGET','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs=ext_data.corr()
ext_data_corrs
plt.figure(figsize=(8,6))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
# polynomial
poly_features = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))
# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = train['SK_ID_CURR']
app_train_poly = train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
app_test_poly = test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)
train.columns
from sklearn.preprocessing import MinMaxScaler, Imputer
