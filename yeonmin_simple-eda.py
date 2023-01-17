# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, BatchNormalization, Activation 
from keras.callbacks import EarlyStopping, ModelCheckpoint

import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/kaggletutorial/covertype_train.csv')
test = pd.read_csv('../input/kaggletutorial/covertype_test.csv')
train.shape
test.shape
train.head()
train.info()
train.describe()
dtype_df = train.dtypes.reset_index()
dtype_df.columns = ['column','dtype']
dtype_df.groupby(['dtype']).agg('count').reset_index()
train.isnull().sum()
missing_df = train.isnull().sum().reset_index()
missing_df.columns = ['column', 'count']
missing_df['ratio'] = missing_df['count'] / train.shape[0]
missing_df.loc[missing_df['ratio'] != 0]
set(train.columns) - set(test.columns)
train['Cover_Type'].value_counts() / train.shape[0] * 100
train['Cover_Type'].value_counts().plot(kind='bar')
plt.show()
plt.figure(figsize=(8,6))
plt.plot(train['Cover_Type'].cumsum())
plt.title('ID Leak')
plt.show()
category_feature = [ col for col in train.columns if train[col].dtypes == "object"]
category_feature
train[category_feature].head()
train[category_feature].nunique()
for col in category_feature:
    train[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()
soiltype_df = train.groupby(['Soil_Type','Cover_Type'])['Soil_Type'].count().unstack('Cover_Type')
soiltype_df
soiltype_df.plot(kind='bar', figsize=(20,10))
plt.title('SoilType')
plt.show()
wildeness_df = train.groupby(['Wilderness_Area','Cover_Type'])['Wilderness_Area'].count().unstack('Cover_Type')
wildeness_df
wildeness_df.plot(kind='bar', figsize=(20,10))
plt.title('Wilderness_Area')
plt.show()
oiltype_df = train.groupby(['oil_Type','Cover_Type'])['oil_Type'].count().unstack('Cover_Type')
oiltype_df.plot(kind='bar', figsize=(20,10))
plt.title('oil_Type')
plt.show()
all_data_cat = pd.concat([train[category_feature], test[category_feature]])
np.sum(np.abs(all_data_cat['Soil_Type'].factorize()[0] - all_data_cat['oil_Type'].factorize()[0]))
all_data_cat['Soil_Type']
all_data_cat['Soil_Type'].factorize()[0]
%timeit(np.sum(np.abs(train['Soil_Type'].factorize()[0] - train['oil_Type'].factorize()[0])))
all_data_cat['is_same'] = all_data_cat.apply(lambda row: 1 if row['Soil_Type']== row['oil_Type'] else 0 ,axis=1)
all_data_cat.loc[all_data_cat['is_same']==0]
%timeit(all_data_cat.apply(lambda row: 1 if row['Soil_Type']== row['oil_Type'] else 0 ,axis=1))
numerical_feature = list(set(train.columns) - set(category_feature) - set(['ID','Cover_Type']))
numerical_feature = np.sort(numerical_feature)
numerical_feature
for col in numerical_feature:
    sns.distplot(train.loc[train[col].notnull(), col])
    plt.title(col)
    plt.show()
for col in numerical_feature:
    col_value = train.loc[train[col].notnull(), col]
    
    fig, axs = plt.subplots(ncols=2,figsize=(10,4))
    sns.distplot(col_value, ax=axs[0])
    axs[0].set_title(col)
    sns.distplot(np.log1p(col_value), ax=axs[1])
    axs[1].set_title('Log transfrom {}'.format(col))
    plt.show()
train.loc[train['Vertical_Distance_To_Hydrology']<0].head()
sns.distplot(train['Vertical_Distance_To_Hydrology'])
plt.title('Vertical_Distance_To_Hydrology Distribution')
plt.show()
distance_feature = [col for col in train.columns if col.find('Distance') != -1 ]
distance_feature
sns.pairplot(train[distance_feature + ['Cover_Type']], hue='Cover_Type', 
             x_vars=distance_feature, y_vars=distance_feature, height=3)
plt.show()
other_numerical_feature = [col for col in numerical_feature if col.find('Distance') == -1]
other_numerical_df = train[other_numerical_feature + ['Cover_Type']]
other_numerical_df['Elevation'] = np.log1p(other_numerical_df['Elevation'])

sns.pairplot(other_numerical_df.dropna(), x_vars=other_numerical_feature, y_vars=other_numerical_feature, 
             hue='Cover_Type', height=3)
plt.show()
other_numerical_feature = [col for col in numerical_feature if col.find('Distance') == -1]
distance_numerical_df = train.copy()
distance_numerical_df['Elevation'] = np.log1p(distance_numerical_df['Elevation'])

sns.pairplot(distance_numerical_df.dropna(), x_vars=other_numerical_feature, y_vars=distance_feature, 
             hue='Cover_Type', height=3)
plt.show()

del distance_numerical_df
for col in train.loc[:,train.dtypes=='object'].columns:
    train[col] = train[col].factorize()[0]
wilderness_area_uniqlist = train['Wilderness_Area'].unique()

for col in numerical_feature:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Wilderness_Area', y=col, hue='Cover_Type', data=train.dropna())
    plt.title("Wilderness_Area - {}".format(col))
    plt.show()
    
    fig, axes = plt.subplots(nrows=4, figsize=(10,8))
    for index, wild in enumerate(wilderness_area_uniqlist):
        try:
            wild_frame = train.loc[train['Wilderness_Area']==wild].dropna()
            
            axes[index].set_title('Wilderness_Area {} vs {}'.format(wild, col))
            sns.distplot(wild_frame.loc[wild_frame['Cover_Type']==1,col], ax=axes[index])
            sns.distplot(wild_frame.loc[wild_frame['Cover_Type']==0,col], ax=axes[index])         
        except:
            pass
        else:
            del wild_frame
            gc.collect()
    plt.tight_layout()
    plt.show()
soil_frame = train.loc[train['Soil_Type']==1].dropna()   
soil_frame.loc[soil_frame['Cover_Type']==0, 'Aspect'].shape[0]
soil_frame.loc[soil_frame['Cover_Type']==1, 'Aspect'].shape[0]
soiltype_uniqlist = train['Soil_Type'].unique()

for col in numerical_feature:
    plt.figure(figsize=(16,8))
    sns.boxplot(x='Soil_Type', y=col, hue='Cover_Type', data=train.dropna())
    plt.title("Soil_Type - {}".format(col))
    plt.show()
    
    soiltype_uniqlist = train['Soil_Type'].unique()
    for index, soil in enumerate(soiltype_uniqlist):
        try:
            fig, axes = plt.subplots(ncols=2, figsize=(6,3))
            soil_frame = train.loc[train['Soil_Type']==soil].dropna()        
            sns.distplot(soil_frame[col], ax=axes[0])  
            sns.distplot(soil_frame.loc[soil_frame['Cover_Type']==1, col], ax=axes[1])  
            sns.distplot(soil_frame.loc[soil_frame['Cover_Type']==0, col], ax=axes[1]) 
            axes[0].set_title('Soil_Type {} \n{}'.format(soil, col))
            axes[1].set_title('CoverType')
            axes[1].legend([1,0])
            plt.tight_layout()
            plt.show()
        except:
            pass
        else: 
            del soil_frame
            gc.collect()