# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('../input/train.csv')
train_dataset.head()
train_dataset.tail()
y_train = train_dataset.iloc[:, -1]
y_train.head()
X_train = train_dataset.iloc[:, :-1]
X_train.head()
test_dataset = pd.read_csv('../input/test.csv')
test_dataset.head()
X_test = test_dataset
X_test.head()
all_data = pd.concat((X_train, X_test)).reset_index(drop=True)
obj_df_train = X_train.select_dtypes(include=['object']).copy()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

for col in obj_df_train:
    lb_make = LabelEncoder() 
    lb_make.fit(list(X_train[col].values)) 
    X_train[col] = lb_make.transform(list(X_train[col].values))


X_train.head()
import matplotlib.pyplot as plt
import seaborn as sns

corrmat = train_dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_dataset[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

sayac = 0
miss_columns = []

for index in range(len(missing_data['Missing Ratio'])):
    if missing_data['Missing Ratio'][index] != 0.0:
        sayac += 1
        miss_columns.append(missing_data['Missing Ratio'][index])
        
if sayac > 0:
    print('There are missing values')
else:
    print('There is not missing value')
    
#all_data = all_data.drop("MiscFeature", axis=1)
train_dataset_na = (train_dataset.isnull().sum() / len(train_dataset)) * 100
test_dataset_na = (X_test.isnull().sum() / len(X_test)) * 100

missing_data = pd.DataFrame({'Missing Ratio' :train_dataset_na})
missing_test_data = pd.DataFrame({'Missing Ratio' :test_dataset_na})
missing_test_data
most_missing_columns = []
missing_indices = []

for index in range(len(missing_data)):
    if missing_data['Missing Ratio'][index] >= 80:
        most_missing_columns.append(missing_data['Missing Ratio'][index])
        missing_indices.append(index)

most_missing_columns
most_missing_test_columns = []
missing_test_indices = []

for index in range(len(missing_test_data)):
    if missing_test_data['Missing Ratio'][index] >= 80:
        most_missing_test_columns.append(missing_test_data['Missing Ratio'][index])
        missing_test_indices.append(index)

most_missing_columns
missing_test_indices
missing_indices
missing_columns = list(missing_data['Missing Ratio'].index)
missing_columns
for indices in missing_indices:
    train_dataset = train_dataset.drop(missing_columns[indices], axis=1)
    test_dataset = test_dataset.drop(missing_columns[indices], axis=1)
train_dataset.head()
test_dataset.head()
corr_matrix = train_dataset.corr().abs()
corr_matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper
dropped = [column for column in upper.columns if any(upper[column] > 0.90)]
train_dataset = train_dataset.drop(train_dataset.columns[dropped], axis=1)
dropped
test_corr_matrix = test_dataset.corr().abs()
#upper = test_corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#dropped_test = [column for column in upper.columns if any(upper[column] > 0.90)]
test_corr_matrix
missing_data = missing_data[missing_data['Missing Ratio'] != 0]

from sklearn.preprocessing import LabelEncoder

obj_df_train = train_dataset.select_dtypes(include=['object']).copy()
categorical_features = obj_df_train.columns

missing_data.index.values
categorical_missing_features = []
numerical_missing_features = []

for data in missing_data.index.values:
    if data in categorical_features.values:
        categorical_missing_features.append(data)
    else:
        numerical_missing_features.append(data)
        
#categorical_missing_features.append('Fence')
#categorical_missing_features.append('Alley')
numerical_missing_features
from sklearn.impute import SimpleImputer

train_data_numerical = pd.DataFrame(train_dataset, columns=numerical_missing_features)

train_data_numerical = train_data_numerical.drop('PoolQC', axis=1) #because poolQC have so many nan values 
train_data_numerical = train_data_numerical.drop('Fence', axis=1) #because Fence have categorical values
train_data_numerical = train_data_numerical.drop('Alley', axis=1) #because Alley have categorical values
train_data_numerical = train_data_numerical.drop('MiscFeature', axis=1)
test_data_numerical = pd.DataFrame(test_dataset, columns=numerical_missing_features)

test_data_numerical = test_data_numerical.drop('PoolQC', axis=1) #because poolQC have so many nan values 
test_data_numerical = test_data_numerical.drop('Fence', axis=1) #because Fence have categorical values
test_data_numerical = test_data_numerical.drop('Alley', axis=1) #because Alley have categorical values
test_data_numerical = test_data_numerical.drop('MiscFeature', axis=1)
train_data_numerical.head()
test_data_numerical.head()
train_data_numerical = train_data_numerical.fillna(train_data_numerical.mean())
test_data_numerical = test_data_numerical.fillna(test_data_numerical.mean())

#for data in numerical_missing_features:
    #imputer = SimpleImputer(missing_values='NaN', strategy='mean')
    #imputer = imputer.fit(X_train_numerical)
#    imputed_numerical_data = imputer.fit_transform(X_train_numerical[data])
train_data_numerical['GarageYrBlt'] = train_data_numerical['GarageYrBlt'].astype('int')
train_data_numerical
test_data_numerical['GarageYrBlt'] = test_data_numerical['GarageYrBlt'].astype('int')
test_data_numerical.head()
train_data_categorical = pd.DataFrame(train_dataset, columns=categorical_missing_features)
train_data_categorical
test_data_categorical = pd.DataFrame(test_dataset, columns=categorical_missing_features)
test_data_categorical.head()
categorical_missing_features
#obj_df_train = train-data.select_dtypes(include=['object']).copy()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

for col in categorical_missing_features:
    lb_make = LabelEncoder() 
    lb_make.fit(list(train_data_categorical[col].values)) 
    train_data_categorical[col] = lb_make.transform(list(train_data_categorical[col].values))
    
train_data_categorical
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

for col in categorical_missing_features:
    lb_make = LabelEncoder() 
    lb_make.fit(list(test_data_categorical[col].values)) 
    test_data_categorical[col] = lb_make.transform(list(test_data_categorical[col].values))
    
test_data_categorical.head()
train_categoriacal_na = (train_data_categorical.isnull().sum() / len(train_data_categorical)) * 100

missing_data = pd.DataFrame({'Missing Ratio' :train_categoriacal_na})

sayac = 0
miss_columns = []

for index in range(len(missing_data['Missing Ratio'])):
    if missing_data['Missing Ratio'][index] != 0.0:
        sayac += 1
        miss_columns.append(missing_data['Missing Ratio'][index])
        
if sayac > 0:
    print('There are missing values')
else:
    print('There is not missing value')
    
train_data_numerical = train_data_numerical.fillna(train_data_numerical.mean())
train_data = pd.concat((train_data_numerical, train_data_categorical), axis=1)
test_data = pd.concat((test_data_numerical, test_data_categorical), axis=1)
train_data.head()
test_data.head()
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(train_data, y_train)
y_pred = regressor.predict(test_data)
columns = ['Id', 'SalePrice']
indices = []
for index in range(1461, 2920):
    indices.append(index)
indices[-1]
y_pred_frame = pd.DataFrame(columns=columns)
y_pred_frame['SalePrice'] = y_pred
y_pred_frame['Id'] = indices
y_pred_frame
y_pred_frame.to_csv('submission.csv', sep='\t', encoding='utf-8')
