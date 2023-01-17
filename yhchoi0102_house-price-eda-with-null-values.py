import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 100)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)
train.head()
# columns check
set(train.columns) - set(test.columns)
target_col = 'SalePrice'
null_col = []
for col in train.columns:
    null_value = train[col].isna().sum()
    if null_value != 0:
        null_col.append(col)
        print("{:>15} has {:>5} null values and {:7.2f}%".format(col, null_value, 100*null_value / len(train)))
        
print("\n총 {}개 변수 중 {}개 변수가 null 값을 가지고 비율은 {:.2f}%".format(len(train.columns), len(null_col), 100*len(null_col)/len(train.columns)))
many_null = [col for col in train.columns if train[col].isna().any() and train[col].isna().sum() / len(train) > 0.9]
train.info()
train.dtypes.value_counts()
continuous_col = [col for col in train.columns if train[col].dtypes == 'float64']
plt.figure(figsize=(20, 30))

for i, col in enumerate(continuous_col):
    ax = plt.subplot(3, 1, i+1)
    sns.scatterplot(col, 'SalePrice', data=train)
    plt.xlabel(col)
    plt.ylabel('SalePrice')
plt.show()
cat_col = [col for col in train.columns if train[col].dtypes == 'object']
len(cat_col)
# 각 변수 별 unique value 확인
for col in cat_col:
    print('{:>15} has {:5} unique values'.format(col, train[col].unique().size))
# 각 변수별 unique 값의 count plot을 그려보자
plt.figure(figsize=(40, 30))
for i, col in enumerate(cat_col):
    plt.subplot(8, 7, i+1)
    sns.countplot(train[col])
plt.show()
cat_col_2 = [col for col in train[cat_col] if train[col].unique().size == 2]
len(cat_col_2)
plt.figure(figsize=(12, 5))

for i, col in enumerate(cat_col_2):
    plt.subplot(1, 3, i+1)
    sns.countplot(train[col], color='#34495e')
    
plt.subplots_adjust()
plt.tight_layout()
plt.show()
cat_col_3 = [col for col in train[cat_col] if train[col].unique().size >= 3 if train[col].unique().size <= 10]
len(cat_col_3)
plt.figure(figsize=(40, 60))

for i, col in enumerate(cat_col_3):
    plt.subplot(14, 3, i+1)
    sns.countplot(train[col], color='#34495e')

plt.tight_layout()
plt.show()
cat_col_4 = [col for col in train[cat_col] if train[col].unique().size > 10]
len(cat_col_4)
plt.figure(figsize=(20, 8))

for i, col in enumerate(cat_col_4):
    plt.subplot(1, 3, i+1)
    sns.countplot(train[col], color='#34495e')
    plt.xticks(rotation=75)
    
plt.tight_layout()
ordinal_col = [col for col in train.columns if train[col].dtypes == 'int64']
ordinal_col.remove('Id')
ordinal_col.remove('SalePrice')
len(ordinal_col)
plt.figure(figsize=(40, 50))

for i, col in enumerate(ordinal_col):
    plt.subplot(10, 4, i+1)
    sns.distplot(train[col], color='#34495e', kde=False)
    
plt.subplots_adjust()
plt.show()

