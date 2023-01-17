# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# instantiate labelencoder object
le = LabelEncoder()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# function used from this notebook - kaggle.com/sunilsj99/fraud-detection-ieee/notebook

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props
train = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv", index_col="TransactionID")
train = reduce_mem_usage(train) 
test = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv", index_col="TransactionID")
test = reduce_mem_usage(test)
print(train.shape)
print(test.shape)
# Delete feautres that have null values more than 50%

null_percent = (train.isnull().sum()/train.shape[0])*100
columns_to_drop = np.array(null_percent[null_percent > 50].index)

columns_to_drop
# Drop columns 

train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

print(train.shape)
print(test.shape)
# Fill columns with null values
null_percent = (train.isnull().sum()/train.shape[0])*100
columns_to_fill = np.array(null_percent[null_percent > 0].index) 
columns_to_fill
# Fill columns with mode value (because all are categorical feautres i.e. dtype=object)

for i in columns_to_fill:
    train[i] = train[i].replace(np.nan, train[i].mode()[0])
    test[i]  = test[i].replace(np.nan, test[i].mode()[0])
cat_data = train.select_dtypes(include='object')
cat_cols = cat_data.columns.values

plt.figure(figsize=(20,15))

j=1
for i in cat_cols:
    
    # plotted below
    if (i == "P_emaildomain"):
        continue
    
    plt.subplot(3,3,j)
    sns.countplot(x=cat_data[i], palette="muted")
    j += 1
    
plt.show()
    
plt.figure(figsize=(20,8))
sns.countplot(x=cat_data['P_emaildomain'], palette='muted')
plt.xticks(rotation = 90)
plt.show()
# Numeric encoding features

train[cat_cols] = train[cat_cols].apply(lambda col: le.fit_transform(col))
# garbage collection

del cat_data, columns_to_drop, null_percent, columns_to_fill, cat_cols
# print(train.shape)
# print(test.shape)

X_train = train.drop('isFraud', axis=1)
y_train = train['isFraud']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state=0)

print(X_train.shape)
print(X_val.shape)
model = RandomForestClassifier(n_jobs=-1, n_estimators=200)
model.fit(X_train, y_train)

print(roc_auc_score(y_val,model.predict_proba(X_val)[:,1] ))