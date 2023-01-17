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
#             print("******************************")
#             print("Column: ",col)
#             print("dtype before: ",props[col].dtype)
            
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
#             print("dtype after: ",props[col].dtype)
#             print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props
train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv", index_col='TransactionID')
train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv", index_col='TransactionID')
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv", index_col='TransactionID')
test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv", index_col='TransactionID')


# let's combine the data and work with the whole dataset
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')
del train_identity, train_transaction, test_identity, test_transaction
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
null_percent = (train.isnull().sum()/train.shape[0])*100
columns_to_drop = np.array(null_percent[null_percent > 50].index)

columns_to_drop
# Drop columns 

train.drop(columns_to_drop, axis=1, inplace=True)

print(train.shape)

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
cat_cols
train[cat_cols] = train[cat_cols].apply(lambda col: le.fit_transform(col))
train
X_train = train.drop('isFraud', axis=1)
y_train = train['isFraud']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state=42)

print(X_train.shape)
print(X_val.shape)
model = RandomForestClassifier(
    max_depth=20, 
    min_samples_leaf=4,
    min_samples_split=10, 
    max_features='sqrt',
    n_jobs=-1, 
    n_estimators=600
)

model.fit(X_train, y_train)

print(roc_auc_score(y_val,model.predict_proba(X_val)[:,1] ))
N = 10
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)

# create a dataframe
importances_df = pd.DataFrame({'variable':X_train.columns, 'importance': importances})

top_N = importances_df.sort_values(by=['importance'], ascending=False).head(N)

top_N