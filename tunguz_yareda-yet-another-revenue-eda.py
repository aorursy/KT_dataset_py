import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.shape
train_df.info()
train_df.isnull().values.any()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
test_df.shape
test_df.info()
test_df.isnull().values.any()
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train_df = load_df()
test_df = load_df("../input/test.csv")
train_df.head()
train_df.info()
train_df.isnull().values.any()
train_df_describe = train_df.describe()
train_df_describe
test_df.head()
test_df.shape
test_df.info()
test_df_describe = test_df.describe()
test_df_describe
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
sample_submission['PredictedLogRevenue'] = np.log(1.26)
sample_submission.to_csv('simple_mean.csv', index=False)
sample_submission.head()
