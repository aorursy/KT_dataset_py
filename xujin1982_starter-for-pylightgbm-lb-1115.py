import numpy as np

import pandas as pd

import time



from sklearn import metrics, model_selection

from sklearn.preprocessing import LabelEncoder, StandardScaler



from scipy.stats import skew, boxcox



from pylightgbm.models import GBMRegressor
# Load data

start = time.time() 

train_data = pd.read_csv('../input/train.csv')

train_size=train_data.shape[0]

test_data = pd.read_csv('../input/test.csv')

# Merge data

full_data=pd.concat([train_data,test_data])

del( train_data, test_data)
data_types = full_data.dtypes  

cat_cols = list(data_types[data_types=='object'].index)

num_cols = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)



id_col = 'id'

target_col = 'loss'

num_cols.remove('id')

num_cols.remove('loss')
SSL = StandardScaler()

skewed_cols = full_data[num_cols].apply(lambda x: skew(x.dropna()))

skewed_cols = skewed_cols[skewed_cols > 0.25].index.values

for skewed_col in skewed_cols:

    full_data[skewed_col], lam = boxcox(full_data[skewed_col] + 1)

for num_col in num_cols:

    full_data[num_col] = SSL.fit_transform(full_data[num_col].values.reshape(-1,1))
LBL = LabelEncoder()

for cat_col in cat_cols:

    full_data[cat_col] = LBL.fit_transform(full_data[cat_col])

    



lift = 200

full_columns = cat_cols + num_cols

train_x = full_data[:train_size][full_columns].as_matrix()

test_x = full_data[train_size:][full_columns].as_matrix()

train_y = np.log(full_data[:train_size].loss.values + lift)

ID = full_data.id[:train_size].values



X_train, X_val, y_train, y_val = model_selection.train_test_split(train_x, train_y, train_size=.80, random_state=42)
seed = 42



gbmr = GBMRegressor(

    exec_path='/path/to/your/LightGBM/lightgbm', # change this to your LighGBM path

    num_threads=4,

    boosting_type = 'gbdt',

    num_iterations=10000,

    learning_rate=0.01,

    num_leaves=90,

    max_bin = 2500,

    max_depth = -1,

    min_data_in_leaf=5,

    min_sum_hessian_in_leaf = 5,

    metric='l1',

    feature_fraction=0.7,

    feature_fraction_seed=seed,

    bagging_fraction=1.0,

    bagging_freq=0,

    bagging_seed=seed,

    lambda_l1 = 20,

    lambda_l2 = 0,

    metric_freq=1,

    early_stopping_round=200,

    verbose= False

)



gbmr.fit(X_train, y_train, test_data=[(X_val, y_val)])
y_test_preds = gbmr.predict(test_x)

y_test_preds=(np.exp(y_test_preds)-lift)

df_submission = pd.read_csv('../output/sample_submission.csv')

df_submission['loss'] = y_test_preds

df_submission.to_csv('submission.csv',index=False)