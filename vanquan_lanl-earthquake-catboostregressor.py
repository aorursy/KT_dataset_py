!pip install catboost
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Model

from catboost import CatBoostRegressor, Pool



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16,

                                                 'time_to_failure': np.float64})

train.head()
pd.options.display.precision = 15 # Hiển thị 15 số thập phân

train.head()
train_chunk = pd.read_csv('../input/train.csv', chunksize=150_000, iterator=True, dtype={'acoustic_data': np.int16,

                                                                                          'time_to_failure': np.float64})
def feature_engineering(df):

    new_feat = []

    new_feat.append(df.mean())

    new_feat.append(df.std())

    new_feat.append(df.min())

    new_feat.append(df.max())

    

    return pd.Series(new_feat)
# Feature Engineering

X_train = pd.DataFrame()

y_train = pd.Series()



for df in train_chunk:

    fe = feature_engineering(df['acoustic_data'])

    X_train = X_train.append(fe, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1])) # Lấy giá trị cuối cùng của mỗi chunk
# Đặt tên cho các cột

columns = ['mean', 'std', 'min', 'max']

X_train.columns = columns



X_train.head()
y_train.head()
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
params = {'iterations': 10000,

          'loss_function': 'MAE'}



train_pool = Pool(X_train_scaled, y_train)



cbr = CatBoostRegressor(**params)

cbr.fit(X_train_scaled, y_train, eval_set=train_pool, silent=True)



y_pred = cbr.predict(X_train_scaled)

print('Best score: ' + str((cbr.best_score_)))

print('MAE: {:.3f}'.format(mean_absolute_error(y_train.values, y_pred)))
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id') # Đọc file sample_submission để lấy form



X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index) # Tạo DF mới cho X_test để thực hiện Feature Engineering tương tự với train data
X_test.head()
# Feature Engineering cho test data

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'mean'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()
X_test.head()
X_test_scaled = scaler.transform(X_test) # Scale feature ở test data



submission['time_to_failure'] = cbr.predict(X_test_scaled)

submission.to_csv('submission.csv')
submission.head()