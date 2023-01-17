import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_session = pd.read_csv('../input/sessions.csv')
df_session.action.value_counts()
df_session.action_detail.value_counts()
df_session.device_type.value_counts()
train_users = pd.read_csv('../input/train_users_2.csv')
train_users.gender.replace('-unknown-', np.nan, inplace=True)
users_nan = (train_users.isnull().sum() / train_users.shape[0]) * 100 
users_nan[users_nan > 0]
train_users[train_users.country_destination=='NDF'].count()
train_users[(train_users.date_first_booking.isnull()) & (train_users.country_destination != 'NDF')].count()
test_users = pd.read_csv('../input/test_users.csv')
test_users.gender.replace('-unknown-', np.nan, inplace=True)
test_users_nan = (test_users.isnull().sum() / test_users.shape[0]) * 100 
test_users_nan[users_nan > 0]