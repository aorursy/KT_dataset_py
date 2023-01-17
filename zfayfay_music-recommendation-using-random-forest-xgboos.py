# Load Python libraries

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn import ensemble, metrics

import xgboost as xgb

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
df = pd.read_csv('../input/train.csv')
df.head()
df.shape
df = df.sample(frac=0.1)
df.info()
songs = pd.read_csv('../input/songs.csv')
songs.info()
df = pd.merge(df, songs, on='song_id', how='left')

del songs
df.info()
members = pd.read_csv('../input/members.csv')
df = pd.merge(df, members, on='msno', how='left')

del members
df.info()
# Replace NA

for i in df.select_dtypes(include=['object']).columns:

    df[i][df[i].isnull()] = 'unknown'

df = df.fillna(value=0)
df.info()
# Create Dates



# registration_init_time

df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='ignore')

df['registration_init_time_year'] = df['registration_init_time'].dt.year

df['registration_init_time_month'] = df['registration_init_time'].dt.month

df['registration_init_time_day'] = df['registration_init_time'].dt.day



# expiration_date

df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')

df['expiration_date_year'] = df['expiration_date'].dt.year

df['expiration_date_month'] = df['expiration_date'].dt.month

df['expiration_date_day'] = df['expiration_date'].dt.day
#Dates to categoty

df['registration_init_time'] = df['registration_init_time'].astype('category')

df['expiration_date'] = df['expiration_date'].astype('category')
# Object data to category

for col in df.select_dtypes(include=['object']).columns:

    df[col] = df[col].astype('category')

    

# Encoding categorical features

for col in df.select_dtypes(include=['category']).columns:

    df[col] = df[col].cat.codes
df.corr()
plt.figure(figsize=[7,5])

sns.heatmap(df.corr())

plt.show()
# Model with the best estimator

model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)

model.fit(df[df.columns[df.columns != 'target']], df.target)

df_plot = pd.DataFrame({'features': df.columns[df.columns != 'target'],

                        'importances': model.feature_importances_})

df_plot = df_plot.sort_values('importances', ascending=False)
plt.figure(figsize=[11,5])

sns.barplot(x = df_plot.importances, y = df_plot.features)

plt.title('Importances of Features Plot')

plt.show()
model.feature_importances_
df = df.drop(df_plot.features[df_plot.importances < 0.04].tolist(), 1)
list(df.columns)
target = df.pop('target')
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(df, target, test_size = 0.3)

model = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=250)

model.fit(train_data, train_labels)
predict_labels = model.predict(test_data)

print(metrics.classification_report(test_labels, predict_labels))