#importing necessery libraries for future analysis of the dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer



csv_file = '../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df = pd.read_csv(csv_file)

df.info()
df.isna().sum()
df[df.last_review.isna()].number_of_reviews.value_counts()
# Drop features `name`, `host_name`, `last_review`

df.drop(columns=['id', 'host_name', 'last_review', 'host_id', 'name'], inplace=True)

df.head()
# Fill NaN value with 0 for feature reviews_per_month

df.fillna({'reviews_per_month':0}, inplace=True)

df.head()
cor = df.corr()

plt.figure(figsize=(8,6))

sns.heatmap(cor, annot=True)
# Convert catogorical to numeric & check the correlations 

lb = LabelEncoder()

df_num = df.copy()

df_num['neighbourhood_group_numeric'] = LabelEncoder().fit_transform(df_num['neighbourhood_group'])

df_num['room_type_numeric'] = LabelEncoder().fit_transform(df_num['room_type'])



cor = df_num.corr()

plt.figure(figsize=(8,6))

sns.heatmap(cor, annot=True)
sns.violinplot(data=df[df.price < 250], x='neighbourhood_group', y='price')
from sklearn.model_selection import train_test_split



X = df_num[df_num.price <= 250]



price = X.price

X = X.drop(columns=['neighbourhood', 'neighbourhood_group', 'room_type', 'price'])



tmp = X.copy()

for f in ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 

          'calculated_host_listings_count', 'availability_365']:

    tmp[f] = StandardScaler().fit_transform(X[f].values.reshape(-1, 1))



Xtrain, Xtest, ytrain, ytest = train_test_split(tmp, price, random_state=3)
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score



xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping=5, silence=False)

xgb.fit(Xtrain, ytrain)

ypred = xgb.predict(Xtest)

print('RMSE value is', np.sqrt(mean_squared_error(ytest, ypred)))
