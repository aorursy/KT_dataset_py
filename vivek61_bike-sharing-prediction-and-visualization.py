import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



sns.set_palette('husl')

sns.set_style("whitegrid")



from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, PowerTransformer

from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV



from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# reading dataset



df = pd.read_csv('../input/bike-sharing/bike.csv', parse_dates=True, encoding = "latin1")

df.head()
# df.tail()
# df.sample()
df.info()
df.describe()
# changing column names





df.head()
df.columns = ['date', 'season', 'hr', 'holiday', 'day_of_week', 

              'working_day', 'weather_type', 'temp', 'temp_feels', 

              'humidity', 'wind_speed', 'casual_users', 'reg_users', 'total']



df['date'] = pd.to_datetime(df['date'])



df['day'] = pd.DatetimeIndex(df['date']).day

df['month'] = pd.DatetimeIndex(df['date']).month



df['year'] = pd.DatetimeIndex(df['date']).year

df['year']=df['year']-2011



df["season"] = df['season'].map({1 : "Spring", 

                                 2 : "Summer", 

                                 3 : "Fall", 

                                 4 : "Winter" })



df["day_of_week"] = df['day_of_week'].map({0 : "Sunday", 

                                           1 : "Monday", 

                                           2 : "Tuesday", 

                                           3 : "Wednesday", 

                                           4 : "Thursday", 

                                           5 : "Friday",

                                           6 : "Saturday" })

# Weather type

# 1 : " Clear + Few clouds + Partly cloudy + Partly cloudy"

# 2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist "

# 3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds"

# 4 : " Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# missing values

df.isna().sum()
# df['wind_speed'].quantile([0.25, 0.75])
# cat_cols = ['day_of_week', 'holiday', 'working_day']

# for i in cat_cols:

#     df[i] = df[i].astype('category')
# value counts



# for i in df.columns:

#     print(df[i].value_counts())
# # target column distribution



# plt.figure(figsize=(16,6))

# sns.countplot(df['total'], kde=False)
# sns.pairplot(df)
# # heatmap

# plt.figure(figsize=(10, 8))

# sns.heatmap(df.corr(), annot=True, fmt='.2f', center=0)
# date vs count plot



plt.figure(figsize=(16,6))

plt.plot(df['date'], df['total'], alpha=0.8)

plt.show()
# date vs temp plot



plt.figure(figsize=(16,6))

plt.plot(df['date'], df['temp_feels'], alpha=0.8, color='orange')
sns.set_palette('RdBu_r')
fig, axes = plt.subplots(figsize=(15, 4), ncols=3)

sns.barplot(x='season', y='total', data=df, ax=axes[0])

sns.barplot(x='working_day', y='total', data=df, ax=axes[1])

sns.barplot(x='holiday', y='total', data=df, ax=axes[2])



fig, axes = plt.subplots(figsize=(15, 4), ncols=2)

sns.barplot(x='month', y='total', data=df, ax=axes[0])

sns.barplot(x='day_of_week', y='total', data=df, ax=axes[1])
sns.set_palette('rocket')

fig, axes = plt.subplots(figsize=(15, 4))

sns.barplot(x='day', y='total', data=df, palette='Greens')



sns.set_palette('rocket')

fig, axes = plt.subplots(figsize=(15, 4))

sns.barplot(x='hr', y='total', data=df, color='Grey')
# for i in ['hr', 'month', 'day', 'day_of_week', 'season']:

#     plt.figure(figsize=(16,6))

#     # plt.bar(df['hr'], df['temp'], alpha=0.8)

#     sns.barplot(x=i, y="total", data=df, estimator=np.mean)
fig, axes = plt.subplots(figsize=(15, 10), ncols=2, nrows=2)

sns.lineplot(x='hr', y='total', hue='season', data=df, ax=axes[0][0])

sns.lineplot(x='hr', y='total', hue='day_of_week', data=df, ax=axes[0][1])

sns.lineplot(x='hr', y='total', hue='weather_type', data=df, ax=axes[1][0])

sns.lineplot(x='hr', y='total', hue='holiday', data=df, ax=axes[1][1])
plt.figure(figsize=(10,6))

sns.boxplot(data=df[['temp_feels','humidity','wind_speed']], palette="Set2")
df.head()
# min-max scaling

features=['temp', 'temp_feels', 'humidity', 'wind_speed']

for i in features:

    scaler = MinMaxScaler()

    df[i] = scaler.fit_transform(df[[i]])

    

# one hot encoding using pandas get_dummies

features=['weather_type', 'season']

for i in features:

    temp=pd.get_dummies(df[i], prefix=i, prefix_sep='_')

    df=pd.concat([df,temp], axis=1)

    df=df.drop(i, axis=1)

    

# cyclic encoding cyclic variables

def cyc_enc(df, col, max_vals):

    df[col+'_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col+'_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df

df = cyc_enc(df, 'hr', 24)

df = cyc_enc(df, 'month', 12)

df = cyc_enc(df, 'day', 31)

    

# PCA to reduce components

pca = PCA(n_components=1)

df['temperature'] = pca.fit_transform(df[['temp','temp_feels']])

df = df.drop(columns=['temp', 'temp_feels'])
# plt.figure(figsize=(16,6))

# plt.boxplot(df[['temp_feels','humidity','wind_speed']])
# # final correlation matrix



plt.figure(figsize=(14, 14))

sns.heatmap(df.corr(), annot=True, fmt='.2f')
df.head()
X = df.drop(['date','total', 'day_of_week', 'reg_users', 'casual_users'], axis=1)

y = df['total']



X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)
# linear regression



lr=LinearRegression()

lr.fit(X_train, y_train)

y_pred=lr.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))



plt.scatter(y_test, y_pred)

plt.show()
# k nearest neighbours regressor



knn=KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))



plt.scatter(y_test, y_pred)

plt.show()
# decision tree regressor



dt=DecisionTreeRegressor()

dt.fit(X_train, y_train)

y_pred=dt.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))

# print(y_test[:10])

# print(y_pred[:10])



plt.scatter(y_test, y_pred)

plt.show()
# random forest regressor



rf=RandomForestRegressor(n_estimators=10)

rf.fit(X_train, y_train)

y_pred=rf.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))



plt.scatter(y_test, y_pred)

plt.show()
from lightgbm import LGBMRegressor

model=LGBMRegressor(boosting_type='gbdt', class_weight=None,

              colsample_bytree=0.6746393485503049, importance_type='split',

              learning_rate=0.03158974434726661, max_bin=55, max_depth=-1,

              min_child_samples=159, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=1458, n_jobs=-1, num_leaves=196, objective=None,

              random_state=18, reg_alpha=0.23417614793823338,

              reg_lambda=0.33890027779706655, silent=False,

              subsample=0.5712459474269626, subsample_for_bin=200000,

              subsample_freq=1)



model.fit(X_train, y_train)

y_pred=model.predict(X_test)



print(mean_squared_error(y_pred, y_test))

print(mean_absolute_error(y_pred, y_test))

print(y_test[:10])

print(y_pred[:10])



plt.scatter(y_test, y_pred)
# from xgboost import XGBRegressor

# model = XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1,

#                 max_depth = 15, n_estimators = 700, random_state=2019)
rf=RandomForestRegressor(n_estimators=10)

rf.fit(X_train, y_train)





hyperparameters = {"criterion": ["mse", "mae"],

                   "max_depth": [5, 10],

                   "max_features": ["log2", "sqrt"],

                   # 'min_samples_leaf' : range(2,5),

                   # 'min_samples_split' : range(2,5),

#                    "n_estimators": range(10,12)

}



grid = GridSearchCV(rf, 

                    param_grid=hyperparameters, 

                    cv=10)



grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



rf = grid.best_estimator_

y_pred = rf.predict(X_test)



print(grid.best_params_)

print(grid.best_estimator_)

print(grid.best_score_)



print(mean_squared_error(y_pred, y_test))

print(mean_absolute_error(y_pred, y_test))

print(y_test[:10])

print(y_pred[:10])

plt.scatter(y_test, y_pred)
# box plot - temp_feels, humidity, windspeed



# one hot encode - holiday, weather_type

# drop - temp, casual_users, reg_users

# min-max scale - temp_feels, humidity, windspeed

# cyclic encode - season, hr, day_of_week, month, day

# year - current_year - 2011



# X - all the required features

# y - target



# train-test split

# linear regression model

# fit

# rmse

# plot actural, predict





# ----------------------------------------------



#cross validation

#grid search

#cyclic encoding

#
rf.feature_importances_