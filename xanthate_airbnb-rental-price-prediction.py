import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor 

from sklearn.preprocessing import scale, RobustScaler

from sklearn.metrics import r2_score, mean_squared_error



from xgboost import XGBRegressor
# read dataset

file_path = "../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv"

airbnb_df = pd.read_csv(file_path, index_col='id')

airbnb_df.head()
data_instances = airbnb_df.shape[0]

predictors = airbnb_df.shape[1]

print(f"Number of Data instances: {data_instances}")

print(f'Number of Predictors: {predictors}')
# let's have last look at the predictors before dropping them

airbnb_df.columns
# drop these columns

drop_predictors = ['host_id', 'host_name', 'name']

airbnb_df = airbnb_df.drop(drop_predictors, axis=1)

airbnb_df.head(3)
# datetime predictors

airbnb_df.last_review = pd.to_datetime(airbnb_df.last_review, infer_datetime_format=True)

airbnb_df.head(3)
# Null values

airbnb_df.isnull().sum()
# fill missing values in "reviews_per_month"

airbnb_df.reviews_per_month = airbnb_df.reviews_per_month.fillna(0)



# fill missing values in "last_review" feature

placeholder = min(airbnb_df.last_review)

airbnb_df.last_review = airbnb_df.last_review.fillna(placeholder)



# Null values

airbnb_df.isnull().sum()
airbnb_df.describe()
airbnb_df.head(3)
airbnb_df.neighbourhood_group.unique(), airbnb_df.neighbourhood_group.value_counts()
plt.figure(figsize=(8, 6))

sns.set(style='dark')

sns.countplot(x=airbnb_df.neighbourhood_group, data=airbnb_df)

plt.show()
len(airbnb_df.neighbourhood.unique()), airbnb_df.neighbourhood.unique()
airbnb_df.neighbourhood.value_counts()
plt.figure(figsize=(12, 40))

ax = sns.countplot(y=airbnb_df.neighbourhood, data=airbnb_df)

#f.set_xticklabels(rotation=90)

plt.show()
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 5))

sns.distplot(airbnb_df.latitude, ax=ax[0])

sns.distplot(airbnb_df.longitude, ax=ax[1])

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x=airbnb_df.latitude, y=airbnb_df.longitude)

plt.show()
airbnb_df.room_type.unique(), airbnb_df.room_type.value_counts()
plt.figure(figsize=(6, 5))

sns.countplot(x=airbnb_df.room_type, data=airbnb_df)

plt.show()
fig, ax = plt.subplots(2, 2, figsize=(20, 12))

sns.distplot(np.log1p(airbnb_df.minimum_nights), kde=True, ax=ax[0][0])

#ax[0][0].set_yscale('log')

ax[0][0].set_ylabel('count')

sns.distplot(airbnb_df.number_of_reviews, kde=False, ax=ax[0][1])

#ax[0][1].set_yscale('log')

ax[0][1].set_ylabel('count')

sns.distplot(airbnb_df.calculated_host_listings_count, kde=False, ax=ax[1][0])

ax[1][0].set_yscale('log')

ax[1][0].set_ylabel('count')

sns.distplot(airbnb_df.availability_365, kde=False, ax=ax[1][1])

ax[1][1].set_yscale('log')

ax[1][1].set_ylabel('count')

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.distplot(airbnb_df.price, kde=True, ax=ax[0])

sns.distplot(np.log1p(airbnb_df.price), kde=True, ax=ax[1])

plt.show()
airbnb_df.head(3)
X = airbnb_df

X.head(3)
# Transform the days into four categories

X.availability_365 = X.availability_365.map(

    lambda days: 'Zero' if days==0 else('Low' if(days >= 1 and days < 100) else ('Moderate' if(days >= 100 and days <250) else 'High')))
X.head(3)
categorical_predictors = ['neighbourhood_group', 'neighbourhood', 'room_type', 'availability_365']

X = pd.get_dummies(X, columns=categorical_predictors)

X.head()
X.price = np.log1p(X.price)

X.head(3)
y = X.price

X = X.drop(['price', 'last_review'], axis=1)
# train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

X_train.shape, y_test.shape
scaler = RobustScaler()

X_train_ht = scaler.fit_transform(X_train)

X_test_ht = scaler.fit_transform(X_test)
# k-fold cross-validation

kf = KFold(n_splits=5, shuffle=True, random_state=18)
# simple linear regression model

linear_reg = LinearRegression()

cv_score = cross_val_score(linear_reg, X_train, y_train, scoring='r2', cv=kf)

cv_score # cross-validation score
# Random Forest Regressor

kf_ht = KFold(n_splits=5, shuffle=True, random_state=27)

random_forest = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=20, min_samples_split=2)

cv_score_ht = cross_val_score(random_forest, X_train_ht, y_train, scoring='r2', cv=kf_ht)

cv_score_ht
#xgboost regressor

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping=5)

cv_score_xgb = cross_val_score(random_forest, X_train_ht, y_train, scoring='r2', cv=kf_ht)

cv_score_xgb  #cross-val score
# training

xgb.fit(X_train_ht, y_train)

# prediction

y_predict_xgb = xgb.predict(X_test_ht)

# evaluation

r2_score(y_test, y_predict_xgb)
'''

def multiple_model_hyperparameter_tuning(X, y):

    __param__ = {'n_estimators': [100, 120, 130, 140, 150], 

                 'criterion': ['mse'],

                 'max_depth': [10, 20, 30, 40, 50],

                 'min_samples_split': [2,5,10]}

    

    random_forest_ht = RandomForestRegressor()

    model_name = 'Random Forest Regressor'

    clf = GridSearchCV(random_forest_ht, __param__, scoring='r2', cv=kf_ht)

    out = clf.fit(X, y)

    print(model_name, ": ", out.best_score_)

    print(model_name, ": ", out.best_estimator_)

    

multiple_model_hyperparameter_tuning(X_train_ht, y_train)

'''