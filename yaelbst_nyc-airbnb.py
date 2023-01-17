# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from collections import Counter # Sort of useful dictionnary counting how many values for each element in a list
import seaborn as sns
%matplotlib inline 

from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
# %matplotlib inline so we see graphs within the code in the notebook
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Get data
data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv").set_index('id')
data.head(5)
# See essential info about dataset: column names, types, number of entries and non-null entries for each one
data.info()
# Fill empty and NaNs values with NaN
data = data.fillna(np.nan)
# Check for Null values: only 16 names and 21 host_names missing. 
# 10052 last_review dates and number of reviews per month missing. 
data.isnull().sum()
#replacing all NaN values in 'reviews_per_month' with 0
data.fillna({'reviews_per_month':0}, inplace=True)
#examing changes
data.reviews_per_month.isnull().sum()

#Although i reckon names and host names and last review will certainly have an influence on demand and thus on prices,
# I will drop those columns later on for this fast modelling
# Know for each column: number of entries, mean, standard deviation, min, max, and percentiles.
data.describe()
# Analysing the price: our target variable
sns.distplot(data['price'])
#skewness and kurtosis: looks very high!!!
print("Skewness: %f" % data['price'].skew())
print("Kurtosis: %f" % data['price'].kurt())
# 2 immediate observations about the price:
# A. Some places price is 0!!!: they have to be taken out.
# B. Some unusual places have really outrageously high prices: they should be taken out as well since our goal is to 
#model the price

#A. Get rid of the 0$ places
free_apts = data.loc[data['price']==0]
print(len(free_apts))
free_apts_ids = list(free_apts.index)
print(free_apts_ids)
# Drop the free appartments 
data = data.drop(free_apts_ids, axis = 0)
data.head(5)
# B. Keep only mainstream reasonnable priced appartments representing more than 97% of the data
main_data = pd.DataFrame(data.loc[data['price']< 500])
print(len(data))
print(len(main_data))
print(len(main_data)*100/len(data))
sns.distplot(main_data['price'])
print("Skewness: %f" % main_data['price'].skew())
print("Kurtosis: %f" % main_data['price'].kurt())

main_data.head(5)
# Feature analysis
# Numerical features

# Correlation matrix between numerical features: low correlation between the numerical features and the price
# Inverse correlation between price and longitude: sounds normal, price differs in different regions
corr_mat = sns.heatmap(main_data[['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',
'availability_365']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# General correlation matrix
# Positive correlation between number_of_reviews and reviews_per_month
corrmat = main_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);
#Relationship with categorical features
#box plot neighbourhood/price
# box plot: Q1 is price below which is 25% of the data, Q3 is 75 percentile, IQR = Q3 - Q1
# Outlier step above is Q3 +1.5IQR, below Q1 -1.5IQR, anything beyond is an outlier
# There are still quite a lot of outliers
var = 'neighbourhood_group'
nei_price = pd.concat([main_data['price'], main_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 8))
fig = sns.boxplot(x=var, y="price", data=nei_price)
fig.axis(ymin=0, ymax=500);
#scatter plot minimum_nights/price
# Seems like there is no correlation
var = 'minimum_nights'
minnights_to_price = pd.concat([main_data['price'], main_data[var]], axis=1)
minnights_to_price.plot.scatter(x=var, y='price', ylim=(0,500));
# על הדרך
# Obviously the minimum_nights number is a mistake for this record, 
#we can replace it maybe with the value of 50% of listings
main_data.loc[data['minimum_nights']==1250]
main_data.iloc[5767,9]= 3
main_data.iloc[5767]
#using violinplot to showcase density and distribtuion of prices 
# We can see higher prices in Manhattan of course, of course prices getting the highest for entire homes/apts
# In general more private rooms and shared rooms than entire homes on the market
fig_2=sns.violinplot(data=main_data, x='neighbourhood_group', y='price', hue= 'room_type')
fig_2.set_title('Density and distribution of prices for each neighberhood_group')
# Plot price according to latitude/longitude
plt.figure(figsize=(10,6))
fig_3=main_data.plot(kind='scatter', x='longitude',y='latitude',c='price',cmap=plt.get_cmap('jet'),colorbar=True);
plt.ioff()
# Plot room types according to latitude/longitude
plt.figure(figsize=(10,6))
fig_4 = sns.scatterplot(main_data.longitude,main_data.latitude,hue=main_data.room_type)
plt.ioff() # Turns the interactive mode off
# Plot neighbourhood_group according to latitude/longitude
plt.figure(figsize=(10,6))
fig_4 = sns.scatterplot(main_data.longitude,main_data.latitude,hue=main_data.neighbourhood_group)
plt.ioff() # Turns the interactive mode off

# Even though i believe name and hostname (wether this is ethical or not) can influence demand and thus the price
# at this stage i will not use them. All the more so as they have some null values.
# Will also get rid of last_review because of its null values. host_id also won't help with a regression. So will drop it
main_data.drop(['name','host_name','host_id','last_review'],axis=1,inplace=True)
main_data.head()
# The most interesting would be to get into NLP and analyse names to see the influence of the text chosen to describe the place
# Now i have to label-encode neighborhoud_group, neighbourhood and room_type
# I am going to go with LabelEncode from Sklearn 
# OneHotEncoder would add as many columns as there are different unique values for each one
# I am aware label encode can also introduce a wrong meaning using higher numbers for different for ex neighborhoods
# when there shouldn't be but i'll do it that way though in this exercise
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

col_to_encode = ['neighbourhood_group', 'neighbourhood', 'room_type']
for col in col_to_encode:
    main_data[col]= label.fit_transform(main_data[col])
    
main_data.head()
# Get the price: our target variable
Y_data = main_data['price']
main_data.drop(['price'],axis=1,inplace=True)
main_data.head()

X_train, X_test, y_train, y_test = train_test_split(main_data, Y_data, test_size=0.3, random_state=101)
from sklearn import metrics

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
from sklearn import linear_model

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)

print(X_test.index, lin_reg.predict(X_test))
print(y_test)
submission = pd.DataFrame({'id': X_test.index, 'price': lin_reg.predict(X_test)})
submission.to_csv('submission_airbnb.csv', index=False)
# Below: for my own use, just because i like to have the true price next to the prediction
#submission_wTrueResults = pd.DataFrame({'id': X_test.index, 'price': lin_reg.predict(X_test), 'true price': y_test})
#submission_wTrueResults.to_csv('submission_airbnb_wTrueResults.csv', index=False)
print_evaluate(y_test, lin_reg.predict(X_test))
# The best results were obtained with the simple linear regression
# I want to try with a Lasso linear regression = a linear regression with L1 regularization
# with constant alpha * L1 norm of the weights vector added to the loss
lasso_reg = linear_model.Lasso(alpha=0.1)
lasso_reg.fit(X_train,y_train)
lasso_pred = lasso_reg.predict(X_test)
print_evaluate(y_test, lasso_pred)
# Simple linear regression is still the best!
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
print_evaluate(y_test, regressor.predict(X_test))
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=main_data,label=Y_data)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
print_evaluate(y_test, preds)
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X_train, y_train = make_regression(n_features=10, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=4, random_state=0)
regr.fit(X_train, y_train)
print_evaluate(y_test, regr.predict(X_test))