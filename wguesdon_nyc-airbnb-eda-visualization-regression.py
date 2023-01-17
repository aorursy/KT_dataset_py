# Version v04-08

# Import all libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt # ploting the data

import seaborn as sns # ploting the data

import math # calculation
# load the data

#data = pd.read_csv('AB_NYC_2019.csv')

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

# data_copy = data.copy()
# Visualize data info

data.info()
# Drop the data that are not of interest and/or causing privacy issues

data.drop(['id','host_name','last_review'], axis=1, inplace=True)

# Visualize the first 5 rows

data.head()
# Determine the number of missing values for every column

data.isnull().sum()
#replacing all NaN values in 'reviews_per_month' with 0

# See https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb

data.fillna({'reviews_per_month':0}, inplace=True)
#examine the dataset

(data[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',

       'calculated_host_listings_count', 'availability_365']]

 .describe())
# Exclude property with listed price of 0

data = data.loc[data['price'] > 0]

# data_copy = data.copy()
#examine the dataset

data.describe()
# Recode data as categorical

# https://datascience.stackexchange.com/questions/29093/continuous-variable-to-categorical-by-quartiles

data_encoded = data.copy()

data_encoded['minimum_nights'] = pd.qcut(data['minimum_nights'], q=2, labels=["minimum_nights_low", "minimum_nights_high"])

data_encoded['number_of_reviews'] = pd.qcut(data['number_of_reviews'], q=3, labels=["number_of_reviews_low", "minimum_nights_medium", "number_of_reviews_high"])

data_encoded['reviews_per_month'] = pd.qcut(data['reviews_per_month'], q=2, labels=["reviews_per_month_low", "reviews_per_month_high"])

data_encoded['calculated_host_listings_count'] = pd.cut(data['calculated_host_listings_count'], 

                                                bins=[0, 2, 327],

                                                labels=["calculated_host_listings_count_low", "calculated_host_listings_count_high"])

data_encoded['availability_365'] = pd.qcut(data['availability_365'], q=2, labels=["availability_low", "availability_high"])
data_encoded.isnull().sum()
data_encoded.head()
sns.set_palette("muted")

from pylab import *

f, ax = plt.subplots(figsize=(8, 6))



subplot(2,3,1)

sns.distplot(data['price'])



subplot(2,3,2)

sns.distplot(data['minimum_nights'])



subplot(2,3,3)

sns.distplot(data['number_of_reviews'])



subplot(2,3,4)

sns.distplot(data['reviews_per_month'])



subplot(2,3,5)

sns.distplot(data['calculated_host_listings_count'])



subplot(2,3,6)

sns.distplot(data['availability_365'])



plt.tight_layout() # avoid overlap of plotsplt.draw()
from pylab import *

f, ax = plt.subplots(figsize=(8, 6))



subplot(2,3,1)

sns.boxplot(y = data['price']) 



subplot(2,3,2)

sns.boxplot(y = data['minimum_nights'])



subplot(2,3,3)

sns.boxplot(y = data['number_of_reviews'])



subplot(2,3,4)

sns.boxplot(y = data['reviews_per_month'])



subplot(2,3,5)

sns.boxplot(y = data['calculated_host_listings_count'])



subplot(2,3,6)

sns.boxplot(y = data['availability_365'])



plt.tight_layout() # avoid overlap of plots

plt.draw()
# Set up color blind friendly color palette

# The palette with grey:

cbPalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# The palette with black:

cbbPalette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]



# sns.palplot(sns.color_palette(cbPalette))

# sns.palplot(sns.color_palette(cbbPalette))



sns.set_palette(cbPalette)

#sns.set_palette(cbbPalette)
title = 'Properties per Neighbourhood Group'

sns.countplot(data['neighbourhood_group'])

plt.title(title)

plt.ioff()
title = 'Properties per Room Type'

sns.countplot(data['room_type'])

plt.title(title)

plt.ioff()
plt.figure(figsize=(20,10))

title = 'Correlation matrix of numerical variables'

sns.heatmap(data.corr(), square=True, cmap='RdYlGn')

plt.title(title)

plt.ioff()
# See https://www.kaggle.com/biphili/hospitality-in-era-of-airbnb

title = 'Neighbourhood Group Location'

plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group).set_title(title)

plt.ioff()



title = 'Room type location per Neighbourhood Group'

plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.room_type).set_title(title)

plt.ioff()
title = 'Room type location per Neighbourhood Group'

sns.catplot(x='room_type', kind="count", hue="neighbourhood_group", data=data);

plt.title(title)

plt.ioff()
#https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html

#http://seaborn.pydata.org/tutorial/color_palettes.html



x= 'neighbourhood_group'

y= 'price'

title = 'Price per Neighbourhood Group'



f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data)

plt.title(title)

plt.ioff()
# alternative visualization of median less impacted by the extreme values

# see https://www.kaggle.com/nidaguler/eda-and-data-visualization-ny-airbnb



title = 'Median Price per Neighbourhood Group'

result = data.groupby(["neighbourhood_group"])['price'].aggregate(np.median).reset_index().sort_values('price')

sns.barplot(x='neighbourhood_group', y="price", data=data, order=result['neighbourhood_group'])

plt.title(title)

plt.ioff()
# https://stackoverflow.com/questions/54132989/is-there-a-way-to-change-the-color-and-shape-indicating-the-mean-in-a-seaborn-bo

x='neighbourhood_group'

y='price'



title = 'Price per neighbourhood_group for Properties under $175'

data_filtered = data.loc[data['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()

f

title = 'Price per neighbourhood_group for Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# Is the location impact on price statiscaly significant?

# Use on way ANOVA and pairwise comaprison

# See https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/



import statsmodels.api as sm

from statsmodels.formula.api import ols



data_filtered = data.loc[data['price'] < 175]



mod = ols('price ~ neighbourhood_group',data=data_filtered).fit()



aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
pair_t = mod.t_test_pairwise('neighbourhood_group')

pair_t.result_frame
title = 'Price per Room Type for Properties under $175'

data_filtered = data.loc[data['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x='room_type', y='price', data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()



title = 'Price per Room Type for Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x='room_type', y='price', data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html

sns.set_palette("muted")

x = 'reviews_per_month'

y = 'price'



title = 'Price relation to number of review per month for Properties under $175'

data_filtered = data.loc[(data['price'] < 175) & (data['reviews_per_month'] < 30)]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()



title = 'Price relation to number of review per month for Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
x='reviews_per_month'

y='price'



title = 'Price per reviews_per_month categories for Properties under $175'

data_filtered = data_encoded.loc[data_encoded['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()



title = 'Price per reviews_per_month categories for Properties more than $175'

data_filtered = data_encoded.loc[data_encoded['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html

sns.set_palette("muted")

x = 'number_of_reviews'

y = 'price'



title = 'Price relation to number of review per month and Room Type for Properties under $175'

data_filtered = data.loc[data['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()



title = 'Price relation to number of review per month and Room Type for Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
x = 'number_of_reviews'

y='price'



title = 'Price per number_of_reviews categories for Properties under $175'

data_filtered = data_encoded.loc[data_encoded['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()



title = 'Price per number_of_reviews categories for Properties more than $175'

data_filtered = data_encoded.loc[data_encoded['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html

sns.set_palette("muted")

x = 'minimum_nights'

y = 'price'



title = 'Price relation to minimum_nights for Properties under $175'

data_filtered = data.loc[data['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()



title = 'Price relation to minimum_nights Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
x = 'minimum_nights'

y='price'



title = 'Price per minimum_nights categories for Properties under $175'

data_filtered = data_encoded.loc[data_encoded['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()



title = 'Price per minimum_nights categories for Properties more than $175'

data_filtered = data_encoded.loc[data_encoded['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html

sns.set_palette("muted")

x = 'calculated_host_listings_count'

y = 'price'



title = 'Price relation to calculated_host_listings_count for Properties under $175'

data_filtered = data.loc[data['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()



title = 'Price relation to calculated_host_listings_count for Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
x = 'calculated_host_listings_count'

y='price'



title = 'Price per calculated_host_listings_count categories for Properties under $175'

data_filtered = data_encoded.loc[data_encoded['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()



title = 'Price per calculated_host_listings_count categories for Properties more than $175'

data_filtered = data_encoded.loc[data_encoded['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html

sns.set_palette("muted")

x = 'availability_365'

y = 'price'



title = 'Price relation to availability for Properties under $175'

data_filtered = data.loc[data['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()



title = 'Price relation to availability for Properties more than $175'

data_filtered = data.loc[data['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data_filtered)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
x = 'availability_365'

y='price'



title = 'Price per calculated_host_listings_count categories for Properties under $175'

data_filtered = data_encoded.loc[data_encoded['price'] < 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()



title = 'Price per calculated_host_listings_count categories for Properties more than $175'

data_filtered = data_encoded.loc[data_encoded['price'] > 175]

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data_filtered, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
# Load the Dataset  

#data.drop(['latitude', 'name',], axis=1, inplace=True)

data.drop(['name'], axis=1, inplace=True)

data_copy = data.copy()
#data.head()
# # Determine the number of missing values for every column

# data.isnull().sum()
# log10 transform

# https://stackoverflow.com/questions/30794525/adding-one-to-all-the-values-in-a-dataframe

# data_copy = data.copy()

data.minimum_nights += 0.000000001

data['minimum_nights'] = np.log10(data['minimum_nights'])

data.number_of_reviews += 0.000000001

data['number_of_reviews'] = np.log10(data['number_of_reviews'])

data.reviews_per_month += 0.000000001

data['reviews_per_month'] = np.log10(data['reviews_per_month'])

data.calculated_host_listings_count += 0.000000001

data['calculated_host_listings_count'] = np.log10(data['calculated_host_listings_count'])

data.availability_365 += 0.000000001

data['availability_365'] = np.log10(data['availability_365'])
# Encoding categorical data

data = pd.get_dummies(data, columns=['room_type'], drop_first=True)

data = pd.get_dummies(data, columns=['neighbourhood'], drop_first=True)

data = pd.get_dummies(data, columns=['neighbourhood_group'], drop_first=True)
# Filter the dataset for prices between 50 and $175

data_filtered_low = data.loc[(data['price'] < 175)]
# data_filtered_low.shape
# Filter the dataset for prices superior to $175

data_filtered_high = data.loc[(data['price'] > 175)]
# data_filtered_high.shape
# Split the dataset

X = data_filtered_low.drop('price', axis=1).values

y = data_filtered_low['price'].values

y = np.log10(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = lr.predict(X_test)
# Compare predicted and actual values

# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://stackoverflow.com/questions/19100540/rounding-entries-in-a-pandas-dafaframe

df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
# Split the dataset

X = data_filtered_high.drop('price', axis=1).values

y = data_filtered_high['price'].values

y = np.log10(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = lr.predict(X_test)
df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
# Split the dataset

X = data_filtered_low.drop('price', axis=1).values

y = data_filtered_low['price'].values

y = np.log10(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=8, n_estimators = 100, random_state = 0)

rfr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = rfr.predict(X_test)
df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, rfr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, rfr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, rfr.predict(X_test), multioutput='variance_weighted'), 2))
# Split the dataset

X = data_filtered_high.drop('price', axis=1).values

y = data_filtered_high['price'].values

y = np.log10(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=8, n_estimators = 100, random_state = 0)

rfr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = rfr.predict(X_test)
df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, rfr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, rfr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, rfr.predict(X_test), multioutput='variance_weighted'), 2))
# # Combined Data and Data_ecoded

# data['availability_365_cat'] = data_encoded['availability_365']

# data.head()
data_encoded.drop(['name'], axis=1, inplace=True)
data_encoded.head()
# Encoding categorical data

data_encoded = pd.get_dummies(data_encoded, columns=['neighbourhood_group'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['neighbourhood'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['room_type'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['minimum_nights'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['number_of_reviews'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['reviews_per_month'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['calculated_host_listings_count'], drop_first=True)

data_encoded = pd.get_dummies(data_encoded, columns=['availability_365'], drop_first=True)
data_encoded.head()
# Data filtering

# Filter the dataset for prices between 50 and $175

data_filtered_low = data_encoded.loc[(data['price'] < 175)]

# Filter the dataset for prices superior to $175

data_filtered_high = data_encoded.loc[(data['price'] > 175)]
# Split the dataset

X = data_filtered_low.drop('price', axis=1).values

y = data_filtered_low['price'].values

y = np.log10(y)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = lr.predict(X_test)



# Compare predicted and actual values

# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://stackoverflow.com/questions/19100540/rounding-entries-in-a-pandas-dafaframe

df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
# Split the dataset

X = data_filtered_high.drop('price', axis=1).values

y = data_filtered_high['price'].values

y = np.log10(y)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = lr.predict(X_test)



df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
# Split the dataset

X = data_filtered_low.drop('price', axis=1).values

y = data_filtered_low['price'].values

y = np.log10(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=8, n_estimators = 100, random_state = 0)

rfr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = rfr.predict(X_test)



df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)
from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, rfr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, rfr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, rfr.predict(X_test), multioutput='variance_weighted'), 2))
# Split the dataset

X = data_filtered_high.drop('price', axis=1).values

y = data_filtered_high['price'].values

y = np.log10(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=8, n_estimators = 100, random_state = 0)

rfr.fit(X_train, y_train)



# Predicting the Test set results

y_pred = rfr.predict(X_test)



df = pd.DataFrame({'Actual': np.round(10 ** y_test, 0), 

                   'Predicted': np.round(10 ** y_pred, 0)})

df.head(10)



from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score
print('Price mean:', np.round(np.mean(y), 2))  

print('Price std:', np.round(np.std(y), 2))

print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, rfr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, rfr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, rfr.predict(X_test), multioutput='variance_weighted'), 2))