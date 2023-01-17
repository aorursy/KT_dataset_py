# Version v01-02

# Import all libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt # ploting the data

import seaborn as sns # ploting the data

import math # calculation
# Set up color blind friendly color palette

# The palette with grey:

cbPalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# The palette with black:

cbbPalette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]



# sns.palplot(sns.color_palette(cbPalette))

# sns.palplot(sns.color_palette(cbbPalette))



sns.set_palette(cbPalette)

#sns.set_palette(cbbPalette)
# Load the dataset

#price_less = pd.read_csv('Data/MELBOURNE_HOUSE_PRICES_LESS.csv')

price_less = pd.read_csv('../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')

# price_full = pd.read_csv('Data/Melbourne_housing_FULL.csv')
price_less.info()
# price_full.info()
price_less.head(10)
# price_full.head(10)
# Determine the number of missing values for every column

price_less.isnull().sum()
# Exclude rows with missing prices

data_filtered = price_less.loc[price_less['Price'] > 0]

data_filtered.isnull().sum()
# price_full.isnull().sum()
data = data_filtered.copy()
data.describe()
data.columns
data['Price'].describe()
x = 'Price'

sns.set_palette("muted")

sns.distplot(data[x])

plt.ioff()

sns.set_palette(cbPalette)
# Log transform the Price variable to approach a normal distribution

x = np.log10(data["Price"])

sns.set_palette("muted")

sns.distplot(x)

plt.ioff()

sns.set_palette(cbPalette)
# data["Price"] = np.log10(data.loc[:, "Price"].values)
# data["Price"] = np.log10(data.loc[:, "Price"].values)
x = 'Rooms'

sns.set_palette("muted")

sns.distplot(data[x])

plt.ioff()

sns.set_palette(cbPalette)
x = 'Propertycount'

sns.set_palette("muted")

sns.distplot(data[x])

plt.ioff()

sns.set_palette(cbPalette)
x = 'Distance'

sns.set_palette("muted")

sns.distplot(data[x])

plt.ioff()

sns.set_palette(cbPalette)
data.head()
# data.shape
# data.columns
#  https://www.datacamp.com/community/tutorials/categorical-data

data['Suburb'].value_counts().count()
#  https://www.datacamp.com/community/tutorials/categorical-data

data['Address'].value_counts().count()
# https://seaborn.pydata.org/generated/seaborn.countplot.html

title = 'Count of properties per Region'

sns.countplot(y = data['Regionname'])

plt.title(title)

plt.ioff()
# https://seaborn.pydata.org/generated/seaborn.countplot.html

title = ''

sns.countplot(y = data['Type'])

plt.title(title)

plt.ioff()
#  https://www.datacamp.com/community/tutorials/categorical-data

# data['Method'].value_counts()
title = ''

sns.countplot(y = data['Method'])

plt.title(title)

plt.ioff()
#  https://www.datacamp.com/community/tutorials/categorical-data

# price_less['CouncilArea'].value_counts().count()
# price_less['CouncilArea'].value_counts()
title = ''

plt.figure(figsize=(20,10))

sns.countplot(y = price_less['CouncilArea'])

plt.title(title)

plt.ioff()
# title = ''

# plt.figure(figsize=(20,10))

# sns.countplot(y = price_less['SellerG'])

# plt.title(title)

# plt.ioff()
data['SellerG'].value_counts().count()
data['Postcode'].value_counts().count()
data.head()
data.columns
# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html

sns.set_palette("muted")

x = 'Rooms'

# x = np.log10(data["Rooms"])

# y = 'Price'

y = np.log10(data["Price"])



title = ''

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
sns.set_palette("muted")

x = "Distance"

y = np.log10(data["Price"])



title = ''

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
sns.set_palette("muted")

x = np.log10(data["Price"] / data["Rooms"])

y = np.log10(data["Price"])



title = ''

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
sns.set_palette("muted")

x = np.log10(data['Propertycount'])

# x = np.log10(data['Propertycount'] / data["Rooms"])

y = np.log10(data["Price"])



title = ''

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x=x, y=y, data=data)

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
data.head()
data.columns
y="Type"

x=np.log10(data["Price"])



title = ""

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
y="Regionname"

x=np.log10(data["Price"])

# x="Price"



title = ""

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
y="Method"

x=np.log10(data["Price"])

# x="Price"



title = ""

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=x, y=y, data=data, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()
sns.set_palette("muted")

y="CouncilArea"

x=np.log10(data["Price"])

# x="Price"



title = ""

f, ax = plt.subplots(figsize=(25, 10))

sns.boxplot(x=x, y=y, data=data, notch=False, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(title)

plt.ioff()

sns.set_palette(cbPalette)
# https://stackoverflow.com/questions/31468176/setting-values-on-a-copy-of-a-slice-from-a-dataframe

data['Price/Rooms'] = (data.loc[:, "Price"] / data.loc[:, "Rooms"])
data.columns
data.drop(['Date', 'Address'], axis=1, inplace=True)
data.head()
# Encoding categorical data

# https://pbpython.com/categorical-encoding.html

data = pd.get_dummies(data, columns=['Suburb', 'Rooms', 'Type',  'Method', 'SellerG', 'Regionname', 'CouncilArea'], drop_first=True)
data.info()
# Split the dataset

X = data.drop('Price', axis=1).values

y = data['Price'].values

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

df = pd.DataFrame({'Actual': np.round(y_test, 2), 

                   'Predicted': np.round(y_pred, 2)})

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