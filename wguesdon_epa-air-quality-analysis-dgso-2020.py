# Import all libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting the data

import seaborn as sns # ploting the data

import math # calculation
file = '/kaggle/input/epa-air-quality/air_status.csv'

df = pd.read_csv(file)



# Visualize data info

df.info()
# Determine the number of missing values for every column

df.isnull().sum()
#examine the dataset

df.describe()
df
# Target variable distribution

# sns.distplot(df['NHNO3'])

sns.distplot(df['TOTAL_NO3'])
# Target according to sites

# see https://stackoverflow.com/questions/54132989/is-there-a-way-to-change-the-color-and-shape-indicating-the-mean-in-a-seaborn-bo



x='SITE_ID'

#y='NHNO3'

y='TOTAL_NO3'



sns.boxplot(x=x, y=y, data=df, notch=True, showmeans=True,

           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"});
# See https://stackoverflow.com/questions/36018681/stop-seaborn-plotting-multiple-figures-on-top-of-one-another

for x in ['TSO4', 'TNO3', 'TNH4', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'NSO4', 'WSO2', 'TOTAL_SO2', 'TOTAL_NO3', 'FLOW_VOLUME', 'VALID_HOURS']:

  plt.figure()

  sns.distplot(df[x]);
# Correlation

plt.figure(figsize=(20,10))

title = 'Correlation matrix of numerical variables'

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

plt.title(title)

plt.ioff()
df['Date'] =  pd.to_datetime(df['DATEON'])

df.dtypes
# Time Serie analysis of NO3

# See https://stackoverflow.com/questions/56150437/how-to-plot-a-time-series-graph-using-seaborn-or-plotly



x='Date'

y='TOTAL_NO3'



plt.figure(figsize=(40,20))

sns.lineplot(x=x, y=y, hue='SITE_ID' ,data=df);

plt.xticks(rotation=30);

plt.show()
df_codes = df['COMMENT_CODES']

df_codes.dropna()
# See https://www.kaggle.com/dfitzgerald3/randomforestregressor



# Select features

# From our EDA the following features seems the most useful to predict the toal NO3 for each site

selected_features = ['TOTAL_NO3', 'SITE_ID', 'DATEON', 'TSO4', 'TNH4', 

                     'Ca', 'Mg', 'Na', 'K', 'Cl', 'NSO4', 'WSO2', 'TOTAL_SO2', 

                     'FLOW_VOLUME', 'TEMP_SOURCE']



df_selected = df[selected_features]
# Recode data as categorical

# https://pbpython.com/categorical-encoding.html

# https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24



df_selected = pd.get_dummies(df_selected, drop_first=True)
# Replace missing values

# Ca, Mg, Na, K, Cl missing values

# Assumption here is that this correspond to measure faillure as fore NO3

# So I will use the average values when values are missing

# See https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns

# Is is important to avaoid leaking information from train to test dataset. 



from sklearn.model_selection import train_test_split

train, test = train_test_split(df_selected, test_size = 0.25, random_state = 0)
# See https://stackoverflow.com/questions/45090639/pandas-shows-settingwithcopywarning-after-train-test-split



pd.options.mode.chained_assignment = None

for feature in ['Ca', 'Mg', 'Na', 'K', 'Cl']:

  train[feature].fillna((train[feature].mean()), inplace=True)

  test[feature].fillna((test[feature].mean()), inplace=True)
train.isnull().sum()
test.isnull().sum()
# Split the dataset features and target

X_train = train.drop('TOTAL_NO3', axis=1).values

X_test = test.drop('TOTAL_NO3', axis=1).values

y_train = train['TOTAL_NO3'].values

y_test = test['TOTAL_NO3'].values
# See https://www.kaggle.com/dfitzgerald3/randomforestregressor



from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer
# See https://www.kaggle.com/dfitzgerald3/randomforestregressor

clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)



clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# Model evaluation R^2 score

# See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# see https://www.kaggle.com/nsrose7224/random-forest-regressor-accuracy-0-91

# See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html



clf.score(X_train, y_train)
clf.score(X_test, y_test)
# Compute MSE score

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)