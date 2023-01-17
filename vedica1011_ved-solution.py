# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading file

df = pd.read_csv('/kaggle/input/regressionhousing/Housing.csv')

df.head()
df.info(memory_usage='deep')
# analysing distribution of continous columns

df.describe()
df.describe(percentiles=[0.25,0.30,0.50,0.75])
from plotly.offline import iplot

import plotly as py

import plotly.tools as tls

import cufflinks as cf

import seaborn as sns

import matplotlib.pyplot as plt
df.corr()
plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.boxplot(x = 'mainroad', y = 'price', data = df)

plt.subplot(2,3,2)

sns.boxplot(x = 'guestroom', y = 'price', data = df)

plt.subplot(2,3,3)

sns.boxplot(x = 'basement', y = 'price', data = df)

plt.subplot(2,3,4)

sns.boxplot(x = 'hotwaterheating', y = 'price', data = df)

plt.subplot(2,3,5)

sns.boxplot(x = 'airconditioning', y = 'price', data = df)

plt.subplot(2,3,6)

sns.boxplot(x = 'furnishingstatus', y = 'price', data = df)

plt.show()
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = df)

plt.show()
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'hotwaterheating', data = df)

plt.show()
# List of variables to map



varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})



# Applying the function to the housing list

df[varlist] = df[varlist].apply(binary_map)

df.head()
# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'



status = pd.get_dummies(df['furnishingstatus'])



# Check what the dataset 'status' looks like

status.head()
# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(df['furnishingstatus'], drop_first = True)



# Add the results to the original housing dataframe

housing = pd.concat([df, status], axis = 1)



# Now let's see the head of our dataframe.

housing.head()
# Drop 'furnishingstatus' as we have created the dummies for it

housing.drop(['furnishingstatus'], axis = 1, inplace = True)



housing.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)

print('Train_data:',df_train.shape)

print('Test_data:',df_test.shape)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.corr()
plt.figure(figsize = (16, 10))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
plt.figure(figsize=[6,6])

plt.scatter(df_train.area, df_train.price)

plt.show()
y_train = df_train.pop('price')

X_train = df_train
X_train.head()
y_train.head()
import statsmodels.api as sm



# Add a constant

X_train_lm = sm.add_constant(X_train[['area']])



X_train_lm.head()
# Create a first fitted model

lr = sm.OLS(y_train, X_train_lm).fit()
# Check the parameters obtained



lr.params
plt.scatter(X_train_lm.iloc[:, 1], y_train)

plt.plot(X_train_lm.iloc[:, 1], 0.126894 + 0.462192*X_train_lm.iloc[:, 1], 'r')

plt.show()
print(lr.summary())
# Assign all the feature variables to X

X_train_lm = X_train[['area', 'bathrooms']]
# Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)



lr = sm.OLS(y_train, X_train_lm).fit()



lr.params
# Check the summary

print(lr.summary())
#Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train)



lr_1 = sm.OLS(y_train, X_train_lm).fit()



lr_1.params
print(lr_1.summary())
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_lm.columns

vif['VIF'] = [variance_inflation_factor(X_train_lm.values, i) for i in range(X_train_lm.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Higher P-value so dropping

X = X_train.drop('semi-furnished', 1,)

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model

print(lr_2.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_lm.columns

vif['VIF'] = [variance_inflation_factor(X_train_lm.values, i) for i in range(X_train_lm.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X = X_train_lm.drop('basement', 1,)

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model

print(lr_2.summary())
vif = pd.DataFrame()

vif['Features'] = X_train_lm.columns

vif['VIF'] = [variance_inflation_factor(X_train_lm.values, i) for i in range(X_train_lm.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lr_2.predict(X_train_lm)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18) 
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']



df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()
y_test = df_test.pop('price')

X_test = df_test
# Adding constant variable to test dataframe

X_test_m4 = sm.add_constant(X_test)
X_test_m4 = X_test_m4.drop(["basement", "semi-furnished"], axis = 1)
# Predictions on test data

y_pred_m4 = lr_2.predict(X_test_m4)
# Plot the histogram of the error terms on test it should be normal

fig = plt.figure()

sns.distplot((y_test - y_pred_m4), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred_m4)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 



# Homoscadasticity

# Hetroscadasticity
from sklearn.metrics import r2_score

r2_score(y_test,y_pred_m4)
df = pd.DataFrame(np.random.randn(100,3), columns = ['A', 'B', 'C'])

df.head()

df['A'] = df['A'].cumsum() + 20

df['B'] = df['B'].cumsum() + 20

df['C'] = df['C'].cumsum() + 20

df.head()
sns.distplot(df['A']);
sns.pairplot(df);
penguins = sns.load_dataset("penguins")

sns.pairplot(penguins, hue="species")
sns.pairplot(penguins, hue="species", diag_kind="hist");
# More visualise when we will apply in local jupyter notebook-not sure about reason

sns.pairplot(penguins, kind="kde");
sns.pairplot(

    penguins,

    plot_kws=dict(marker="+", linewidth=1),

    diag_kws=dict(fill=False),

);

#! pip install chart_studio
plt.plot(df);
df.plot();
titanic = sns.load_dataset('titanic')

titanic.head()
# run in your jupyter notebook

#titanic.iplot(kind = 'bar', x = 'sex', y = 'survived', title = 'Survived', xTitle='Sex', yTitle='#Survived')
# run in your jupyter notebook

#cf.set_config_file(theme='polar')

#df.iplot(kind = 'bar', barmode='stack', bargap=0.5)
# run in your jupyter notebook

#df.iplot(kind = 'box')
#df.iplot(kind = 'area')
#df[['A', 'B']].iplot(kind = 'spread');
#df.iplot(kind='hist', bins = 25, barmode = 'overlay', bargap=0.5)
from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)



import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",

                   dtype={"fips": str})



import plotly.graph_objects as go



fig = go.Figure(go.Choroplethmapbox(geojson=counties, locations=df.fips, z=df.unemp,

                                    colorscale="Viridis", zmin=0, zmax=12,

                                    marker_opacity=0.5, marker_line_width=0))

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129})

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
import plotly.express as px



df = px.data.stocks()

fig = px.line(df, x='date', y="GOOG")

fig.show()
import plotly.express as px



df = px.data.stocks(indexed=True)-1

fig = px.bar(df, x=df.index, y="GOOG")

fig.show()
import plotly.express as px



df = px.data.stocks(indexed=True)-1

fig = px.area(df, facet_col="company", facet_col_wrap=2)

fig.show()