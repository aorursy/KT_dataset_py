import os

print(os.listdir("../input"))



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics, preprocessing

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from plotly import __version__

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
data_2017 = pd.read_csv('../input/2017.csv',index_col='Country')
data_2016 = pd.read_csv('../input/2016.csv',index_col='Country')
data_2015 = pd.read_csv('../input/2015.csv',index_col='Country')
data_2017.head()
data_2017.drop(['Whisker.high','Whisker.low'], axis=1, inplace=True)
data_2017.rename(index=str, columns={'Happiness.Rank': 'Happiness Rank',

                                     'Happiness.Score': 'Happiness Score',

                  'Economy..GDP.per.Capita.': 'Economy (GDP per Capita)',

                 'Health..Life.Expectancy.': 'Health (Life Expectancy)',

                 'Trust..Government.Corruption.': 'Trust (Government Corruption)',

                 'Dystopia.Residual': 'Dystopia Residual'}, inplace=True)
data_2017.head()
data_2016.head()
data_2016.drop(['Lower Confidence Interval', 'Upper Confidence Interval','Region'],axis=1,inplace=True)
data_2016.head()
data_2015.head()
data_2015.drop(['Region', 'Standard Error'],axis=1,inplace=True)
data_2015.head()
data_2015.columns
data_2016.columns
cols = data_2017.columns

cols
data_2015.info()
data_2016.info()
data_2017.info()
data_2015.describe()
data_2016.describe()
data_2017.describe()
def country_plot(country):

    cols = data_2015.columns[1:]

    r = len(cols)

    d = {}

    for n in range(r):

        a1 = data_2015.loc[country][cols[n]]

        a2 = data_2016.loc[country][cols[n]]

        a3 = data_2017.loc[country][cols[n]]

        d = {cols[n]: [a1,a2,a3]}

        df = pd.DataFrame(data=d,index=[2015,2016,2017])

        plt.figure(figsize=(15,50))

        plt.subplot(r, 1, n+1)

        plt.title(country + ' ' + cols[n])

        sns.barplot(x=df.index,y=df[cols[n]])
#example

country_plot('Poland')
for x in cols[1:]:

    a1 = data_2015[x].mean()

    a2 = data_2016[x].mean()

    a3 = data_2017[x].mean()

    d = {x: [a1,a2,a3]}

    df = pd.DataFrame(data=d,index=[2015,2016,2017])

    layout = dict(title='Mean value '+x+' in the world', geo=dict(showframe=False))

    df.iplot(kind='bar',layout=layout)
for x in cols:

    layout = dict(title=x+' in 2015',

             geo=dict(showframe=False))

    data_2015.sort_values(by=[x],ascending=False)[x].iplot(kind='bar',layout=layout)
for x in cols:

    layout = dict(title=x+' in 2016',

             geo=dict(showframe=False))

    data_2016.sort_values(by=[x],ascending=False)[x].iplot(kind='bar',layout=layout)
for x in cols:

    layout = dict(title=x+' in 2017',

             geo=dict(showframe=False))

    data_2017.sort_values(by=[x],ascending=False)[x].iplot(kind='bar',layout=layout)
for x in cols:

    data = dict(type='choropleth',

               locations=data_2015.index.unique(),

               locationmode='country names',

               z=data_2015[x],

               colorscale='Jet',

               colorbar={'title':x})



    layout = dict(title=x+' in 2015',

                 geo=dict(showframe=False,projection={'type':'mercator'}))



    choromap = go.Figure(data=[data],layout=layout)

    iplot(choromap,validate=False)
for x in cols:

    data = dict(type='choropleth',

               locations=data_2015.index.unique(),

               locationmode='country names',

               z=data_2016[x],

               colorscale='Jet',

               colorbar={'title':x})



    layout = dict(title=x+' in 2016',

                 geo=dict(showframe=False,projection={'type':'mercator'}))



    choromap = go.Figure(data=[data],layout=layout)

    iplot(choromap,validate=False)
for x in cols:

    data = dict(type='choropleth',

               locations=data_2017.index.unique(),

               locationmode='country names',

               z=data_2015[x],

               colorscale='Jet',

               colorbar={'title':x})



    layout = dict(title=x+' in 2017',

                 geo=dict(showframe=False,projection={'type':'mercator'}))



    choromap = go.Figure(data=[data],layout=layout)

    iplot(choromap,validate=False)
# Lets try some ML on 2015

# Note: of course ML for Happiness score will be perfect because this value is 

# calculated from other parameters. However I want to see, how other parameters impacts to Economy.
data_2015_2 = data_2015.drop(['Happiness Rank'],axis=1)

df_scaled = pd.DataFrame(preprocessing.scale(data_2015_2), columns=data_2015_2.columns)

df_scaled.head()
plt.figure(figsize=(50,50))

sns.pairplot(df_scaled,y_vars='Economy (GDP per Capita)',x_vars=df_scaled.columns[:-1])
plt.figure(figsize=(20,20))

sns.heatmap(df_scaled.corr(),annot=True,fmt='.1f',linewidths=2)
X = df_scaled.drop(['Happiness Score','Economy (GDP per Capita)'],axis=1)

y = data_2015['Economy (GDP per Capita)']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
linmodel = LinearRegression()

linmodel.fit(X_train,y_train)

linpred = linmodel.predict(X_test)
plt.scatter(y_test,linpred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, linpred))

print('MSE:', metrics.mean_squared_error(y_test, linpred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linpred)))
# Lets try to beat it!
from sklearn.linear_model import Lasso

alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1,1.5, 2,3,4, 5, 10, 20, 30, 40]

temp_mae = {}

temp_mse = {}

temp_rmse = {}

for i in alpha_ridge:

    lasso_reg = Lasso(alpha=i, normalize=True) 

    lasso_reg.fit(X_train, y_train)

    lasso_pred = lasso_reg.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, lasso_pred)

    mse = metrics.mean_squared_error(y_test, lasso_pred)

    rmse = np.sqrt(metrics.mean_squared_error(y_test, lasso_pred))

    temp_mae[i] = mae

    temp_mse[i] = mse

    temp_rmse[i] = rmse
temp_mae
temp_mse
temp_rmse
# Well lasso didn`t work

# Random forest? SVR?
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=1000)

forest_fit = forest_reg.fit(X_train,y_train)

forest_pred = forest_fit.predict(X_test)
from sklearn.svm import SVR



svr_reg = SVR(gamma='auto')

svr_fit = svr_reg.fit(X_train,y_train)

svr_pred = svr_fit.predict(X_test)
plt.scatter(y_test,svr_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
plt.scatter(y_test,forest_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
plt.scatter(y_test,linpred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
print('Linear Regression metrics')

print('MAE:', metrics.mean_absolute_error(y_test, linpred))

print('MSE:', metrics.mean_squared_error(y_test, linpred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linpred)))

print('')

print('Random Forest Regression metrics')

print('MAE:', metrics.mean_absolute_error(y_test, forest_pred))

print('MSE:', metrics.mean_squared_error(y_test, forest_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, forest_pred)))

print('')

print('SVR metrics')

print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))

print('MSE:', metrics.mean_squared_error(y_test, svr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
# And the winner is linear regression!