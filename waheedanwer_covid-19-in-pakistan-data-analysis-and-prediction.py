import plotly_express as px
#gapminder = px.data.gapminder()
#gapminder2007 = gapminder.query("year == 2007")

#px.scatter(gapminder2007, x="gdpPercap", y="lifeExp")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#import plotly_express as px
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline
#import plotly_express as px
df_pk = pd.read_csv('../input/covid19-pakistan-official-data/Corona Data Pakistan.csv',index_col='SNo')
df_pk.shape
df_pk.head()
#Total cases of carona in Pakistan
df_pk['Total Cases'] = df_pk['Cured'] + df_pk['Deaths'] + df_pk['Confirmed']
#Active cases of carona in Pakistan
df_pk['Active Cases'] = df_pk['Total Cases'] - df_pk['Cured'] - df_pk['Deaths']
df_pk
#Till 3rd April Cases in Pakistan
df1= df_pk[df_pk['Date']=='3/4/2020']
#print(df1)
fig = px.bar(df1, x='State/UnionTerritory', y='Total Cases', color='Total Cases', height=600)
fig.update_layout(title='Till 3rd April Total Cases in Pakistan')
fig.show()
#Till  April Active Cases in Pakistan
df1= df_pk[df_pk['Date']=='3/4/2020']
fig = px.bar(df1, x='State/UnionTerritory', y='Active Cases', color='Active Cases',barmode='group', height=600)
fig.update_layout( title='Till 3rd April Active Cases in Pakistan')
fig.show()

df_pk['Date'] =pd.to_datetime(df_pk.Date,dayfirst=True)
df_pk
#Daily Cases in Pakistan Date wise
carona_data = df_pk.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = True)
carona_data['Daily Cases'] = carona_data['Total Cases'].sub(carona_data['Total Cases'].shift())
carona_data['Daily Cases'].iloc[0] = carona_data['Total Cases'].iloc[0]
carona_data['Daily Cases'] = carona_data['Daily Cases'].astype(int)
fig = px.bar(carona_data, y='Daily Cases', x='Date',hover_data =['Daily Cases'], color='Daily Cases', height=500)
fig.update_layout(
    title='Daily Cases in Pakistan Date wise')
fig.show()

#Total Cases in Pakistan Provinace/State Datewise
carona_data = df_pk.groupby(['Date','State/UnionTerritory','Total Cases'])['Cured','Deaths','Active Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)
fig = px.bar(carona_data, y='Total Cases', x='Date',hover_data =['State/UnionTerritory','Active Cases','Deaths','Cured'], color='Total Cases',barmode='group', height=700)
fig.update_layout(
    title='Pakistan States with Current Total Corona Cases')
fig.show()
#Total Cases,Active Cases,Cured,Deaths from Corona Virus in Pakistan
carona_data = df_pk.groupby(['Date'])['Total Cases','Active Cases','Cured','Deaths'].sum().reset_index().sort_values('Date',ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Total Cases'],
                    mode='lines+markers',name='Total Cases'))
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Active Cases'], 
                mode='lines+markers',name='Active Cases'))
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Cured'], 
                mode='lines+markers',name='Cured'))
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Deaths'], 
                mode='lines+markers',name='Deaths'))
fig.update_layout(title_text='Curve Showing Different Cases from COVID-19 in Pakistan',plot_bgcolor='rgb(225,230,255)')
fig.show()
#Testing till 2 April
df_pk['Date'] =pd.to_datetime(df_pk['Date'],dayfirst=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_pk['Date'], y=df_pk['Cumulative Tests'],
                    mode='lines+markers',name='Cumulative Tests'))
fig.add_trace(go.Scatter(x=df_pk['Date'], y=df_pk['Still Admitted'], 
                mode='lines+markers',name='Still Admitted'))
fig.add_trace(go.Scatter(x=df_pk['Date'], y=df_pk['Confirmed'], 
                mode='lines+markers',name='Confirmed'))
fig.update_layout(title_text='TEST for COVID-19',plot_bgcolor='rgb(225,230,255)')
fig.show()
#Total Cases,Active Cases,Cured,Deaths from Corona Virus in Pakistan
carona_data = df_pk.groupby(['Date'])['Cumulative Tests','Still Admitted','Confirmed','Home Facility'].sum().reset_index().sort_values('Date',ascending=False)
fig = go.Figure()
#fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Cumulative Tests'], 
                #mode='lines+markers',name='Cumulative Tests'))
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Confirmed'], 
                mode='lines+markers',name='Confirmed'))
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Still Admitted'], 
                mode='lines+markers',name='Still Admitted'))
fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Home Facility'], 
                mode='lines+markers',name='Home Facility'))
fig.update_layout(title_text='Curve Showing Test Performed and Status of Different Cases from COVID-19 in Pakistan',plot_bgcolor='rgb(225,230,255)')
fig.show()

#Last update
last_date = df_pk['Date'].iloc[-1]
last_df = df_pk[df_pk['Date'] == last_date].groupby('State/UnionTerritory').sum()[['Confirmed', 'Deaths']]
last_df = last_df.sort_values(by='Confirmed', ascending=False)
print('Pakistan Results by Region')
#We can find different camp options here: https://matplotlib.org/3.2.0/tutorials/colors/colormaps.html
last_df.style.background_gradient(cmap='Greens')
#Cumulative Partition
c = last_df
conf_max = c['Confirmed'][:4] 
conf_max.loc['Other'] = c['Confirmed'][4:].sum()
plt.figure(figsize=(11,6))
plt.pie(conf_max, labels=conf_max.index, autopct='%1.1f%%', explode=(0,0,0,0,1), shadow=True)
plt.title('COVID-19 Cumulative Patients Partition')
plt.show()
# Now Data Analysis and Prediction
## Import data
confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recoveries_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
## Showing the data

confirmed_df.head()
confirmed_df.columns
plt.figure(figsize=(15, 10))
sns.barplot(x=confirmed_df['4/4/20'], y=confirmed_df.index)
# Add label for horizontal axis
plt.xlabel("")
# Add label for vertical axis
plt.title("Confirmed")
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 
 
pk_cases = [] 

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)
    
    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    # case studies 
    pk_cases.append(confirmed_df[confirmed_df['Country/Region']=='Pakistan'][i].sum())
def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

world_daily_increase = daily_increase(world_cases)

pk_daily_increase = daily_increase(pk_cases)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
## print(("Pakistan Daily Increase"))
print(pk_daily_increase)
print(("**")*30)
print(("Pakistan Total Clases"))
print(pk_cases)


days_in_future = 30
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
#print(future_forcast)
adjusted_dates = future_forcast[:-30]
#print(adjusted_dates)
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, pk_cases, test_size=0.2, shuffle=False)
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=6, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(y_test_confirmed)
plt.plot(svm_test_pred)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)
# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)
plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression predictions'])
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)
bayesian_search.best_params_
bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Regression Predictions'])
adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, pk_cases)
plt.title('Num of Coronavirus Cases Over Time (Total)', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('Num of Cases', size=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, np.log10(pk_cases))
plt.title('Log of num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.bar(adjusted_dates, pk_daily_increase)
plt.title('Pakistan Daily Increases in Confirmed Cases', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, pk_cases)
plt.plot(future_forcast, svm_pred, linestyle='dashed', color='green')
plt.title('num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.legend(['Confirmed Cases', 'SVM predictions'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
# Future predictions using SVM 
print('SVM future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(svm_pred[-30:])))
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, pk_cases)
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='purple')
plt.title('num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, pk_cases)
plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='red')
plt.title('num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Time', size=12)
plt.ylabel('num of Cases', size=12)
plt.legend(['Confirmed Cases', 'Polynomial Bayesian Ridge Regression Predictions'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
# Future predictions using SVM 
print('SVM future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(svm_pred[-30:])))
# Future predictions using Polynomial Regression 
linear_pred = linear_pred.reshape(1,-1)[0]
print('Polynomial regression future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(linear_pred[-30:])))
# Future predictions using Linear Regression 
print('Ridge regression future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(bayesian_pred[-30:])))