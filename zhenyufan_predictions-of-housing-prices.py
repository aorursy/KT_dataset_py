import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.plotly import plot_mpl

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from pylab import rcParams

import cufflinks as cf

init_notebook_mode(connected = True)

cf.go_offline()

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB



from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm

import itertools

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/Zillow Single Family Residence.csv")
df.head()
# Remove null values

df = df.dropna()
df.rename(columns={"RegionName":"ZipCode"}, inplace=True)

df["ZipCode"]=df["ZipCode"].map(lambda x: "{:.0f}".format(x))

df["RegionID"]=df["RegionID"].map(lambda x: "{:.0f}".format(x))

df.head()
df_usa = df.loc[:,'1996-04':'2018-12']

df_usa = df_usa.transpose()

df_usa.head()
usa_price = df_usa.mean(axis=1)

usa_price = pd.DataFrame(usa_price)

usa_price = usa_price.reset_index()

usa_price = usa_price.rename(columns={'index':'Time', 0:'Average Price'})

usa_price['Time'] = pd.to_datetime(usa_price['Time'])

usa_price.set_index('Time', inplace=True)

usa_price.head()
usa_price.iplot(title="The USA Single Family Home Prices 1996-2018",

                    xTitle="Year",

                    yTitle="Sales Price",

                    shape=(12,1)

                    )
df_state = df.copy()
df_state['row_mean'] = df_usa.mean(axis=0) 
df_state.head()
df_state_price = df_state[['State', 'row_mean']]

df_state_avg_price = df_state_price.groupby(['State'], as_index=False).mean()

df_state_avg_price = df_state_avg_price.rename(columns={'State': 'Code', 'row_mean': 'State Housing Price'})

df_state_avg_price['State Housing Price'] = df_state_avg_price['State Housing Price'].round(2)

df_state_avg_price.head()
us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY',

}
us_state_code = pd.DataFrame.from_dict(us_state_abbrev, orient='index')

us_state_code = us_state_code.reset_index()

us_state_code = us_state_code.rename(columns={'index': 'State', 0: 'Code'})

df_state_avg_price = df_state_avg_price.merge(us_state_code, on='Code', how='inner')

df_state_avg_price['Description'] = df_state_avg_price['State'].map(str) + '-' + df_state_avg_price['State Housing Price'].map(str)

df_state_avg_price.head()
data = dict(type = 'choropleth', 

            colorscale = 'Picnic', 

            locations = df_state_avg_price['Code'], 

            z = df_state_avg_price['State Housing Price'], 

            locationmode = 'USA-states', 

            text = df_state_avg_price['Description'], 

            marker = dict(line = dict(color = 'rgb(255, 255,255)', width = 2)),

            colorbar = {'title':"Housing Price"}

           )



layout = dict(title = 'The USA Housing Price',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )



choromap = go.Figure(data = [data], layout=layout)



iplot(choromap)
df_2018 = df.loc[:, '2018-01':'2018-12']

df_state = pd.DataFrame(df['State'])

df_recent = pd.concat([df_state, df_2018], axis=1)

df_recent['row_mean'] = df_recent.mean(axis=1) 

df_state_recent_price = df_recent[['State', 'row_mean']]

df_state_recent_avg_price = df_state_recent_price.groupby(['State'], as_index=False).mean()

df_state_recent_avg_price = df_state_recent_avg_price.rename(columns={'State': 'Code', 'row_mean': '2018 State Housing Price'})

df_state_recent_avg_price['2018 State Housing Price'] = df_state_recent_avg_price['2018 State Housing Price'].round(2)

df_state_recent_avg_price.head()
df_state_recent_avg_price = df_state_recent_avg_price.merge(us_state_code, on='Code', how='inner')

df_state_recent_avg_price['Description'] = df_state_recent_avg_price['State'].map(str) + '-' + df_state_recent_avg_price['2018 State Housing Price'].map(str)

df_state_recent_avg_price.head()
data = dict(type = 'choropleth', 

            colorscale = 'Picnic', 

            locations = df_state_recent_avg_price['Code'], 

            z = df_state_recent_avg_price['2018 State Housing Price'], 

            locationmode = 'USA-states', 

            text = df_state_recent_avg_price['Description'], 

            marker = dict(line = dict(color = 'rgb(255, 255,255)', width = 2)),

            colorbar = {'title':"2018 Housing Price"}

           )



layout = dict(title = 'The USA 2018 Housing Price',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )



choromap = go.Figure(data = [data], layout=layout)



iplot(choromap)
df_price = df.loc[:, '1996-04':'2018-12']

df_state = pd.DataFrame(df['State'])

df_state_price = pd.concat([df_state, df_price], axis=1)

df_state_price = df_state_price.rename(columns={'State':'Code'})

df_state_price.head()
df_state_price = df_state_price.merge(us_state_code, on='Code', how='inner')

df_state_price.head()
df_state_price_trend = df_state_price.groupby(['State']).mean()

df_state_price_trend = df_state_price_trend.transpose()

df_state_price_trend.head()
df_state_price_trend.iplot(title="The States' Single Family Home Prices 1996-2018",

                    xTitle="Year",

                    yTitle="Sales Price",

                    shape=(12,1)

                    )
#def main():

    #print('Please input the state name you want to show')

    #state_name = input()

state_name = 'California'

one_state_price_trend = pd.DataFrame(df_state_price_trend[state_name])

state_usa_price_trend = pd.concat([one_state_price_trend, usa_price], axis=1) # usa_price is the above one

state_usa_price_trend = state_usa_price_trend.rename(columns={'Average Price': 'USA Average Price'})



state_usa_price_trend.iplot(title=f"The {state_name} Single Family Home Prices 1996-2018",

                    xTitle="Year",

                    yTitle="Sales Price",

                    shape=(12,1)

                    )

#if __name__ == "__main__":

    #main()
#def main():

#print('Please input the city name you want to show')

    #city_name = input()

city_name = 'San Francisco'

df_price = df.loc[:, '1996-04':'2018-12']

df_city = pd.DataFrame(df['City'])

df_city_price = pd.concat([df_city, df_price], axis=1)

df_city_avg_price = df_city_price.groupby(['City']).mean()

df_city_avg_price = df_city_avg_price.transpose()

one_city_price_trend = pd.DataFrame(df_city_avg_price[city_name])

city_usa_price_trend = pd.concat([one_city_price_trend, usa_price], axis=1) # usa_price is the above one

city_usa_price_trend = city_usa_price_trend.rename(columns={'Average Price': 'USA Average Price'})

    

city_usa_price_trend.iplot(title=f"The {city_name} Single Family Home Prices 1996-2018",

                    xTitle="Year",

                    yTitle="Sales Price",

                    shape=(12,1)

                    )

#if __name__ == "__main__":

    #main()
X = df.loc[:,'1996-04':'2018-11']

y = df['2018-12']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
gau = GaussianNB()

gau.fit(X_train, y_train)

gau.score(X_test, y_test)
usa_price_trend = usa_price.copy()

usa_price_trend = usa_price_trend.round(2)

usa_price_trend.head()
rolmean = usa_price_trend.rolling(12).mean()

rolstd = usa_price_trend.rolling(12).std()



#Plot rolling statistics:

fig = plt.figure(figsize=(12, 6))

orig = plt.plot(usa_price_trend, color='blue',label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
usa_decomposition = sm.tsa.seasonal_decompose(usa_price_trend, model='additive')

rcParams['figure.figsize'] = 18, 8

fig = usa_decomposition.plot()
p = range(0, 2)

d = range(0, 2)

q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



warnings.filterwarnings("ignore") 



grid_results= []

for param in pdq:

    for seasonal_param in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(usa_price_trend,

                                            order=param,

                                            seasonal_order=seasonal_param,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            model = mod.fit()

            grid_results.append([param, seasonal_param, model.aic])

        except:

            print('error')

            continue



grid_results
model = sm.tsa.statespace.SARIMAX(usa_price_trend,

                                 order=(1,1,1), 

                                 seasonal_order=(1,1,0,12),   

                                 enforce_stationarity=False,

                                 enforce_invertibility=False)

result = model.fit()

print(result.summary())
result.plot_diagnostics(figsize=(12, 12))

plt.show()
pred_dynamic = result.get_prediction(start=pd.to_datetime('2014-01-01'), dynamic=True, full_results=True)

pred_dynamic_conf_int = pred_dynamic.conf_int()



axes = usa_price_trend['1996-04-01':].plot(label='Observed', figsize=(18, 10))

pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=axes)

axes.fill_between(pred_dynamic_conf_int.index,

                pred_dynamic_conf_int.iloc[:, 0],

                pred_dynamic_conf_int.iloc[:, 1], 

                color='r', 

                alpha=.3)

axes.fill_betweenx(axes.get_ylim(), 

                  pd.to_datetime('2014-01-01'), 

                  usa_price_trend.index[-1],

                  alpha=.1, zorder=-1)



axes.set_xlabel('Time (years)')

axes.set_ylabel('Average Housing Price')

plt.legend()
usa_forecast = result.get_forecast(steps= 60)

usa_forecast_conf_int = usa_forecast.conf_int()



axes = usa_price_trend.plot(label='Observed', figsize=(18, 10))

usa_forecast.predicted_mean.plot(ax=axes, label='Forecast')

axes.fill_between(usa_forecast_conf_int.index,

                 usa_forecast_conf_int.iloc[:, 0],

                 usa_forecast_conf_int.iloc[:, 1], color='b', alpha=.4)

axes.set_xlabel('Time')

axes.set_ylabel('USA Housing Price')

plt.legend()
df_state_price_trend.head()
df_state_price_trend = df_state_price_trend.round(2)

df_state_price_trend.head()
ca_price_trend = pd.DataFrame(df_state_price_trend['California'])

ca_price_trend = ca_price_trend.reset_index()

ca_price_trend['index'] = pd.to_datetime(ca_price_trend['index'])

ca_price_trend = ca_price_trend.set_index('index')

ca_price_trend.head()
rolmean = ca_price_trend.rolling(12).mean()

rolstd = ca_price_trend.rolling(12).std()



#Plot rolling statistics:

fig = plt.figure(figsize=(12, 6))

orig = plt.plot(ca_price_trend, color='blue',label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
ca_decomposition = sm.tsa.seasonal_decompose(ca_price_trend, model='additive')

rcParams['figure.figsize'] = 18, 8

fig = ca_decomposition.plot()
p = range(0, 2)

d = range(0, 2)

q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



warnings.filterwarnings("ignore") 



grid_results= []

for param in pdq:

    for seasonal_param in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(ca_price_trend,

                                            order=param,

                                            seasonal_order=seasonal_param,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            model = mod.fit()

            grid_results.append([param, seasonal_param, model.aic])

        except:

            print('error')

            continue



grid_results
model = sm.tsa.statespace.SARIMAX(ca_price_trend,

                                 order=(1,1,1), 

                                 seasonal_order=(0,1,1,12),   

                                 enforce_stationarity=False,

                                 enforce_invertibility=False)

result = model.fit()

print(result.summary())
result.plot_diagnostics(figsize=(12, 12))

plt.show()
pred_dynamic = result.get_prediction(start=pd.to_datetime('2014-01-01'), dynamic=True, full_results=True)

pred_dynamic_conf_int = pred_dynamic.conf_int()



axes = ca_price_trend['1996-04-01':].plot(label='Observed', figsize=(18, 10))

pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=axes)

axes.fill_between(pred_dynamic_conf_int.index,

                pred_dynamic_conf_int.iloc[:, 0],

                pred_dynamic_conf_int.iloc[:, 1], 

                color='r', 

                alpha=.3)

axes.fill_betweenx(axes.get_ylim(), 

                  pd.to_datetime('2014-01-01'), 

                  ca_price_trend.index[-1],

                  alpha=.1, zorder=-1)



axes.set_xlabel('Time (years)')

axes.set_ylabel('California Average Housing Price')

plt.legend()
ca_forecast = result.get_forecast(steps= 60)

ca_forecast_conf_int = ca_forecast.conf_int()



axes = ca_price_trend.plot(label='Observed', figsize=(18, 10))

ca_forecast.predicted_mean.plot(ax=axes, label='Forecast')

axes.fill_between(ca_forecast_conf_int.index,

                 ca_forecast_conf_int.iloc[:, 0],

                 ca_forecast_conf_int.iloc[:, 1], color='b', alpha=.4)

axes.set_xlabel('Time')

axes.set_ylabel('California Housing Price')

plt.legend()
#def main():

    #print('Please input the state name you want to show')

    #state_name = input()

state_name = 'California'

one_state_price_trend = pd.DataFrame(df_state_price_trend[state_name])

one_state_price_trend = one_state_price_trend.reset_index()

one_state_price_trend['index'] = pd.to_datetime(one_state_price_trend['index'])

one_state_price_trend = one_state_price_trend.set_index('index')

# Build up ARIMA model

model = sm.tsa.statespace.SARIMAX(one_state_price_trend,

                                 order=(1,1,1), 

                                 seasonal_order=(1,1,1,12),   

                                 enforce_stationarity=False,

                                 enforce_invertibility=False)

result = model.fit()

one_state_forecast = result.get_forecast(steps= 60)

one_state_forecast_conf_int = one_state_forecast.conf_int()



axes = one_state_price_trend.plot(label='Observed', figsize=(18, 10))

one_state_forecast.predicted_mean.plot(ax=axes, label='Forecast')

axes.fill_between(one_state_forecast_conf_int.index,

                 one_state_forecast_conf_int.iloc[:, 0],

                 one_state_forecast_conf_int.iloc[:, 1], color='b', alpha=.4)



axes.set_xlabel('Time (years)')

axes.set_ylabel(f'{state_name} Average Housing Price')

plt.legend()



#if __name__ == "__main__":

    #main()
#def main():

    #print('Please input the city name you want to show')

    #city_name = input()

city_name = 'New York'

df_price = df.loc[:, '1996-04':'2018-12']

df_city = pd.DataFrame(df['City'])

df_city_price = pd.concat([df_city, df_price], axis=1)

df_city_avg_price = df_city_price.groupby(['City']).mean()

df_city_avg_price = df_city_avg_price.transpose()

one_city_price_trend = pd.DataFrame(df_city_avg_price[city_name])

one_city_price_trend = one_city_price_trend.reset_index()

one_city_price_trend['index'] = pd.to_datetime(one_city_price_trend['index'])

one_city_price_trend = one_city_price_trend.set_index('index')

    # Build up ARIMA model

model = sm.tsa.statespace.SARIMAX(one_city_price_trend,

                                 order=(1,1,1), 

                                 seasonal_order=(1,1,1,12),   

                                 enforce_stationarity=False,

                                 enforce_invertibility=False)

result = model.fit()

    # Make predictions and get confidence interval

one_city_forecast = result.get_forecast(steps= 60)

one_city_forecast_conf_int = one_city_forecast.conf_int()



axes = one_city_price_trend.plot(label='Observed', figsize=(18, 10))

one_city_forecast.predicted_mean.plot(ax=axes, label='Forecast')

axes.fill_between(one_city_forecast_conf_int.index,

                 one_city_forecast_conf_int.iloc[:, 0],

                 one_city_forecast_conf_int.iloc[:, 1], color='b', alpha=.4)

axes.set_xlabel('Time')

axes.set_ylabel('USA Housing Price')

plt.legend()



axes.set_xlabel('Time (years)')

axes.set_ylabel(f'{city_name} Average Housing Price')

plt.legend()



#if __name__ == "__main__":

    #main()