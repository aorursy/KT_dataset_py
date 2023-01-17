# https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c

# ARIMA (Stationary --> then ACF PACF plots to find q and p) vs Exponential smoothening

# ARIMA

# Takes into account trends, seasonality, errors

# Autoregressive models predict future based on past values

# Implicit assumption: Future will resemble the past. Therefore, not good for periods of change





# Why prophet?

# Univariate data. Prophet uses time as a regressor

# stochastic demand

# targeted at business time series

# intuitive parameters easy to tune

# GAM: Additive regression model: uses a decomposable time series model

# with three main model components: trend, seasonality, and holidays. + error term

# non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects

# trend is modelled by fitting a peicewise linear curve. Hence non linear trends can be fit

# Robust to missing data and shifts in the trend, and typically handles outliers .

# based on Bayesian statistics
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import chart_studio.plotly as ply



# prophet by Facebook

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
recipe = pd.read_csv('../input/quickservicerestaurantinventorydata/recipe.csv')

costs = pd.read_csv('../input/quickservicerestaurantinventorydata/key_cost_items.csv')

sales = pd.read_csv('../input/quickservicerestaurantinventorydata/historical_sales.csv', thousands=',')

suppliers = pd.read_csv('../input/quickservicerestaurantinventorydata/supplier_details.csv')

orders = pd.read_csv('../input/quickservicerestaurantinventorydata/historical_order_summary.csv')
sales['Week Starting'] = pd.to_datetime(sales['Week Starting'], infer_datetime_format=True, dayfirst=True)

sales.columns = ['week_starting', 'units_sold']

sales.sort_values(by='week_starting', inplace=True)

sales.head()
# https://plot.ly/python/time-series/

fig = px.line(sales, x='week_starting', y='units_sold')

fig.show()



################# Below code if for timeseries with a rangeslider #########################

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=sales.week_starting, y=sales['units_sold'], name="units_sold",

#                          line_color='deepskyblue'))



# fig.update_layout(title_text='Sales over time',

#                   xaxis_rangeslider_visible=True)

# fig.show()
# Handling outliers

# https://facebook.github.io/prophet/docs/outliers.html

sales['units_sold'][sales['units_sold']<60000] = 65000
fig = px.line(sales, x='week_starting', y='units_sold')

fig.show()
dt_index = pd.DatetimeIndex(sales.week_starting, freq='infer')

sales_date_index = sales.set_index(dt_index)

sales_date_index.drop(columns=['week_starting'], inplace=True)

sales_date_index.sort_index(inplace=True)

sales_date_index.head()
# train test split

train_size = int(len(sales_date_index) * 0.9)

train, test = sales_date_index[0:train_size], sales_date_index[train_size:len(sales_date_index)]
print(len(sales_date_index))

print(len(train))

print(len(test))
# Carrying capacity, forecasting growth and saturating minimum

# We are not using these parameters in our case
# Prophet requires the variable names in the time series to be:

# y – Target

# ds – Datetime

train['ds'] = train.index

train['y'] = train.units_sold

train.drop(['units_sold'],axis = 1, inplace = True)
# confidence interval

model1=Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=False) # by default is 80%
model1.fit(train)
future = model1.make_future_dataframe(periods=8, freq='W')

future
forecast = model1.predict(future)

forecast[['yhat', 'yhat_lower', 'yhat_upper']]
components = model1.plot_components(forecast)

components.show()
# Calculate MAPE (mean absolute percentage error on predicted values)

test['units_sold']
# Calculate MAPE (Mean absolute percentage error)

forecast.set_index('ds', inplace=True) # Uncomment this while running

temp = (test['units_sold'] - forecast.loc['2013-04-07':,'yhat'])

print("MAPE: ",(temp.abs()/test['units_sold']).mean() * 100, "%")
forecast.loc['2013-04-07':,'yhat']
fig = go.Figure()

fig.add_trace(go.Scatter(x=sales_date_index.index, y=sales_date_index['units_sold'], name="units_sold",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat'], name="predicted",

                         line_color='indianred'))



fig.add_trace(go.Scatter(name='Upper Bound',x=sales_date_index.index, y=forecast['yhat_upper'],

                         mode='lines', marker=dict(color="#444"), line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.3)',

                         fill='tonexty'))



fig.add_trace(go.Scatter(name='Lower Bound',x=sales_date_index.index, y=forecast['yhat_lower'],

                         mode='lines', marker=dict(color="#444"), line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.3)',

                         fill='tonexty'))



fig.update_layout(title_text='Actual & Forecasted Time-Series with Rangeslider',

                  xaxis_rangeslider_visible=True)

fig.show()
# GRID SEARCH FOR BEST PARAMETERS

from sklearn.model_selection import ParameterGrid

params_grid = {'yearly_seasonality':[5,10],

               'changepoint_prior_scale':[0.01,0.05, 0.1]}

grid = ParameterGrid(params_grid)

print([p for p in grid])
for p in grid:

    m =Prophet(**p)

    m.fit(train)

    future_dates = m.make_future_dataframe(periods=8, freq='W')

    future_predict = m.predict(future_dates)

    future_predict.set_index('ds', inplace=True)

    # Calculate MAPE (Mean absolute percentage error)

    temp = (test['units_sold'] - future_predict.loc['2013-04-07':,'yhat'])

    print("MAPE: ",(temp.abs()/test['units_sold']).mean() * 100, "% ", p)
!pip install pyramid-arima

from pyramid.arima import auto_arima
stepwise_model = auto_arima(sales_date_index, start_p=1, start_q=1, max_p=5, max_q=5, 

                            m=52,start_P=0, seasonal=True,trace=True,error_action='ignore',

                            suppress_warnings=True, stepwise=True, n_jobs=-1)

print(stepwise_model.aic())
arima_forecast = stepwise_model.predict(n_periods=8)
arima_forecast = pd.DataFrame(arima_forecast,index = test.index,columns=['Prediction'])

arima_forecast.head()
# Calculate MAPE (Mean absolute percentage error)

temp = (test['units_sold'] - arima_forecast.loc['2013-04-07':,'Prediction'])

print("MAPE: ",(temp.abs()/test['units_sold']).mean() * 100, "%")
arima_forecast
fig = go.Figure()

fig.add_trace(go.Scatter(x=sales_date_index.index, y=sales_date_index['units_sold'], name="units_sold",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=arima_forecast.index, y=arima_forecast['Prediction'], name="predicted",

                         line_color='indianred'))



fig.update_layout(title_text='Actual & Forecasted ARIMA',

                  xaxis_rangeslider_visible=False)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=sales_date_index.index, y=sales_date_index['units_sold'], name="units_sold",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=forecast.loc['2013-04-07':,'yhat'].index, y=forecast.loc['2013-04-07':,'yhat'], name="predicted",

                         line_color='indianred'))



fig.update_layout(title_text='Actual & Forecasted Prophet',

                  xaxis_rangeslider_visible=False)

fig.show()
# ### Fit a distribution to data

# # !pip install fitter

# from fitter import Fitter

# f = Fitter(np.array(orders[orders['Item']=='Potatoes']['Lead Time']).astype(int))

# f.fit()

# # f.summary()

# f.get_best()
# # generate random numbers from distribution

# from scipy.stats import dweibull

# dweibull.rvs(*f.fitted_param['dweibull']) # ideally you'd want a discrete output
# daily demand with std dev for the next 2 months

temp = forecast[-8:][['yhat','yhat_lower','yhat_upper']]

temp['std_dev'] = (temp['yhat_upper'] - temp['yhat'])/2

temp.drop(columns=['yhat_lower','yhat_upper'], inplace=True)

temp['yhat'] = temp['yhat']/7

temp['std_dev'] = temp['std_dev']/7

temp = temp.loc[temp.index.repeat(7)]

temp = temp.reset_index(drop=True)

temp.head(10)
# generate demand from mean and std deviation

from scipy.stats import norm

def generate_demand(mean, std_dev):

    return norm.rvs(loc=mean, scale=std_dev, size=1, random_state=None)[0]
# simulating daily demand mean and std dev

temp['demand'] = temp.apply(lambda x: generate_demand(x['yhat'],x['std_dev']), axis=1)

daily_demand = temp

daily_demand.head(10)
beginning_inventory = 140000

# define cost/ penalty:

p2 = 1 # unmet demand: penalty = 1

p1 = 1 # expired product: penalty = 1



def simulation_one_run(order_qty = 40000, min_qty = 20000): # parameters fixed order qty and min inventory levels

    

    delivery_date = -1 # date when order is supposed to be delivered

    expired = []

    unmet_demand = []

    pending_order = False

    

    # create inventory dataframe with beginning inventory = 140000. Columns represent how old the inventory is

    inventory = pd.DataFrame(index= np.arange(56), columns=np.arange(7))

    inventory.fillna(0, inplace=True)

    inventory.iloc[0,0] = beginning_inventory

    

    for index, row in inventory.iterrows():

        demand = daily_demand.loc[index,'demand'] # get demand for the day



        if delivery_date == index: # receive order

            row[0] += order_qty # parameter 2

            pending_order = False



        for i in range(len(row)-1,-1,-1): # iterate from 6 to 0. oldest inventory gets consumed first

            if demand >= row[i]:

                demand -= row[i]

                row[i] = 0

            else:

                row[i] -= demand

                demand = 0



        unmet_demand.append(demand)



        if row.sum() < min_qty and not pending_order:

            delivery_date = index + 7 # replenish demand by ordering

            pending_order = True



        if index < len(inventory)-1: # update inventory levels for next day

            for i in range(len(row)-1):

                inventory.loc[index+1,i+1] += row[i]



        expired.append(row.iloc[-1])

        

    return (sum(expired)*p1 + sum(unmet_demand)*p2)
# initial_guess = np.array([1])

# result = optimize.minimize(cost, initial_guess, bounds=bounds, method='TNC')



import scipy.optimize as optimize



def cost(params):

    order_qty, min_qty= params # <-- params is a NumPy array

    return simulation_one_run(order_qty, min_qty)



bnds = ((40000,200000),(10000, 200000))



# result = optimize.differential_evolution(cost, bounds=bnds)

# if result.success:

#     fitted_params = result.x

#     print(fitted_params)

# else:

#     raise ValueError(result.message)
# simulate 10 times

result_arr = []

for i in range(10):

    

    # simulating daily demand table for the next 2 months

    temp['demand'] = temp.apply(lambda x: generate_demand(x['yhat'],x['std_dev']), axis=1)

    daily_demand = temp

    

    # optimizing parameters

    result = optimize.differential_evolution(cost, bounds=bnds)

    if result.success:

        fitted_params = result.x

        result_arr.append(fitted_params)

    else:

        raise ValueError(result.message)
beginning_inventory = 140000

# define cost/ penalty:

p2 = 1 # unmet demand: penalty = 1

p1 = 1 # expired product: penalty = 1



def simulation_one_run_test(order_qty = 40000, min_qty = 20000): # parameters fixed order qty and min inventory levels

    

    # simulating daily demand table

    temp['demand'] = temp.apply(lambda x: generate_demand(x['yhat'],x['std_dev']), axis=1)

    daily_demand = temp

    

    delivery_date = -1 # date when order is supposed to be delivered

    expired = []

    unmet_demand = []

    pending_order = False

    # create inventory dataframe with beginning inventory = 140000. Columns represent how old the inventory is

    inventory = pd.DataFrame(index= np.arange(56), columns=np.arange(7))

    inventory.fillna(0, inplace=True)

    inventory.iloc[0,0] = beginning_inventory

    

    for index, row in inventory.iterrows():

        demand = daily_demand.loc[index,'demand'] # get demand for the day



        if delivery_date == index: # receive order

            row[0] += order_qty # parameter 2

            pending_order = False



        for i in range(len(row)-1,-1,-1): # iterate from 6 to 0. oldest inventory gets consumed first

            if demand >= row[i]:

                demand -= row[i]

                row[i] = 0

            else:

                row[i] -= demand

                demand = 0



        unmet_demand.append(demand)



        if row.sum() < min_qty and not pending_order:

            delivery_date = index + 7 # replenish demand by ordering

            pending_order = True



        if index < len(inventory)-1: # update inventory levels for next day

            for i in range(len(row)-1):

                inventory.loc[index+1,i+1] += row[i]



        expired.append(row.iloc[-1])

        

    return (sum(expired), sum(unmet_demand))
for fitted_parameters in result_arr:

    a,b = simulation_one_run_test(fitted_parameters[0],fitted_parameters[1])

    d = a + b

    print(fitted_parameters[0],fitted_parameters[1])

    print(d,b,a)# total penalty, b = unmet demand, c = expired