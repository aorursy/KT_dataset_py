# lets import the libraries we need for EDA:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
plt.style.use("ggplot") # to make the plots to look nicer
import os
print(os.listdir("../input"))
#lets import the training and test files:
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
store_df = pd.read_csv("../input/store.csv")
#how many datas are in the files:
print("in the training set we have", train_df.shape[0], "observations and", train_df.shape[1], "columns/variables.")
print("in the testing set we have", test_df.shape[0], "observations and", test_df.shape[1], "columns/variables.")
print("in the store set we have", store_df.shape[0], "observations and", store_df.shape[1], "columns/variables.")
#how does the data looks like:
train_df.head().append(train_df.tail()) #show the first and last 5 rows.
#wow. no missing values.
train_df.isnull().all()
opened_sales = (train_df[(train_df.Open == 1) & (train_df.Sales)]) #if the stores are opend
opened_sales.Sales.describe()
f, ax = plt.subplots(1,2, figsize = (20, 5))

opened_sales.Sales.plot(kind = "hist", title = "Sales Histogram", bins = 20, ax = ax[0])
opened_sales.Sales.plot.box(title = "Sales Boxplot", ax = ax[1])
print("Rossmann has", round(opened_sales.Sales[(opened_sales.Sales > 10000)].count() / opened_sales.shape[0] * 100, 2), 
      "% of the time big sales, over 10.000 Euros")
print("Rossmann has", round(opened_sales.Sales[(opened_sales.Sales < 1000)].count() / opened_sales.shape[0] * 100, 4), 
      "% of the time low sales, under 1000 Euros")
train_df.Customers.describe()
f, ax = plt.subplots(1,2, figsize = (20, 5))

train_df.Customers.plot(kind = "hist", title = "Customers Histogram", bins = 20, ax = ax[0])
train_df.Customers.plot.box(title = "Sales Boxplot", ax = ax[1])
#Seems to had a great sortiment on 22th of January 2013. They hit the record of customers. 
train_df[(train_df.Customers > 6000)]
print("In 3 years, different stores where", train_df[(train_df.Open == 0)].count()[0], "times closed")
print("From this days,", train_df[(train_df.Open == 0) & 
         ((train_df.StateHoliday == "a") | 
         (train_df.StateHoliday == "b") | 
         (train_df.StateHoliday == "c"))].count()[0], "times the stores were closed because of holidays")
print(train_df[(train_df.Open == 0) & (train_df.SchoolHoliday == 1)].count()[0], "times, some stores were closed because of school holiday")
print("The stores were in some sundays opend ->", train_df[(train_df.Open == 1) & (train_df.DayOfWeek == 7)].count()[0], "times")
print("However,", train_df[(train_df.Open == 0) & ((train_df.StateHoliday == "0") | (train_df.StateHoliday == 0)) & (train_df.SchoolHoliday == 0)].count()[0], 
      "times, the stores were closed for no reason (No Holidays o Sunday)")
print("""Rossman described clearly, that they were undergoing refurbishments sometimes and had to close. 
      Most probably those were the times this event was happening. 
      However, we dont want to have those obsvervations in our dataset, when predicting. So lets delete those days after we finished 
      our analysis""")

print(round((train_df.Promo[train_df.Promo == 1].count() / train_df.shape[0] * 100), 2), "% of the time, has been promotions made")
# StateHoliday is not a continous number. 
train_df.StateHoliday.value_counts()
# StateHoliday is a string and I for me is not so important to know what kind of holiday (a, b or c). I will convert it into 0 and 1, by creating a new variable
train_df["StateHoliday_cat"] = train_df["StateHoliday"].map({0:0, "0": 0, "a": 1, "b": 1, "c": 1})
train_df.StateHoliday_cat.count()
# let get rid of the StateHoliday column and use only the new one
train_df = train_df.drop("StateHoliday", axis = 1)
train_df.tail()
#lets delete the times, where the stores were opened with no sales because of days in inventory.
train_df = train_df.drop(train_df[(train_df.Open == 0) & (train_df.Sales == 0)].index)
train_df = train_df.reset_index(drop = True) # to ge the indexes back to 0, 1, 2,etc.

train_df.isnull().all() #to check for NaNs
store_df.head().append(store_df.tail())
#how may missing data do we have in %:
100- (store_df.count() / store_df.shape[0] * 100)
store_df.info()
store_df.CompetitionDistance.plot.box() #let me see the outliers, so we can choose between mean and median to fill the NaNs
print("the median is", store_df.CompetitionDistance.median(), "and mean is", store_df.CompetitionDistance.mean())
print("Since we have here some outlier, its better to input the median value to those few missing values.")
store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median(), inplace = True)
#The missing values, are not there, because the stores had no competition. So I would suggest to fill the missing values with zeros.
store_df["CompetitionOpenSinceMonth"].fillna(0, inplace = True)
store_df["CompetitionOpenSinceYear"].fillna(0, inplace = True)
store_df.groupby(by = "Promo2", axis = 0).count() 
# so if no promo has been made, then we should replace the NaN from Promo since Week and Year with zero
store_df["Promo2SinceWeek"].fillna(0, inplace = True)
store_df["Promo2SinceYear"].fillna(0, inplace = True)
store_df["PromoInterval"].fillna(0, inplace = True)

store_df.info()
train_store_df = pd.merge(train_df, store_df, how = "left", on = "Store")
train_store_df.info()
train_store_df["Avg_Customer_Sales"] = train_store_df.Sales / train_store_df.Customers
f, ax = plt.subplots(2, 3, figsize = (20,10))

store_df.groupby("StoreType")["Store"].count().plot(kind = "bar", ax = ax[0, 0], title = "Total StoreTypes in the Dataset")
train_store_df.groupby("StoreType")["Sales"].sum().plot(kind = "bar", ax = ax[0,1], title = "Total Sales of the StoreTypes")
train_store_df.groupby("StoreType")["Customers"].sum().plot(kind = "bar", ax = ax[0,2], title = "Total nr Customers of the StoreTypes")
train_store_df.groupby("StoreType")["Sales"].mean().plot(kind = "bar", ax = ax[1,0], title = "Average Sales of StoreTypes")
train_store_df.groupby("StoreType")["Avg_Customer_Sales"].mean().plot(kind = "bar", ax = ax[1,1], title = "Average Spending per Customer")
train_store_df.groupby("StoreType")["Customers"].mean().plot(kind = "bar", ax = ax[1,2], title = "Average Customers per StoreType")

plt.subplots_adjust(hspace = 0.3)
plt.show()
sns.countplot(data = train_store_df, x = "StoreType", hue = "Assortment", order=["a","b","c","d"]) 
print("""So only the StoreType B has all assortments. I think thats why they are performing so good. Maybe this StoreType has more sales area.
      The assortment C is a good one, because the StoreType D has the best average customer spending.""")

plt.show()
train_store_df.Date = train_store_df.Date.astype("datetime64[ns]")

train_store_df["Month"] = train_store_df.Date.dt.month
train_store_df["Year"] = train_store_df.Date.dt.year
train_store_df["Day"] = train_store_df.Date.dt.day
sns.factorplot(data = train_store_df, x ="Month", y = "Sales", 
               col = 'Promo', # per store type in cols
               hue = 'Promo2',
               row = "Year"
             )
# So, of course, if the stores are having promotion the sells are higher.
# Overall the store promotions sellings are also higher than the seasionality promotions (Promo2). However I can't see no yearly trend. 
sns.factorplot(data = train_store_df, x = "DayOfWeek", y = "Sales", hue = "Promo")
print("""So, no promotion in the weekend. However, the sales are very high, if the stores have promotion. 
The Sales are going crazy on Sunday. No wonder.""")
print("There are", train_store_df[(train_store_df.Open == 1) & (train_store_df.DayOfWeek == 7)].Store.unique().shape[0], "stores opend on sundays")    

sns.factorplot(data = train_store_df, x = "Month", y = "Sales", col = "Year", hue = "StoreType")
# Yes, we can see a seasonalities, but not trends. The sales stays constantly yearly. 
train_store_df.CompetitionDistance.describe()
# The obsverations are continous numbers, so we need to convert them into a categories. Lets a create a new variable.
train_store_df["CompetitionDistance_Cat"] = pd.cut(train_store_df["CompetitionDistance"], 5)
f, ax = plt.subplots(1,2, figsize = (15,5))

train_store_df.groupby(by = "CompetitionDistance_Cat").Sales.mean().plot(kind = "bar", title = "Average Total Sales by Competition Distance", ax = ax[0])
train_store_df.groupby(by = "CompetitionDistance_Cat").Customers.mean().plot(kind = "bar", title = "Average Total Customers by Competition Distance", ax = ax[1])

# It is pretty clear. If the competions is very far away, the stores are performing better (sales and customers)
# first we have to convert the variables to categories, bevor we convert them to codes.

# train_store_df["Promo"] = train_store_df["Promo"].astype("category") # it's already numerica
# train_store_df["SchoolHoliday"] = train_store_df["SchoolHoliday"].astype("category") # it's already numerica
train_store_df["StoreType"] = train_store_df["StoreType"].astype("category")
train_store_df["Assortment"] = train_store_df["Assortment"].astype("category")
# train_store_df["Promo2"] = train_store_df["Promo2"].astype("category") # it's already numerica
train_store_df["PromoInterval"] = train_store_df["PromoInterval"].astype("category")

train_store_df["StoreType_cat"] = train_store_df["StoreType"].cat.codes
train_store_df["Assortment_cat"] = train_store_df["Assortment"].cat.codes
train_store_df["PromoInterval_cat"] = train_store_df["Assortment"].cat.codes

train_store_df["StateHoliday_cat"] = train_store_df["StateHoliday_cat"].astype("float")
train_store_df["StoreType_cat"] = train_store_df["StoreType_cat"].astype("float")
train_store_df["Assortment_cat"] = train_store_df["Assortment_cat"].astype("float")
train_store_df["PromoInterval_cat"] = train_store_df["PromoInterval_cat"].astype("float")
train_store_df.info()
df_correlation = train_store_df[["Store", "DayOfWeek", "Sales", "Customers", "Promo", "SchoolHoliday", "CompetitionDistance", 
                                 "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Avg_Customer_Sales", 
                                 "Month", "Year", "Day", "StateHoliday_cat", "Assortment_cat", "StoreType_cat", "PromoInterval_cat"]]


f, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(df_correlation.corr(),ax = ax, annot=True, cmap=sns.diverging_palette(10, 133, as_cmap=True), linewidths=0.5)
ts_arima = train_store_df.set_index("Date").resample("W").mean() #set the index to date and resample it by summing to monthly values
ts_arima = ts_arima[["Sales"]]
ts_arima.plot()
# Les see if we have stationary or non-stationary time series

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
#     #Perform Dickey-Fuller test:
#     print('Results of Dickey-Fuller Test:')
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
#     print (dfoutput)
    
test_stationarity(ts_arima)
import warnings
import itertools
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

# Let's begin by generating the various combination of parameters that we wish to assess:

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(ts_arima,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
# this is the optimal paramater for our model: ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:1847.5087433770632

mod = sm.tsa.statespace.SARIMAX(ts_arima,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()
# Lets go ahead with validating forecasts

pred = results.get_prediction(start = pd.to_datetime("2015-01-11"), dynamic = False) 
# lets start the fc to start from 1.11.2015. 
# The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.
pred_ci = pred.conf_int() # Get confidence intervals of forecasts

ax = ts_arima["2014":].plot(label = "observed", figsize=(15, 7))
pred.predicted_mean.plot(ax = ax, label = "One-step ahead FC", alpha = 1)
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], 
                color = "k", alpha = 0.05)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")

plt.legend
plt.show()

# -----------
#  extract the predicated and true values of our time series
ts_forecasted = pred.predicted_mean
ts_truth = ts_arima["2015-01-11":]
# to use, in my case, the mean squared error:
rms_arima = sqrt(mean_squared_error(ts_truth, ts_forecasted))
print("RMS:", rms_arima) 
# lets try a dynamic forecast. In this case we will use information from the time series up to a certain point, to generate future values.
pred_dynamic = results.get_prediction(start = pd.to_datetime("2015-01-11"), dynamic = True, full_results = True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = ts_arima["2014":].plot(label = "observed", figsize = (20, 7))
pred_dynamic.predicted_mean.plot(label ="Dynamic Forecast", ax = ax)

ax.fill_between(pred_dynamic_ci.index,
               pred_dynamic_ci.iloc[:, 0],
               pred_dynamic_ci.iloc[:, 1], color ="k", alpha = 0.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime("2015-01-11"), ts_arima.index[-1],
                alpha = 0.1, zorder =-1)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")

plt.legend()
plt.show()

# -----------
#  extract the predicated and true values of our time series
ts_forecasted = pred_dynamic.predicted_mean
ts_truth = ts_arima["2015-01-11":]
# to use, in my case, the mean squared error:
rms_arima_dynamic = sqrt(mean_squared_error(ts_truth, ts_forecasted))
print("RMS:", rms_arima_dynamic) 
pred_uc = results.get_forecast(steps = 60) # lets get a forecast for the next few periods
pred_ci = pred_uc.conf_int() # Get confidence intervals of forecasts

ax = ts_arima.plot(label = "observed", figsize = (20,7))
pred_uc.predicted_mean.plot(ax = ax, label = "Forecast")
ax.fill_between(pred_ci.index, 
               pred_ci.iloc[:, 0],
               pred_ci.iloc[0, 1], color = "k", alpha = 0.25)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

plt.legend()
plt.show()

from fbprophet import Prophet
# I want to create a new dataframe for this model.
ts_prophet = train_store_df.drop(['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo',
       'StoreType', 'Assortment',
       'CompetitionDistance', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
       'PromoInterval', 'Avg_Customer_Sales', 'Month', 'Year', 'Day',
       'CompetitionDistance_Cat', 'StoreType_cat', 'Assortment_cat',
       'PromoInterval_cat', "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"], axis = 1)

ts_prophet.head()
# as I understand from the documentation, the variables should have a specific names
ts_prophet = ts_prophet.rename(columns = {"Date": "ds",
                          "Sales": "y"})

ts_prophet.tail()

# In prophet we can also model the holidays. so lets go for it.

state_dates = ts_prophet[(ts_prophet.StateHoliday_cat == 1)].loc[:, "ds"].values
school_dates = ts_prophet[(ts_prophet.SchoolHoliday == 1)].loc[:, "ds"].values

state = pd.DataFrame({"holiday": "state_holiday", 
                     "ds": pd.to_datetime(state_dates)})
school = pd.DataFrame({"holiday": "school_holiday",
                      "ds": pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))
holidays.head()
ts_prophet = ts_prophet.drop(["SchoolHoliday", "StateHoliday_cat"], axis = 1) # we dont need them anymore.
# it takes just too long to fit the model with a daily time series, so lets make it weekly.
ts_week_prophet = ts_prophet.set_index("ds").resample("W").sum()
ts_week_prophet_train = ts_week_prophet["2013-01-01": "2015-01-11"] #I will slice the dataframe, so we can have some testing data 
ts_week_prophet_train = ts_week_prophet_train.reset_index()
ts_week_prophet = ts_week_prophet.reset_index() # here are all the weekly data

holidays_week = holidays.set_index("ds").resample("W").min()
holidays_week = holidays_week.dropna(axis = 0)
# holidays_week.holiday.fillna(0, inplace = True)
holidays_week = holidays_week.reset_index()
# help(Prophet)
# lets fit the model
prophet = Prophet(holidays = holidays) # holidays = holidays_week
# prophet = Prophet(interval_width = 0.80, holidays = holidays, weekly_seasonality=True, daily_seasonality=False) # the default uncertainty is 80 %
prophet.fit(ts_week_prophet_train)
print("done")

future = prophet.make_future_dataframe(periods = 52, freq = "W") # here we are extending our dataframe with the dates for which a prediction is to be made.
forecast = prophet.predict(future) # with predict method I asign each row in future dates a predicted value, which it names yhat

forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail() # We have a new dataframe, which includes, the forecast and the uncertainity invervals.

fig1 = prophet.plot(forecast) #plot the results for the forecast time.
# with this method we can see the components (trend, yearly seasonality and weekly seasonality of the time series.).
fig2 = prophet.plot_components(forecast)
fc_week_prophet = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]] # create a df with only the importan variables
fc_week_prophet = fc_week_prophet.merge(ts_week_prophet, how = "left", on = "ds") #add the original data to the fc frame, so we can compare 
fc_week_prophet = fc_week_prophet.set_index("ds") # make a time series index

plt.figsize=(10,20)
fc_week_prophet["y"].plot(figsize=(17,10))
fc_week_prophet["yhat"].plot()

plt.legend()
plt.show()
# y_prophet = fc_week_prophet["y"][: "2015-08-02"]
# yhat_prophet = fc_week_prophet["yhat"][: "2015-08-02"]

# rms_prophet = sqrt(mean_squared_error(fc_week_prophet["y"], fc_week_prophet["yhat"]))
# print("RMS:", rms_prophet) 
train_store_df["CompetitionOpenSince"] = np.where((train_store_df["CompetitionOpenSinceMonth"] == 0) & (train_store_df["CompetitionOpenSinceYear"] == 0), 
                                                0,(train_store_df.Month - train_store_df.CompetitionOpenSinceMonth) + (12 *(train_store_df.Year - train_store_df.CompetitionOpenSinceYear)))

# lets drop the variables
train_store_df = train_store_df.drop(["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"], axis = 1)
# lets drop few variables, that either or not numeric or we dont need them anymore
# lets create a new data frame for this model
ts_rfr = train_store_df.copy()
ts_rfr = train_store_df.drop(["Date","StoreType", "Assortment", "PromoInterval", "CompetitionDistance_Cat"], axis = 1) #dop this columns, as we already have them in categories
# ts_rfr = pd.get_dummies(ts_rfr, columns = ["Assortment_cat", "StoreType_cat", "PromoInterval_cat"], prefix = ["is_Assortment", "is_StoreType", "is_PromoInterval"]) # create dummies
from sklearn import model_selection
from sklearn import metrics

features = ts_rfr.drop(["Customers", "Sales", "Avg_Customer_Sales"], axis = 1)
target = ts_rfr["Sales"]

X_train, X_train_test, y_train, y_train_test = model_selection.train_test_split(features, target, test_size = 0.20, random_state = 15) 
# I call here train_test_set which is  divided 80% and 20% validation
print(X_train.shape, X_train_test.shape, y_train.shape, y_train_test.shape)
# Try different numbers of n_estimators - this will take a minute or so
# from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(n_jobs=-1)
# estimators = np.arange(10, 200, 10)
# scores = []
# for n in estimators:
#     model.set_params(n_estimators=n)
#     model.fit(X_train, y_train)
#     scores.append(model.score(X_train_test, y_train_test))
# plt.title("Effect of n_estimators")
# plt.xlabel("n_estimator")
# plt.ylabel("score")
# plt.plot(estimators, scores)

# #another script that takes toooo long, to find the right parameters for RFR
# params = {'max_depth':(4,6,8,10,12,14,16,20),
#          'n_estimators':(4,8,16,24,48,72,96,128),
#          'min_samples_split':(2,4,6,8,10)}
# #scoring_fnc = metrics.make_scorer(rmspe)
# #the dimensionality is high, the number of combinations we have to search is enormous, using RandomizedSearchCV 
# # is a better option then GridSearchCV
# grid = model_selection.RandomizedSearchCV(estimator=rfr,param_distributions=params,cv=10) 
# #choosing 10 K-Folds makes sure i went through all of the data and didn't miss any pattern.(takes time to run but is worth doing it)
# grid.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=10)
rfr.fit(X_train, y_train)
yhat = rfr.predict(X_train_test)
rms_rfr = sqrt(mean_squared_error(y_train_test, yhat))
print("RMS:", rms_rfr) 
importances = rfr.feature_importances_
std = np.std([rfr.feature_importances_ for tree in rfr.estimators_],
             axis=0)
indices = np.argsort(importances)
palette1 = itertools.cycle(sns.color_palette())
# Store the feature ranking
features_ranked=[]
for f in range(X_train.shape[1]):
    features_ranked.append(X_train.columns[indices[f]])
# Plot the feature importances of the forest

plt.figure(figsize=(10,10))
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances[indices],
            color=[next(palette1)], align="center")
plt.yticks(range(X_train.shape[1]), features_ranked)
plt.ylabel('Features')
plt.ylim([-1, X_train.shape[1]])
plt.show()
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)
import xgboost as xgb

param = {'max_depth':10, # maximum depth of a tree
         "booster": "gbtree",   # use tree based models 
         'eta':1, # learning rate
         'silent':1, # silent mode
         'objective':'reg:linear', # for linear regression
#          "seed": 10,   # Random number seed
#          "subsample": 0.9,    # Subsample ratio of the training instances
        }

num_round = 100 #how many boosting rounds

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_train_test, y_train_test)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

xgboost = xgb.train(param, dtrain, num_round, evals=watchlist, \
  early_stopping_rounds= 100, feval=rmspe_xg, verbose_eval=True)
         
# make prediction
preds = xgboost.predict(dtest)

# model = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
#   early_stopping_rounds= 100, feval=rmspe_xg, verbose_eval=True)
rms_xgboost = sqrt(mean_squared_error(y_train_test, preds))
print("RMS:", rms_xgboost) 
# Lets see the feature importance
fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xgboost, max_num_features=50, height=0.8, ax=ax)
plt.show()
print("now lets see the scores togher")

model_errors = pd.DataFrame({
    "Model": ["SARIMAX", "SARIMAX Dynamic", "Random Forest Regression", "XGBoost"],
    "Score": [rms_arima, rms_arima_dynamic, rms_rfr, rms_xgboost]
})

model_errors.sort_values(by = "Score", ascending = True)
# # As we are working with a decision tree based model, we need to use dummy variables instead of categorical levels. 
# # Why? Because this alters the bias of the algorithm who will favor a higher weight to the categories like 4 and deprioritize levels like 1.

# train_store_df = pd.get_dummies(train_store_df, columns = ["Assortment", "StoreType", "PromoInterval"], prefix = ["is_Assortment", "is_StoreType", "is_PromoInterval"])
# test_df = pd.read_csv("../input/test.csv", parse_dates = ["Date"])
# print("The Test Dataset has", test_df.shape[0], "observations and", test_df.shape[1], "variables")
# # We want to make sure that we consider all events, so lets join the test dataset with the store set.
# test_store_df = pd.merge(test_df, store_df, how = "left", on = "Store")

# print("Now we have", test_store_df.shape[0], "observations and", test_store_df.shape[1], "columns")
# # Lets create and conver the variables. Just like we did with the training dataset.

# test_store_df["Month"] = test_store_df.Date.dt.month
# test_store_df["Year"] = test_store_df.Date.dt.year
# test_store_df["Day"] = test_store_df.Date.dt.day

# test_store_df["StateHoliday"] = test_store_df["StateHoliday"].astype("category")
# test_store_df["PromoInterval"] = test_store_df["SchoolHoliday"].astype("category")
# test_store_df["StoreType"] = test_store_df["StoreType"].astype("category")
# test_store_df["Assortment"] = test_store_df["Assortment"].astype("category")

# test_store_df["StateHoliday_cat"] = test_store_df["StateHoliday"].cat.codes
# test_store_df["PromoInterval_cat"] = test_store_df["PromoInterval"].cat.codes
# test_store_df["StoreType_cat"] = test_store_df["StoreType"].cat.codes
# test_store_df["Assortment_cat"] = test_store_df["Assortment"].cat.codes

# test_store_df["StateHoliday_cat"] = test_store_df["StateHoliday_cat"].astype("float")
# test_store_df["PromoInterval_cat"] = test_store_df["PromoInterval_cat"].astype("float")
# test_store_df["StoreType_cat"] = test_store_df["StoreType_cat"].astype("float")
# test_store_df["Assortment_cat"] = test_store_df["Assortment_cat"].astype("float")

# test_store_df["CompetitionOpenSince"] = np.where((test_store_df["CompetitionOpenSinceMonth"] == 0) & (test_store_df["CompetitionOpenSinceYear"] == 0), 
#                                                 0,(test_store_df.Month - test_store_df.CompetitionOpenSinceMonth) + (12 *(test_store_df.Year - test_store_df.CompetitionOpenSinceYear)))

# test_store_df["StateHoliday_cat"] = test_store_df["StateHoliday"].map({0:0, "0": 0, "a": 1, "b": 1, "c": 1})

# test_store_df = test_store_df.drop(["Date", "StateHoliday", "StoreType", "Assortment", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "PromoInterval"], axis = 1)

# # test_store_df = test_store_df.sort_index(axis = 1).reset_index("Id") #make the ID variabe as index.

# print("So we have in our training set", train_store_df.shape[1], "variables and in our testing set", test_store_df.shape[1])
# print("in the testing set, we are missing only 3 and this are: Sales, Customer and Avg Sales per Customer")
# import tsfresh
