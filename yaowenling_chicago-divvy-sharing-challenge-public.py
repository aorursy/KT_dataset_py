# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Step 0: Load dataset

import pandas as pd

from time import time

t0 = time()

data = pd.read_csv("/kaggle/input/chicago-divvy-bicycle-sharing-data/data.csv")

print("Done in %0.3fs." % (time() - t0))
# Step 1: Data pre-processing

# 1.1 First look into the dataset

data.head()
print("The dataset contains {0[0]: ,.0f} rows and {0[1]: .0f} variables.".format(data.shape))
data.info() # There is no missing value in the dataset
# 1.2 Distribution across the years

import matplotlib.pyplot as plt

plt.style.use("seaborn")

fig = data["year"].value_counts().sort_index().plot(kind="bar")

plt.title("Number of Trips in Each Year")

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

year_month_pivot = pd.pivot_table(data, index="year", columns="month", values="trip_id", aggfunc="count")

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(year_month_pivot)

plt.title("Number of Trips per Year and per Month")
# 1.2 Create weekend column

t0 = time()

data["day"].value_counts() # Confirm that this column indicates the day of week

data[["starttime", "day"]].head() # Confirm the coding of DOW: 0 = Monday

dowmapping = pd.DataFrame.from_dict({"0": "Monday", "1": "Tuesday", "2": "Wednesday", 

                                     "3": "Thursday", "4": "Friday", "5": "Saturday", 

                                     "6": "Sunday"}, 

                                    orient="index")

dowmapping.reset_index(inplace=True)

dowmapping.columns = ["num_coding", "dow"]

dowmapping["num_coding"] = dowmapping["num_coding"].astype("int64")

data = pd.merge(data, dowmapping, how="left", left_on="day", right_on="num_coding")

del(data["num_coding"])

data["weekend"] = data["day"].apply(lambda x: x in (5, 6))

print("Done in %0.3fs." % (time()-t0))
# 1.3 Create date column

# This process can take 15 minutes.

# t0 = time()

# from datetime import datetime

# data["starttime"] = data["starttime"].astype("str")

# data["starttime"] = data["starttime"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

# data["startdate"] = data["starttime"].apply(lambda x: x.date())

# data["starthour"] = data["starttime"].apply(lambda x: x.replace(minute=0))

# data["startmonth"] = data["startdate"].apply(lambda x: x.replace(day=1))

# # Since this step takes 15 minutes to complete this step I will save the file to avoid long runtime.

# data.to_csv("data_preprocess.csv")

# print("Done in %0.3fs." % (time()-t0))
# data = pd.read_csv("data_preprocess.csv")
# 1.4 Creat starthour pivot table for prediction modelling

# starthour_pivot = pd.pivot_table(data=data,

#                                  index=["starthour", 

#                                    "year", "month", "day", "hour", "weekend", "startdate",

#                                    "temperature", "events", 

#                                    "from_station_id", "from_station_name"], 

#                                  values="trip_id", aggfunc="count")

# starthour_pivot = pd.DataFrame(starthour_pivot)

# starthour_pivot.to_csv("starthour_pivot.csv")
data.describe(include="all")
# Step 2: Explanatory analysis

# 2.1 Temporal trends

# Monthly trend

year_month = pd.DataFrame(data.groupby(["year", "month"]).count()["trip_id"])

year_month.reset_index(inplace=True)

year_month.columns = ["year", "month", "count_trips"]

g = sns.catplot(x="month", y="count_trips", col="year", data=year_month, kind="bar")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Number of Trips per Month')

plt.show()
# DOW trend

year_day = pd.DataFrame(data.groupby(["year", "day"]).count()["trip_id"])

year_day.reset_index(inplace=True)

year_day.columns = ["year", "day", "count_trips"]

g = sns.catplot(x="day", y="count_trips", col="year", data=year_day, kind="bar")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Number of Trips per DOW (0=Monday)')

plt.show()
# Hourly trend

day_hour = pd.DataFrame(data[data["year"]==2017].groupby(["day", "hour"]).count()["trip_id"])

day_hour.reset_index(inplace=True)

day_hour.columns = ["day", "hour", "count_trips"]

g = sns.catplot(x="hour", y="count_trips", col="day", data=day_hour, kind="bar")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Number of Trips per Hour (day=0 means Monday)')

plt.show()
# 2.2 Geo patterns

# Import packages

import plotly.graph_objects as go

import plotly.express as px
# Generate the list of stations

start = data[["latitude_start", "longitude_start", "from_station_name"]]

start.columns = ["lat", "lon", "station_name"]

end = data[["latitude_end", "longitude_end", "to_station_name"]]

end.columns = ["lat", "lon", "station_name"]

station_info = (pd.concat([start, end])).groupby("station_name").mean()[["lat", "lon"]] # One station name can be mapped with multiple pairs of latitude and longitude

station_info.reset_index(inplace=True)

station_info.columns = ["station_name", "lat", "lon"]



station_start_trips = pd.DataFrame(data[data["year"]==2017]["from_station_name"].value_counts())

station_start_trips.reset_index(inplace=True)

station_start_trips.columns = ["station_name", "count_start_trips_2017"]

station_info = station_info.merge(station_start_trips, how="left")



station_end_trips = pd.DataFrame(data[data["year"]==2017]["to_station_name"].value_counts())

station_end_trips.reset_index(inplace=True)

station_end_trips.columns = ["station_name", "count_end_trips_2017"]

station_info = station_info.merge(station_end_trips, how="left")



station_info.info()
station_info = station_info.dropna()
station_info["count_start_trips_2017"].describe()
quantiles = []

levels = []

for i in range(11):

    quantiles.append(np.percentile(station_info["count_start_trips_2017"], i*10))

    levels.append("level_"+str(i+1))

station_info["count_start_trips_intervals"] = pd.cut(station_info["count_start_trips_2017"], quantiles, labels=levels[:-1])

station_info[station_info["count_start_trips_2017"]==3]["count_start_trips_intervals"] = "level_1"

station_info = station_info.sort_values(by="count_start_trips_2017", ascending=True)
plt.figure(figsize=(10,6))

sns.scatterplot(x="lon", y="lat", hue="count_start_trips_intervals", palette="RdBu",

                data=station_info, alpha=0.5)

plt.title("Scatter Plot of Stations that were active in 2017 (colored by heat)")
# Plot all stations on Chicago map

# px.set_mapbox_access_token("XXX") # Replace XXX with your Mapbox Token

# fig = px.scatter_mapbox(station_info.sort_values(by="count_start_trips_2017", ascending=True),

#                         lat="lat", 

#                         lon="lon",

#                         color="count_start_trips_2017",

#                         color_continuous_scale=px.colors.sequential.Plasma,

#                         text="station_name",

#                         opacity=0.7, 

#                         zoom=10)

# fig.show()
station_info = station_info.sort_values(by="count_start_trips_2017", ascending=False)

station_info["count_start_trips_cumsum"] = station_info["count_start_trips_2017"].cumsum()

(station_info["count_start_trips_cumsum"].reset_index(drop=True) / sum(station_info["count_start_trips_2017"])).plot()

plt.title("Cum. Density of Number of Trips Starting from Each Station")

plt.show()
# Plot an animation of scatter plot to see which are the most popular start stations in each hour

start_station_hour = data[data["year"]==2017].groupby(["from_station_name", "hour"]).count()["trip_id"].reset_index()

start_station_hour = start_station_hour.merge(station_info[["station_name", "lat", "lon"]], how="left", left_on="from_station_name", right_on="station_name")

del(start_station_hour["from_station_name"])

start_station_hour = start_station_hour.rename({"trip_id": "count_start_trips"}, axis="columns")
for hour in range(24):

    # Slice the whole dataframe

    station_start_trips_per_hour = start_station_hour[start_station_hour["hour"]==hour]

    # Take the hourly max

    hourly_max = station_start_trips_per_hour["count_start_trips"].max()

    # Generate labels

    start_station_hour.loc[start_station_hour["hour"]==hour, "start_trips_heat_per_hour"] = start_station_hour.loc[start_station_hour["hour"]==hour, "count_start_trips"].apply(lambda x: x/hourly_max)
fig = px.scatter(start_station_hour.sort_values(by=["hour", "count_start_trips"], ascending=[True, True]),

                 x="lon", y="lat", 

                 animation_frame="hour",

                 color="start_trips_heat_per_hour", 

                 opacity=0.5,

                 size="count_start_trips", size_max=60,

                 range_x=[-87.9,-87.5], range_y=[41.7, 42.1])

fig.update_layout(title="Hourly Start Trips from Each Station (Year 2017)")

fig.show()
station_info.info()
starthour = pd.read_csv("/kaggle/input/starthour-pivot/starthour_pivot.csv")

starthour = starthour.rename({"trip_id": "count_start_trips"}, axis="columns")

starthour.head()
print("We have {0[0]: ,.0f} records of hourly number of trips starting from each station.".format(starthour.shape))
# 2.3 Temparature

starthour["temperature"].hist()

plt.title("Histogram of Temperature")

plt.show()
starthour.head(10)
starthour["events"].value_counts()
# from datetime import datetime

unknown_events = starthour[starthour["events"]=="unknown"]

unknown_events.groupby(["starthour"]).count()
# Check the weather condition on "201x-04-18 11:00:00"

starthour[starthour["starthour"].apply(lambda x: x in ["2015-04-18 11:00:00", "2016-04-18 11:00:00", "2017-04-18 11:00:00"])]                                               
# Fill the weather condition for "2014-04-18 11:00:00" as "cloudy"

starthour.loc[starthour["starthour"]=="2014-04-18 11:00:00", "events"] = "cloudy"
# Check the weather condition on "201x-06-30 10:00:00"

starthour[starthour["starthour"].apply(lambda x: x in ["2014-06-30 10:00:00", "2015-06-30 10:00:00", "2017-06-130 10:00:00"])]                                                                 
# Fill the weather condition for "2016-06-30 10:00:00" as "cloudy"

starthour.loc[starthour["starthour"]=="2016-06-30 10:00:00", "events"] = "cloudy"
starthour_agg = pd.pivot_table(starthour,

                               index=['starthour', 'year', 'hour', 'temperature', 'events'],

                               values=['count_start_trips'], 

                               aggfunc=np.sum)

starthour_agg.reset_index(inplace=True)
hour_temp_sum_trips_2017 = pd.pivot_table(starthour_agg[starthour_agg["year"]==2017],

                                          index=['hour', 'temperature'],

                                          values=['starthour', 'count_start_trips'], 

                                          aggfunc={"starthour": np.count_nonzero, "count_start_trips": np.sum})

hour_temp_sum_trips_2017.reset_index(inplace=True)

hour_temp_sum_trips_2017["avg_trips"] = hour_temp_sum_trips_2017["count_start_trips"] / hour_temp_sum_trips_2017["starthour"]
g = sns.FacetGrid(hour_temp_sum_trips_2017,

                  col="hour", col_wrap=4,

                  margin_titles=True)

g.set(ylim=(0, 60))

g = g.map(plt.scatter, "temperature", "avg_trips", s=25, color="c", alpha=0.3)
# 2.4 Weather conditions

hour_event_sum_trips_2017 = pd.pivot_table(starthour_agg[starthour_agg["year"]==2017],

                                           index=['hour', 'events'],

                                           values=['starthour', 'count_start_trips'], 

                                           aggfunc={"starthour": np.count_nonzero, 

                                                    "count_start_trips": np.sum})

hour_event_sum_trips_2017.reset_index(inplace=True)

hour_event_sum_trips_2017["avg_trips"] = hour_event_sum_trips_2017["count_start_trips"] / hour_event_sum_trips_2017["starthour"]
g = sns.FacetGrid(hour_event_sum_trips_2017,

                  col="hour", col_wrap=4, hue="events",

                  height=4, aspect=1.5,

                  margin_titles=True)

g.set(ylim=(0, 60))

g = g.map(plt.bar, "events", "avg_trips")
starthour.info()
starthour.head()
starthour2017 = starthour[starthour["year"]==2017]

starthour2017 = starthour2017.merge(station_info[["station_name", "lat", "lon", "count_start_trips_intervals"]], 

                                    how = "left",

                                    left_on ="from_station_name", right_on ="station_name")

del(starthour)
starthour2017.head()
# Step 3: Run prediction model

# 3.1 Create dummy variables

weekend = pd.get_dummies(starthour2017["weekend"], prefix="weekend", drop_first=True)

events = pd.get_dummies(starthour2017["events"], prefix="event", drop_first=True)

station_heat = pd.get_dummies(starthour2017["count_start_trips_intervals"], prefix="station_heat", drop_first=True)

X = starthour2017[["month", "day", "hour", "temperature", "lat", "lon"]]

for cols in [weekend, events, station_heat]:

    X = pd.concat([X, cols], axis=1)

    del(cols)
# 3.2 Split train and test set

from sklearn.model_selection import train_test_split

y = starthour2017[['count_start_trips']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

del(X)

del(y)
# 3.4 Linear Regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

lr = LinearRegression()

lr.fit(X_train, y_train)

y_test_pred = lr.predict(X_test)

print("The R-square score of linear model on test set is {0: .4f}.".format(r2_score(y_test, y_test_pred)))

print("The MSE of linear model on test set is {0: .4f}.".format(mean_squared_error(y_test, y_test_pred)))
# 3.5 GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

# gbr = GradientBoostingRegressor()

# parameters = {

#     'n_estimators': [100, 120, 140],

#     'max_depth': [3, 5, 7]

# }

# grid_search = GridSearchCV(estimator=gbr, param_grid=parameters, cv=3, n_jobs=1)

# grid_search.fit(X_train, y_train)

# print("The best parameters are: \n")

# print(grid_search.best_params_)

t0 = time()

gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3)

gbr.fit(X_train, y_train)

print("Done in %0.3fs." % (time()-t0))

y_test_pred = gbr.predict(X_test)

print("The R-square score of GBR model on test set is {0: .4f}.".format(r2_score(y_test, y_test_pred)))

print("The MSE of GBR model on test set is {0: .4f}.".format(mean_squared_error(y_test, y_test_pred)))

gbr_feature_inportance = pd.DataFrame(gbr.feature_importances_, index=X_train.columns)

gbr_feature_inportance.sort_values(by=0, ascending=True).tail(10).plot(kind="barh")

plt.title("Feature Importance by GBR Model")

plt.show()
# 3.3 XGBoost

import xgboost as xgb

t0 = time()

xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=3)

xgb_reg.fit(X_train, y_train)

print("Done in %0.3fs." % (time()-t0))

y_test_pred = xgb_reg.predict(X_test)

print("The R-square score of XGBoost model on test set is {0: .4f}.".format(r2_score(y_test, y_test_pred)))

print("The MSE of XGBoost model on test set is {0: .4f}.".format(mean_squared_error(y_test, y_test_pred)))

xgb_feature_inportance = pd.DataFrame(xgb_reg.feature_importances_, index=X_train.columns)

xgb_feature_inportance.sort_values(by=0, ascending=True).tail(10).plot(kind="barh")

plt.title("Feature Importance by XGBoost Model")

plt.show()