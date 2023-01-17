# Update seaborn

!pip install --upgrade seaborn
import pandas as pd

import numpy as np

import glob

import matplotlib.pyplot as plt

import os

import datetime

from datetime import time

import seaborn as sns



sns.set_theme()

pd.options.display.max_rows = 200
# Load



path = "/kaggle/input/vic-electricity-demand-covid19/"



max_temps = pd.read_csv(path + "melbourne_temps.csv", sep = " ")

min_temps = pd.read_csv(path + "melbourne_min_temps.csv", sep = " ")

    

# Drop unused columns

max_temps.drop(["product_code", "station_number", "accum_days_max", "quality"], inplace=True , axis=1)

min_temps.drop(["product_code", "station_number", "accum_days_min", "quality"], inplace=True , axis=1)



# Only look at data from 2016-2020

max_temps = max_temps[max_temps["year"] > 2015]

min_temps = min_temps[min_temps["year"] > 2015]



# Join into a single dataframe

weather = min_temps

weather["max_temperature"] = max_temps["max_temperature"]



# Trim the data so it finishes on 30SEP2020 and we can work with whole months

weather = weather.loc[0:2830,:]



# Create a datetime column

weather.set_index(pd.to_datetime(weather[["year","month","day"]]), inplace=True)



print(f"Weather data shape: {weather.shape}")

print(f"\nData types: \n{weather.dtypes}")

print(f"\nMissing Data? \n{weather.isnull().sum().sum()}")



weather.tail()
# The data is separated into monthly files, load all files and concat them together

i=1



for file in glob.glob(path + "VIC_demand/*.csv"):

    if i==1:

        demand = pd.read_csv(file)

        i+=1

        

    load = pd.read_csv(file)

    demand = pd.concat([demand, load], ignore_index=True)



print(f"Demand data shape: {demand.shape}")

print(f"\nData types:\n{demand.dtypes}")    

print(f"\nMissing Data? \n{demand.isnull().sum().sum()}")

print(f"\nPeriodtype unique values:\n{demand.PERIODTYPE.unique()}")

demand.head(2)
# Drop columns that we do not need

demand.drop(["REGION", "RRP", "PERIODTYPE"], inplace=True, axis=1)



# Convert the datecolumn to a date time object

demand.loc[:,"SETTLEMENTDATE"] = pd.to_datetime(demand["SETTLEMENTDATE"])

print(demand.dtypes)



# set the index to the date so we can join with the weather data

demand.set_index(demand["SETTLEMENTDATE"].dt.date, inplace=True)



demand.head(2)
# Join the weather and demand dataframes together

df = demand.join(weather)

df.drop_duplicates(inplace=True)



df.head(2)
df.tail(2)
# Create a weekday column

df["weekday"] = df["SETTLEMENTDATE"].dt.weekday



# Create a time column

df["time"] = df["SETTLEMENTDATE"].dt.time



# Convert weekday column to show 1 for a weekday and 0 for a weekend

df["weekday"] = df["weekday"].apply(lambda x: 1 if x in range(5) else 0)



# I also want a date column so add one here

df["date"] = df["SETTLEMENTDATE"].dt.date



df.head(2)
# Rename columns and reorder

df.columns = ["datetime", "demand", "year", "month", "day", "min", "max", "weekday", "time", "date"]

df = df[["datetime", "date", "year", "month", "day", "weekday", "min", "max", "demand", "time"]]

df.head(2)
# 2016 and 2020 are leap years, remove feb29 from these years for a fair comparison.

df = df.drop(df[(df.day == 29) & (df.month == 2)].index)
# Making an assumption that most of the differences will be seen during the day time

# When most people would be at work, but are currently working from home.

# Filter out all times from 8pm-8am



eight_am = datetime.time(7,0,0)

eight_pm = datetime.time(22,0,0)

daytime = df[(df["datetime"].dt.time >= eight_am) & (df["datetime"].dt.time <= eight_pm)]
# Group usage by day

day = daytime.groupby("date").mean()
# Calculate the year over year percentage change in daily average demand 2016-2019

# Create a function we can reuse later for weather adjusted demand

def calc_yearly_change(data, weather_adjusted=False):

       

    totals = []

    result = []

    

    # Check if we are using raw data or weather adjusted

    if weather_adjusted==True:

        demand_column = "weather_adjusted_demand"

    else:

        demand_column = "demand"

    

    for year in [2016,2017,2018,2019]:

        totals.append(data[(data["year"]==year)][demand_column].sum())

     

    for total in range(3):

        result.append(((totals[total+1] - totals[total]) / totals[total])*100) 

    

    # We need to compare only from January - September for 2019-2020

    nineteen = data[(data["month"]<=9)&(data["year"]==2019)].sum()[demand_column]

    twenty = data[(data["month"]<=9)&(data["year"]==2020)].sum()[demand_column]

    

    result.append(((twenty-nineteen)/nineteen)*100)

    

    # Round the results to three places

    result = np.round(result, 3)

    

    return result
# Use the function

change = calc_yearly_change(day)



# Check the results

print(f"\nAverage yearly percentage change 2016-2019: {np.round(np.mean(change[0:3]), 3)}%")

print(f"Demand percentage change 2019-2020: {change[3]}%")
change = {"Year":["16-17", "17-18", "18-19", "19-20"],

                 "percentage_change":change}



change = pd.DataFrame(change)

change.set_index("Year", drop=True, inplace=True)



change.plot(legend=None)

plt.title("Total Average Daily Demand 7am-10pm\nYear Over Year Percentage Change")

plt.ylabel("Percentage Change")

plt.savefig("year-over-year-percentage-change.svg", format="svg")

plt.show()
plt.figure(figsize=(6,4))

sns.scatterplot(x="max", y="demand", data=day, hue="weekday")

plt.title("Daily Max Temperature & Daily Average Demand \nVictoria January 2016 - September 2020, 7am-10pm")

plt.xlabel("Max Temperature")

plt.ylabel("Daily Average Demand (MWh)")

plt.savefig("weekdays_and_weather.svg", format="svg")

plt.show()



sns.scatterplot(x="min", y="demand", data=day, hue="weekday")

plt.title("Daily Min Temperature & Daily Average Demand \nVictoria January 2016 - September 2020, 7am-10pm")

plt.ylabel("Daily Average Demand (MWh)")

plt.xlabel("Min Temperature")

plt.show()



sns.violinplot(x="weekday", y="demand", data=day)

plt.title("Electricity Demand Distribution\nVictoria January 2016 - September 2020, 7am-10pm")

plt.ylabel("Daily Average Demand (MWh)")

plt.show()
lookup = day[["demand", "weekday"]].groupby([pd.cut(day["max"], np.arange(10,40,3)), "weekday"]).describe()

lookup.head()
weekdays = lookup[lookup.index.get_level_values('weekday').isin([1])]

weekends = lookup[lookup.index.get_level_values('weekday').isin([0])]

weekends.head()
def normalize(df):

    """

    Normalize each demand rating

    Takes ito account:

        - The daily max temperature

        - Weekend VS Weekday

        

    """

      

    # Create the grouped by dataframe

    lookup_df = df[["demand", "weekday"]].groupby([pd.cut(df["max"], np.arange(10,40,3)), "weekday"]).describe()

    

    # extract the values for weekdays

    weekdays = lookup_df[lookup_df.index.get_level_values('weekday').isin([1])]

    weekday_means = weekdays.loc[:,("demand", "mean")]

    weekday_stds = weekdays.loc[:,("demand", "std")]

    

    # extract the values for weekends

    weekends = lookup_df[lookup_df.index.get_level_values('weekday').isin([0])]

    weekend_means = weekends.loc[:,("demand", "mean")]

    weekend_stds = weekends.loc[:,("demand", "std")]

    

    # Loop over each value in the main dataframe

    for obs in df.itertuples():

        

        # For each value check which bin we need to use

        i=0

        for temp in np.arange(13,40,3):

            if obs[6] <= temp:

                # we have found the bin

                # check if it is a weekday or weekend

                if obs[4]==1:

                    # It is a weekday

                    # Norm the data in df

                    df.loc[obs[0], "weather_adjusted_demand"] = (df.loc[obs[0], "demand"] - weekday_means[i]) #/ weekday_stds[i]

                    

                else:

                    # It is a weekend

                    # Norm the data in df

                     df.loc[obs[0], "weather_adjusted_demand"] = (df.loc[obs[0], "demand"] - weekend_means[i]) #/ weekend_stds[i]

            else:

                i+=1

    df.reset_index(inplace=True)

    

    # Add the mean demand back to the adjusted demand so we can compare

    df["weather_adjusted_demand"] = df.weather_adjusted_demand.apply(lambda x: x+df.demand.mean())

    return df
# Call the function

day = normalize(day)
palette = sns.color_palette("husl", 3)

sns.set_palette(palette)
# Plot the result

plt.figure(figsize=(15,3))

sns.lineplot(x="date", y="demand", data=day, legend="brief")

sns.lineplot(x="date", y="weather_adjusted_demand", data=day, legend="brief")

plt.ylabel("Demand")

plt.legend(loc='upper left', labels=['Demand', 'Weather Adjusted Demand'])

plt.title("Daily Average Electricity Demand 7am-10pm")

plt.tight_layout()

plt.savefig("demand.svg", format="svg")

plt.show()
# Use the function

weather_adjusted_change = calc_yearly_change(day, weather_adjusted=True)



# Check the results

print(f"\nAverage yearly percentage change 2016-2019: {np.round(np.mean(weather_adjusted_change[0:3]), 3)}%")

print(f"Demand percentage change 2019-2020: {weather_adjusted_change[3]}%")
sns.set_theme()

palette = sns.color_palette("husl", 3)

sns.set_palette(palette)



change["percentage_change_weather_adjusted"] = weather_adjusted_change

change.plot()

plt.gca().invert_yaxis()

plt.title("Total Average Daily Demand 7am-10pm\nYear Over Year Percentage Change")

plt.ylabel("Percentage Change")

plt.savefig("year_over_year_change.svg", format="svg")

plt.show()
print(f"Year over year average percentage change: {change.percentage_change.mean()}")

print(f"Year over year average percentage change - adjusted for weather: {change.percentage_change_weather_adjusted.mean()}")

print(f"2019-2020 Change: {change.iloc[3,1]}")

change
# Read in the daylight data file

daylight = pd.read_csv(path + "daylight_hours.csv")



# Convert the date to a date object

daylight.loc[:,"date"] = pd.to_datetime(daylight["date"], dayfirst=True)



daylight.head(1)
def minutes(time_string):

    

    split = time_string.split(":")

    total = (int(split[0]) * 60) + int(split[1])

  

    return total



daylight["daylight_minutes"] = daylight["daylight"].apply(lambda x: minutes(x))





daylight.to_csv("daylight_hours.csv", header=True, index=False)
plt.figure(figsize=(15,3))

plt.title("Amount of Daylight in Minutes")

sns.lineplot(x="date", y="daylight_minutes", data=daylight)

plt.show()



plt.figure(figsize=(15,3))

plt.title("Normalized demand")

sns.lineplot(x="date", y="weather_adjusted_demand", data=day)

plt.show()
# Plot the result

sns.set_style("dark")

plt.figure(figsize=(15,3))

sns.lineplot(x="date", y="demand", data=day, legend="brief")

sns.lineplot(x="date", y="weather_adjusted_demand", data=day, legend="brief")

legend_one = plt.legend(loc='upper left', labels=['Demand', 'Weather Adjusted Demand'])

legend_one.remove()

plt.ylabel("Daily Average Demand (MWh)")

plt.title("Daily Average Electricity Demand 7am-10pm")

ax2 = plt.twinx()

sns.lineplot(x="date", y="daylight_minutes", data=daylight, ax=ax2, color=palette[2])

plt.legend(loc='upper right', labels=["Minutes of daylight"])

ax2.add_artist(legend_one)

plt.ylabel("Minutes of Daylight")

plt.tight_layout()

plt.savefig("demand.svg", format="svg")

plt.show()
sns.set_theme()

palette = sns.color_palette("husl", 5)

sns.set_palette(palette)
times = []

for i in range(7,23):

    times.append(i)

    times.append(i + 0.5)



sixteen = daytime[(daytime.year==2016)&(daytime.month<10)].groupby("time").mean()[["demand"]]

sixteen["times"]=times[:-1]

seventeen = daytime[(daytime.year==2017)&(daytime.month<10)].groupby("time").mean()[["demand"]]

seventeen["times"]=times[:-1]

eighteen = daytime[(daytime.year==2018)&(daytime.month<10)].groupby("time").mean()[["demand"]]

eighteen["times"]=times[:-1]

nineteen = daytime[(daytime.year==2019)&(daytime.month<10)].groupby("time").mean()[["demand"]]

nineteen["times"]=times[:-1]

twenty = daytime[daytime.year==2020].groupby("time").mean()[["demand"]]

twenty["times"]=times[:-1]





plt.figure(figsize=(9,6))

sns.lineplot(y="demand", x="times", data=sixteen)

sns.lineplot(y="demand", x="times", data=seventeen)

sns.lineplot(y="demand", x="times", data=eighteen)

sns.lineplot(y="demand", x="times", data=nineteen)

sns.lineplot(y="demand", x="times", data=twenty)

plt.legend(loc='upper left', labels=["2016", "2017", "2018", "2019", "2020"])

plt.xlabel("Hour (24hr time)")

plt.ylabel("Average Demand (MWh)")

plt.title("Average Daily Demand Distribution\nJanuary-September")

plt.savefig("distribution.svg", format="svg")

plt.show()
solar = pd.read_csv(path+"VIC_solar.csv")

solar.head()
solar.set_index(pd.to_datetime(solar.Month, format="%Y-%m"), inplace=True)

solar["Month"] = pd.to_datetime(solar.Month, format="%Y-%m")

solar.head()
solar.drop(["Postcode"], inplace=True, axis=1)



# Create a year column

solar["year"] = solar.Month.dt.year

solar.head()
# Convert KW to MW

solar["mW"] = solar["Capacity (kW)"] / 1000

solar.drop("Capacity (kW)", inplace=True, axis=1)
sns.set_theme()

solar[solar.year > 2014].groupby("year").sum().plot()

plt.ylabel("mW")

plt.xlabel("")

plt.title("Total Yearly Installed Solar Capacity Victoria")

plt.legend().remove()

plt.show()
# Look at the cumulative capacity of installed panels

solar.mW.cumsum().plot()

plt.title("Cumulative Capacity of Solar Panels Victoria")

plt.xlabel("")

plt.ylabel("mW")

plt.savefig("solar_capacity_vic.svg", format="svg")

plt.show()
df.head(1)
daylight.head(1)
# Merge daylight hours into our dataframe

df.set_index("date", inplace=True)

daylight.set_index("date", inplace=True)

df=pd.merge(df, daylight, how="left", left_index=True, right_index=True)

df.head(1)
public_holidays = pd.read_csv(path+"australian_public_holidays_2019.csv")

public_holidays = pd.concat([public_holidays, pd.read_csv(path+"australian_public_holidays_2020.csv")], ignore_index=True)

public_holidays = public_holidays[public_holidays["Jurisdiction"]=="vic"]

public_holidays.head()
public_holidays["Date"] = pd.to_datetime(public_holidays["Date"], format="%Y%m%d")

public_holidays.set_index("Date", inplace=True)

public_holidays["public_holiday"] = 1

public_holidays.drop(["Raw Date", "Holiday Name", "Information", "More Information", "Jurisdiction"], axis=1, inplace=True)

public_holidays.head()
df=pd.merge(df, public_holidays, how="left", left_index=True, right_index=True)

df[df["public_holiday"]==1].head(1)
# Only use data from 2019 onwards

df = df[df["year"] > 2018]



# Drop the columns which we will not need for this

df.drop(["year", "month", "day", "daylight"], inplace=True, axis=1)

df = df[["datetime", "min", "max", "daylight_minutes", "weekday", "public_holiday", "time", "demand"]]

df.head()
from sklearn.preprocessing import LabelEncoder 

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import mean_absolute_error as mae



le = LabelEncoder()

df["time"] = le.fit_transform(df["time"])

df.head()
# Currently our non public holiday days are NaN's replace them with zero

df["public_holiday"].fillna(0, inplace=True)

df.isna().sum().sum()
# Create a train test split before scaling our features

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:7], df.iloc[:,7], test_size=0.025, shuffle=False)
# Scale the features to the 0-1 interval

# Fit the scaler only on the training data

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)



# Scale the testing data also

X_test = scaler.transform(X_test)



y_train = np.asarray(y_train)

y_test = np.asarray(y_test)
from sklearn.tree import DecisionTreeRegressor



clf = DecisionTreeRegressor()

clf.fit(X_train, y_train)

preds = clf.predict(X_test)



# Check metrics

print(f"Mean Squared Error: {mse(y_test, preds)}")

print(f"Mean absolute Error: {mae(y_test, preds)}")



# Plot the results

plt.figure(figsize=(15,3))

plt.plot(df["datetime"].iloc[-len(y_test):], preds)

plt.plot(df["datetime"].iloc[-len(y_test):], y_test)

plt.title("Actual and Predicted demand - Decision tree model")

plt.legend(["Predicted Demand","Actual Demand"])

plt.show()
from sklearn.neighbors import KNeighborsRegressor



# Create and fit the model

knn = KNeighborsRegressor(3)

knn.fit(X_train, y_train)

preds = knn.predict(X_test)



# Check metrics

print(f"Mean Squared Error: {mse(y_test, preds)}")

print(f"Mean absolute Error: {mae(y_test, preds)}")



# Plot the results

plt.figure(figsize=(15,3))

plt.plot(df["datetime"].iloc[-len(y_test):], preds)

plt.plot(df["datetime"].iloc[-len(y_test):], y_test)

plt.title("Actual and Predicted demand - KNN model")

plt.ylabel("Demand (mW)")

plt.legend(["Predicted Demand","Actual Demand"])

plt.savefig("kNN.svg", format="svg")

plt.show()
from xgboost import XGBRegressor



xgb = XGBRegressor()

xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)



# Check metrics

print(f"Mean Squared Error: {mse(y_test, preds)}")

print(f"Mean absolute Error: {mae(y_test, preds)}")



# Plot the results

plt.figure(figsize=(15,3))

plt.plot(df["datetime"].iloc[-len(y_test):], preds)

plt.plot(df["datetime"].iloc[-len(y_test):], y_test)

plt.title("Actual and Predicted demand - XGB model")

plt.ylabel("Demand (mW)")

plt.legend(["Predicted Demand","Actual Demand"])

plt.savefig("kNN.svg", format="svg")

plt.show()