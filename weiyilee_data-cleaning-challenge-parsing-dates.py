# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes["Date"].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# There are invalid date format in the data.
# My goal is to locate those data and then decide how to deal with them individually.
# Step1: get the index of invalid date
earthquakes["date_parsed"]=pd.to_datetime(earthquakes["Date"], format="%m/%d/%Y", errors="coerce")
invalid_date_index=earthquakes["date_parsed"][earthquakes["date_parsed"].isnull()==True].index.tolist()
# Step2: print out invalid date
for index in invalid_date_index:
    print("index {} has date {}".format(index, earthquakes.loc[index,["Date"]].values))
# Step3: fix invalid date one by one
## As we see, we can still parse the date we want by slicing the original date string.
fixed_date=[]
for index in invalid_date_index:
    sliced=earthquakes.loc[index,["Date"]].values[0][:10]
    fixed_date.append((index, sliced))
## then put the correct date format back to our dataframe  
for fixed in fixed_date:
    earthquakes.loc[fixed[0], "date_parsed"]=pd.to_datetime(fixed[1], format="%Y-%m-%d")
## check fixed values in column "date_parsed"
earthquakes.iloc[[3378, 7512, 20650]]
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)