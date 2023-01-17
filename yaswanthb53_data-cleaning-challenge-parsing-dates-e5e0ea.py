# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
%matplotlib inline
# set seed for reproducibility
np.random.seed(0)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
earthquakes['Date'].head()
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date'].head(10)
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['parsed_date']=pd.to_datetime(earthquakes['Date'],infer_datetime_format=True)
earthquakes['parsed_date']
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
day_of_month_landslides
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides
# Your turn! get the day of the month from the date_parsed column
month=landslides['date_parsed'].dt.month
month
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes=earthquakes['parsed_date'].dt.day
day_of_month_earthquakes=day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes,kde=False,bins=30)
volcanos['Last Known Eruption'].head(15)
col=volcanos['Last Known Eruption']
new_date=list()
for item in col:
    if (item!='Unknown'):
        year,text=item.split()
        if text=='BCE':
            new_date.append(0-int(year))
        elif text=='CE':
            new_date.append(int(year))
    else:
        new_date.append(item)
volcanos['Last Known Eruption']=new_date
volcanos['Last Known Eruption']
