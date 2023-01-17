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
# Let's take a look at the data.
print("earthquakes shape is ", earthquakes.shape, ":", "landslides shape is ", landslides.shape, ":", "volcanos shape is ", volcanos.shape)
# Examine the individual dataframes.
earthquakes.head()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
# If you print the dtype rather than simply return the dtype, Python interprets the data type for you. In this case 'object' instead of 'O'.
print(earthquakes['Date'].dtype)
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y", errors='coerce')
#earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format='True')

# I suspect that some values of 'Date' were not converted correctly, which is why I coerced the errors in the line above. Coerced errors are converted to the 
#    'NaT' which is a null value. lets see if null values exist in the new variable.
missing_dates = earthquakes[earthquakes.date_parsed.isnull()]
print(missing_dates.shape)
# There are 3 missing date values. This is because the dates for these 3 observations are in a different format. For those 3 observations, let Python infer the date format.
earthquakes.loc[earthquakes.date_parsed.isnull() == True, 'date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

# Now lets see how many missing values we have.
missing_dates = earthquakes[earthquakes.date_parsed.isnull()]
print(missing_dates)
# Success! No more missing dates.
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())
# Your turn! get the day of the month from the earthquakes date_parsed column.
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
print(day_of_month_earthquakes.head())
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)