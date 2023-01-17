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
#landslides['date'].dtype
landslides.date.dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.Date.dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'])
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head() 
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
#volcanos['Last Known Eruption'].sample(5)

#volcanos['XXX'] = volcanos[volcanos['Last Known Eruption'].str.contains("Unknown") == False]
# Above will not work It's because you're trying to put more than one column into a single column. 
#Use this: 
#volcanos['XXX'] = volcanos[volcanos['Last Known Eruption'].str.contains("Unknown") == False]['Last Known Eruption']

#olcanos_known_dates  = volcanos[volcanos['Last Known Eruption'].str.contains("Unknown") == False]
#volcanos_known_dates.groupby(["Last Known Eruption"]).count()

volcanos_known_dates_bce  = volcanos[volcanos['Last Known Eruption'].str.contains(" BCE") == True]
volcanos_known_dates_ce  = volcanos[volcanos['Last Known Eruption'].str.contains(" CE") == True]

#volcanos_known_dates_ce.sample(5)
#volcanos_known_dates_ce['year'] = volcanos_known_dates_ce['Last Known Eruption'].str[:4]
#volcanos_known_dates_bce['year'] = pd.to_numeric(volcanos_known_dates_bce['Last Known Eruption'].str[:4])

# makes a boolean vector with "True" if 'Last Known Eruption' isn't "Unknown"
date_unknown = volcanos['Last Known Eruption'].str.contains("Unknown") == False
# creates a new column "Date Unknown", that has the boolean vector in it
volcanos["Date Unknown"] = date_unknown
# use our boolean vector to filter the "Last Known Eruption" column
known_dates = volcanos.loc[date_unknown, "Last Known Eruption"]

# put our filtered column in a new column called "Known Dates"
# (the ones we filtered out will be replaced with "NaN")
volcanos["Known Dates"] = known_dates
volcanos['date_parsed'] = volcanos[volcanos['Last Known Eruption'].str.endswith('CE')]['Last Known Eruption']
last_known_eruption_dates = volcanos['date_parsed'].dropna()
last_known_eruption_dates.sample(5)
eruption_dates_int = last_known_eruption_dates.apply(lambda x: -int(x[:-4]) if x.endswith('BCE') else int(x[:-3]))
eruption_dates_int #FINAL
'''
https://www.kaggle.com/rtatman/data-cleaning-challenge-parsing-dates/comments
# makes a boolean vector with "True" if 'Last Known Eruption' isn't "Unknown"
date_unknown = volcanos['Last Known Eruption'].str.contains("Unknown") == False

# get a dataframe made up of only the rows where the boolean vector is True
volcanos[date_unknown]

# creates a new column "Date Unknown", that has the boolean vector in it
volcanos["Date Unknown"] = date_unknown

# trying to put the whole dataframe in a single column (won't work!)
# volcanos["Dataframe Column"] = volcanos

# use our boolean vector to filter the "Last Known Eruption" column
known_dates = volcanos.loc[date_unknown, "Last Known Eruption"]

# put our filtered column in a new column called "Known Dates"
# (the ones we filtered out will be replaced with "NaN")
volcanos["Known Dates"] = known_dates


'''