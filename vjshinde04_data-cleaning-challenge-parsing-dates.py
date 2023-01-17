# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
earthquakes.sample(5)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].sample(5)
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
earthquakes['Date'].sample(5)
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

#earthquakes['dateParsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%y")

#print (pd.to_datetime(earthquakes['Date'], errors='coerce', format="%m/%d/%Y"))
#mask = pd.to_datetime(earthquakes['Date'], errors='coerce', format="%m/%d/%Y").isnull()
#print (earthquakes['Date'][mask])

earthquakes['dateParsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format= True)
earthquakes['dateParsed'].sample(5)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.sample(5)
# Your turn! get the day of the month from the date_parsed column
dayOfMonth_earthquakes = earthquakes['dateParsed'].dt.day
dayOfMonth_earthquakes.sample(5)
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
dayOfMonth_earthquakes = dayOfMonth_earthquakes.dropna()
sns.distplot(dayOfMonth_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
volcanos.head(5)
v = pd.read_csv("../input/volcanic-eruptions/database.csv", parse_dates=True)
v.head(5)
#does'nt wrok here
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
volcanos['Last Known Eruption'] = volcanos['Last Known Eruption'].replace('Unknown', np.nan)
volcanos.head(5)
volcanos.dtypes
#volcanos = volcanos.dropna()
#is droppping all the data
categoricalVar = volcanos.dtypes.loc[volcanos.dtypes == 'object'].index
categoricalVar
volcanos[categoricalVar].apply(lambda x: sum(x.isnull()))
volcanos.shape
volcanos.head(10)
volcanos = volcanos.dropna()
volcanos.head(10)
#removinh BCE and CE
#last_known_eruption_dates.apply(lambda x: -int(x[:-4]) if x.endswith('BCE') else int(x[:-3]))
volcanos['lastKnownEruptionParsed'] = volcanos['Last Known Eruption'].apply(lambda x: -int(x[:-4]) if x.endswith('BCE') else int(x[:-3]))
volcanos.head()
tectonicColoumn = volcanos['Tectonic Setting'].unique()
tectonicColoumn
#We can plot and find : In which year what layer of earth was affected
#For this we'll have to take values of depth from 'Tectonic Setting' and use it ;)
volcanos['Tectonic Setting'] = volcanos['Tectonic Setting'].astype('str')
volcanos['Tectonic Setting'].unique()
#s[s.find("(")+1:s.find(")")]
import re
abc = abd = 'Subduction Zone / Oceanic Crust (< 15 km)'
abd[abd.find("(")+1 : abd.find(")")]
#tectonicDepth = volcanos['Tectnoic Setting'].apply(lambda x: )
import re
volcanos['tectonicDepth'] = volcanos['Tectonic Setting'].apply(lambda x : x[x.find("(")+1 : x.find(")")])
volcanos.sample(10)
volcanos['tectonicDepth'].unique()
volcanos['tectonicDepth'] = volcanos['tectonicDepth'].replace('Subduction Zone / Crust Thickness Unknow', 'Unknown')
sns.swarmplot(x=volcanos['tectonicDepth'], y=volcanos['lastKnownEruptionParsed'], hue=volcanos['Type'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)