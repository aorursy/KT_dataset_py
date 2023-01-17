import pandas as pd
import numpy as np
from scipy import stats
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import chardet
import fuzzywuzzy
from fuzzywuzzy import process
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
np.random.seed(0) 
nfl_data.sample(5)
missing_values_count = nfl_data.isnull().sum()
missing_values_count[0:10]
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
(total_missing/total_cells) * 100
nfl_data.dropna()
columns_with_na_dropped = nfl_data.dropna(axis=1)
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
columns_with_na_dropped.head()
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
subset_nfl_data.fillna(0)
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
original_data = np.random.exponential(size = 1000)
scaled_data = minmax_scaling(original_data, columns = [0])
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()
normalized_data = stats.boxcox(original_data)
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
plt.show()
print(landslides['date'].head())
landslides['date'].dtype
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides['date_parsed'].head()
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
day_of_month_landslides = day_of_month_landslides.dropna()
sns.distplot(day_of_month_landslides, kde=False, bins=31)
plt.show()
before = "This is the euro symbol: €"
type(before)
after = before.encode("utf-8", errors = "replace")
type(after)
after
print(after.decode("utf-8"))
#print(after.decode("ascii"))
before = "This is the euro symbol: €"
after = before.encode("ascii", errors = "replace")
print(after.decode("ascii"))
#kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
print(result)
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')
kickstarter_2016.head()
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
with open("../input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
suicide_attacks = pd.read_csv("../input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')
cities = suicide_attacks['City'].unique()
cities.sort()
cities
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
cities = suicide_attacks['City'].unique()
cities.sort()
cities
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    strings = df[column].unique()
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    rows_with_matches = df[column].isin(close_matches)
    df.loc[rows_with_matches, column] = string_to_match
    print("All done!")
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
cities = suicide_attacks['City'].unique()
cities.sort()
cities