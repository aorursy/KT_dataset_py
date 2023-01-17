# modules we'll use
import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

# set seed for reproducibility
np.random.seed(0)
# look at the first ten thousand bytes to guess the character encoding
with open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)
# read in our dat
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')
suicide_attacks.head()
suicide_attacks.tail()
suicide_attacks.info()
suicide_attacks.shape
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
type(cities)
sorted(cities)
suicide_attacks.City
# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
suicide_attacks['City']
suicide_attacks.head()
suicide_attacks.Province.unique()
suicide_attacks.Province.unique().size
suicide_attacks.Province.unique().shape
suicide_attacks.Province.isnull().sum()
suicide_attacks.Province.size
sorted(suicide_attacks.Province.unique())
suicide_attacks.Province = suicide_attacks.Province.str.lower()
suicide_attacks.Province = suicide_attacks.Province.str.strip()
sorted(suicide_attacks.Province.unique())
suicide_attacks.Province.unique().sort()
suicide_attacks.Province.unique()
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# get the top 10 closest matches to "d.i khan"
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")
# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
strings = suicide_attacks['City'].unique()
strings
strings1 = strings.copy()
type(strings)
strings1.sort()
strings1
matches = fuzzywuzzy.process.extract('kuram agency',strings,
                                    limit=10,scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
close_matches = [match[0] for match in matches if match[1] >= 90]
close_matches
rows_with_matches = suicide_attacks.City.isin(close_matches)
rows_with_matches
rows_with_matches.sort_values()
type(rows_with_matches)
rows_with_matches.describe()
rows_with_matches
c = suicide_attacks.City
c
c == 'karachi'
rows_with_matches is True
suicide_attacks.loc[rows_with_matches, 'City']
suicide_attacks.loc[suicide_attacks.City.isin(close_matches), 
                    'City']
suicide_attacks.loc[rows_with_matches, 'City'] = 'kuram agency'
cities1 = suicide_attacks.City.unique()
cities1.sort()
cities1
