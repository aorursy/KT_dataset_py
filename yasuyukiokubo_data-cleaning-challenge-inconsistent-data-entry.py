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
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
cities_fixed = suicide_attacks["City"].unique()
cities_fixed.sort()
cities_fixed
# Your turn! Take a look at all the unique values in the "Province" column. 
# Then convert the column to lowercase and remove any trailing white spaces
provinces = suicide_attacks["Province"].unique()
provinces.sort()
provinces
suicide_attacks["Province"] = suicide_attacks["Province"].str.lower()
suicide_attacks["Province"] = suicide_attacks["Province"].str.strip()
provinces_fixed = suicide_attacks["Province"].unique()
provinces_fixed.sort()
provinces_fixed
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
# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.
#C "kurram agency" is the right one by -> https://en.wikipedia.org/wiki/Kurram_Agency
matches_try = fuzzywuzzy.process.extract("kurram agency", cities, limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)
matches_try
#C Matching function

min_ratio = 90
limit = 10
def replace_to_match_string(df, column, string_to_match, min_ratio = min_ratio):
    
    strings = df[column].unique()
    matches = fuzzywuzzy.process.extract(string_to_match,
                                         strings,
                                         limit = limit,
                                         scorer = fuzzywuzzy.fuzz.token_sort_ratio)
    closed_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    rows_to_match = df[column].isin(closed_matches)
    df.loc[rows_to_match, column] = string_to_match
    
    print("We are all done!")
replace_to_match_string(df = suicide_attacks, column = "City", string_to_match = "kurram agency")
cities_try = suicide_attacks["City"].unique()
cities_try.sort()
cities_try
