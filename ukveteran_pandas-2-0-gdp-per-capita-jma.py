import pandas as pd
data=pd.read_csv("../input/gdp-per-capita/gdp_per_capita_2020July.csv", skiprows=4)
data.head()
# set index of the dataframe as Country Name

data.set_index("Country Name", inplace=True)
data.head()
#delete Country Code, Indicator Name and Indicator Code

data.drop(labels=["Country Code","Indicator Name","Indicator Code","Unnamed: 64"],axis="columns",inplace=True)
# check the dataframe

data.head()
italy=data.loc["Italy"]

italy
type(italy)
# get italy's gdp per capita for 2019

italy["2019"]
italy
italy_pct_change=italy.pct_change()
# multiply italy_pct_change by 100

italy_pct_change=italy_pct_change*100
italy_pct_change
italy_pct_change.idxmax()
italy_negative=italy_pct_change[italy_pct_change>0]
# get only years

italy_negative.index
data.head()
# first get the data for world

# there is an index named World

world=data.loc["World"]
# now find the year that has highest increase in percentange

world.pct_change().idxmax()
# first calculate standart deviation for each year

data_std=data.std()
data_std
# now top-10 years having the hightest standart deviation

data_std.nlargest(10)
data.head()
# get the data for the year 2019

data_2019=data["2019"]
data_2019
# now get the country

data_2019.idxmax()
data_2019.nlargest(1)
# calculate the rank of each country for each year

data_rank=data.rank(ascending=False)
data_rank
# now get the rank of Luxembourg

lux_rank=data_rank.loc["Luxembourg"]
lux_rank
# plot the rank of luxembourg

lux_rank.plot()
# print the years where Luxembourg ranks top

lux_rank[lux_rank==1]