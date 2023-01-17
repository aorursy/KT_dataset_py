# Loading packages

import numpy as np

import pandas as pd



# Loading required data

richlist = pd.read_csv("../input/World_Richest.csv")

hdi = pd.read_csv("../input/Human Development Index (HDI).csv", header=1)



# Cleaning up worlds richest data

richlist.Age = richlist.Age.str.extract('(\d+)', expand = True)
# Having a look at the list of Rich people by country

richlist.Country.value_counts()
# Creating a new dataframe with required richlist data for merge with HDI data

merge_richlist = pd.DataFrame(richlist.groupby(['Country']).Asset.sum().sort_values(ascending=False))

merge_richlist.reset_index(inplace = True)



# Having a look at the richlist data by Asset and Country

merge_richlist
# Creating a new dataframe with required HDI data for merge with richlist

merge_hdi = pd.DataFrame(hdi[['Country', 'HDI Rank (2015)']].sort_values('HDI Rank (2015)'))

merge_hdi.reset_index()



# Cleaning up country column

merge_hdi.Country = merge_hdi.Country.str.strip()



# Creating a new dataframe with merged data

compare = pd.merge(merge_hdi, merge_richlist, on = ['Country'], how='inner')
# Sort compare by total Asset and the two lists by countries

compare.sort_values('Asset', ascending = False)