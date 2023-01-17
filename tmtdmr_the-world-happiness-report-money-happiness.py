# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv",sep=",")
data.info()
data.head()
# I change "." by "_" in column names to prevent to fail. Beacause "." is specific character in methods

data.columns = [each.replace('.','_') for each in data.columns]  
# corelation map

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt='.2f', ax=ax)
# histogram (frequency of Happiness Score)

plt.hist(data.Happiness_Score, bins=50)

plt.xlabel("Happiness Score")

plt.ylabel("Frequency")

plt.title("hist")

plt.show()

# line graphic family-happiness score

plt.plot(data.Family, data.Happiness_Score, color='red')

plt.xlabel("Family")

plt.ylabel("Happiness Score")

plt.grid()

plt.legend()

plt.show()
# line graphic GDP-Happiness Score

plt.plot(data.Economy__GDP_per_Capita_, data.Happiness_Score, color='blue')

plt.xlabel("GDP")

plt.ylabel("Happiness Score")

plt.grid()

plt.legend()

plt.show()



# average happiness

average_happiness = data.Happiness_Score.mean()
# countries which has higher value than average happiness

filter_higher_average_happiness = data.Happiness_Score>average_happiness

filtred_higher_data = data[filter_higher_average_happiness]
# countries which has lower value than average happiness

filter_lower_average_happiness = data.Happiness_Score<average_happiness

filtred_lower_data = data[filter_lower_average_happiness]
# countires which is in top 15 happiet countries

filter_happy_countries = data.Happiness_Rank<16

happiest_countries = data[filter_happy_countries]
# countries which is in last 15 happiest countries

nonhappy_countries = data.tail(15)
# top 15

happiest_countries
# the last 15:

nonhappy_countries
# visualization together Happy and Unhappy countries (15 countries) in scatter graphic

plt.scatter(happiest_countries.Economy__GDP_per_Capita_, happiest_countries.Happiness_Score,color="blue",label="Happy Countries")

plt.scatter(nonhappy_countries.Economy__GDP_per_Capita_, nonhappy_countries.Happiness_Score,color="red",label="Unhappy Countries")



plt.legend()

plt.xlabel("Economy")

plt.ylabel("Happiness")

plt.title("Economy-Happiness")

plt.show()

'''Graphics says there is a relation between happiness scor and economy but I think there are some exceptions that

some countries have good economy (in top 30 economy) but there aren't in top 30 happies country. In contrast

to this some countries in top 30 happy countries but their economies are not in top 30 economy. For this reason

I create 2 groups that are poor-happy countries and rich-unhappy countries

'''
# rich and unhappy

economy_sorted = data.Economy__GDP_per_Capita_.sort_values(ascending=False)

list_of_economy_sorted = list(economy_sorted)





rich_and_unhappy = data[(data.Economy__GDP_per_Capita_> list_of_economy_sorted[30]) & (data.Happiness_Rank>30) ]

rich_and_unhappy
# rich and unhappy corelation map

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(rich_and_unhappy.corr(),annot=True, linewidths=.5, fmt='2f', ax=ax)
# poor and happy

poor_and_happy = data[(data.Economy__GDP_per_Capita_< list_of_economy_sorted[30]) & (data.Happiness_Rank<30) ]

poor_and_happy
# poor and happy corelation map

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(poor_and_happy.corr(),annot=True, linewidths=.5, fmt='2f', ax=ax)
# visualization freedom-corruption together with poor happy and rich unhappy countries

plt.scatter(rich_and_unhappy.Freedom, rich_and_unhappy.Trust__Government_Corruption_,color="blue",label="Happy Countries")

plt.scatter(poor_and_happy.Freedom, poor_and_happy.Trust__Government_Corruption_,color="red",label="Unhappy Countries")



plt.legend()

plt.xlabel("Freedom")

plt.ylabel("Corruption")

plt.title("Freedom-Corruption")

plt.show()
# visualization life expextancy-economy

plt.scatter(rich_and_unhappy.Health__Life_Expectancy_, rich_and_unhappy.Economy__GDP_per_Capita_,color="blue",label="Happy Countries")

plt.scatter(poor_and_happy.Health__Life_Expectancy_, poor_and_happy.Economy__GDP_per_Capita_,color="red",label="Unhappy Countries")



plt.legend()

plt.xlabel("Life Expectancy")

plt.ylabel("Economy")

plt.title("Freedom-Corruption")

plt.show()