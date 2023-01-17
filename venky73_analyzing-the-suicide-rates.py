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
df = pd.read_csv("../input/master.csv")
df.sample(5)
df.isna().sum()
first_obs = df[["country","year","sex","age","suicides/100k pop"]]
first_obs.head()
print("Min : ",first_obs.year.min())

print("Max : ",first_obs.year.max())
len(first_obs.year.unique())
def decade_mapping(data):

    if 1987<= data <= 1996:

        return "1987-1996"

    elif 1997<= data <= 2006:

        return "1997-2006"

    else:

        return "2007-2016"

first_obs.year = first_obs.year.apply(decade_mapping)
first_obs.sample()
plt.figure(figsize=(10,5))

sns.barplot(x = "age", y = "suicides/100k pop", hue = "sex",data = first_obs.groupby(["age","sex"]).sum().reset_index()).set_title("Age vs Suicides")

plt.xticks(rotation = 90)
first_obs.groupby(["year","sex"]).sum().reset_index()
plt.figure(figsize=(10,5))

sns.barplot(x = "year", y = "suicides/100k pop", hue = "sex",data = first_obs.groupby(["year","sex"]).sum().reset_index()).set_title("Decades vs Suicides")
sns.barplot(x = "sex", y = "suicides/100k pop", data = first_obs.groupby("sex").sum().reset_index()).set_title("Gender wise Suicides")
country_sucides = first_obs.groupby("country").sum().reset_index()

country_sucides.head()
plt.figure(figsize=(10,5))

best_10 = country_sucides.sort_values(by = "suicides/100k pop",ascending= True)[:10]

sns.barplot(x = "country", y = "suicides/100k pop", data = best_10).set_title("countries with less suicides")

plt.xticks(rotation = 90)
plt.figure(figsize=(10,5))

best_10 = country_sucides.sort_values(by = "suicides/100k pop",ascending= False)[:10]

sns.barplot(x = "country", y = "suicides/100k pop", data = best_10).set_title("Countries with most suicides")

plt.xticks(rotation = 90)
recent = first_obs[first_obs.year =="2007-2016"].groupby("country").sum().reset_index()

recent.head()
plt.figure(figsize=(10,5))

recent_best_10 = recent.sort_values(by = "suicides/100k pop")[:10]

sns.barplot(x = "country", y = "suicides/100k pop", data = recent_best_10).set_title("Countries with less suicides in 2007-2016")

plt.xticks(rotation = 90)
plt.figure(figsize=(10,5))

recent_bad_10 = recent.sort_values(by = "suicides/100k pop",ascending=False)[:10]

sns.barplot(x = "country", y = "suicides/100k pop", data = recent_bad_10).set_title("Countries with most suicides in 2007-2016")

plt.xticks(rotation = 90)
zone_assess = first_obs.groupby(["country","year"]).sum().reset_index()

zone_assess.head()
#countries having data of three decades

three_gen = zone_assess.country.value_counts().reset_index(name = "count")

three_gen.columns = ["country", "counts"]

three_gen_countries = three_gen[three_gen.counts == 3].country.tolist()
nations = three_gen_countries

years = zone_assess.year.unique()

green_zones = []

danger_zones = []

for country in nations:

    s_year1 = float(zone_assess[(zone_assess.country == country) & (zone_assess.year == "1987-1996")]["suicides/100k pop"])

    s_year2 = float(zone_assess[(zone_assess.country == country) & (zone_assess.year == "1997-2006")]["suicides/100k pop"])

    s_year3 = float(zone_assess[(zone_assess.country == country) & (zone_assess.year == "2007-2016")]["suicides/100k pop"])

    if s_year1 <= s_year2 <= s_year3:

        danger_zones.append(country)

    if s_year1 >= s_year2 >= s_year3:

        green_zones.append(country)

        
plt.figure(figsize=(18,8))

sns.barplot(x = "country", y = "suicides/100k pop", hue = "year",data = zone_assess[zone_assess.country.isin(green_zones)]).set_title("Decreasing Suicide Rate")

plt.xticks(rotation = 90)
plt.figure(figsize=(18,8))

sns.barplot(x = "country", y = "suicides/100k pop", hue = "year",data = zone_assess[zone_assess.country.isin(danger_zones)]).set_title("Increasing Suicide Rate")

plt.xticks(rotation = 90)