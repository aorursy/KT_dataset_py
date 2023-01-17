# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')







world = pd.read_csv('../input/countries-of-the-world/countries of the world.csv')
#SEE IF NaN VALUES EXISTS

print(world.isna().any())



print(world.info())

print(world.head())

print(world.dtypes)

print(world.describe())





world["Literacy (%)"] = world["Literacy (%)"].str.replace("," , ".").astype("float64")

world["Net migration"] = world["Net migration"].str.replace("," , ".").astype("float64")

world["Birthrate"] = world["Birthrate"].str.replace("," , ".").astype("float64")

world["Deathrate"] = world["Deathrate"].str.replace("," , ".").astype("float64")

world["Agriculture"] = world["Agriculture"].str.replace("," , ".").astype("float64")

world["Industry"] = world["Industry"].str.replace("," , ".").astype("float64")

world["Service"] = world["Service"].str.replace("," , ".").astype("float64")







world.fillna(0, inplace=True)









print(world.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median())







#BAR GRAPH OF TOP 30 COUNTRIES WITH HIGHEST POPULATIONS

mostPop30Data = world.sort_values("Population", ascending = False).head(30)



plt.figure(figsize = (10, 5))

sns.barplot(x = mostPop30Data["Country"], y = mostPop30Data["Population"])

plt.xticks(rotation = 90)

plt.title("Top 30 Countries with the Highest Populations")

plt.xlabel("Countries")

plt.ylabel("Population (in billion)")

plt.show()









#SIDE BAR GRAPH OF TOP 30 COUNTRIES WITH LARGEST AREA

largeArea30Data = world.sort_values("Area (sq. mi.)", ascending = False).head(30)



plt.figure(figsize = (5, 10))

sns.barplot(x = largeArea30Data["Area (sq. mi.)"], y = largeArea30Data["Country"])

plt.title("Top 30 Countries with the Largest Area")

plt.xlabel("Area")

plt.ylabel("Countries")

plt.show()











#POINT GRAPH OF B/D RATIO FOR TOP 50 COUNTRIES WITH HIGH LITERACY

highLitData = world.sort_values("Literacy (%)", ascending = False).head(50)



plt.figure(figsize = (10, 5))

sns.pointplot(x = highLitData["Country"], y = highLitData["Birthrate"], color = "green", alpha = 0.8)

sns.pointplot(x = highLitData["Country"], y = highLitData["Deathrate"], color = "red", alpha = 0.6)

plt.text(1, 32, "Birthrate", color = "green", fontsize = 14)

plt.text(1, 29, "Deathrate", color = "red", fontsize = 14)

plt.xticks(rotation = 90)

plt.title("Birth and Death Ratios of 50 Countries of Highest Literacy")

plt.xlabel("Countries")

plt.ylabel("Ratios (%)")

plt.grid()

plt.show()







#POINT GRAPH OF B/D RATIO FOR TOP 50 COUNTRIES WITH LOW LITERACY

lowLitData = world[world["Literacy (%)"] != 0].sort_values("Literacy (%)", ascending = True).head(50)



plt.figure(figsize = (10, 5))

sns.pointplot(x = lowLitData["Country"], y = lowLitData["Birthrate"], color = "green", alpha = 0.8)

sns.pointplot(x = lowLitData["Country"], y = lowLitData["Deathrate"], color = "red", alpha = 0.6)

plt.text(1, 5, "Birthrate", color = "green", fontsize = 14)

plt.text(1, 1, "Deathrate", color = "red", fontsize = 14)

plt.xticks(rotation = 90)

plt.title("Birth and Death Ratios of 50 Countries of Lowest Literacy")

plt.xlabel("Countries")

plt.ylabel("Ratios (%)")

plt.grid()

plt.show()















print(world["Region"].value_counts())

#BOX PLOT OF GDP VS REGION

sns.boxplot(x="Region",y="GDP ($ per capita)",data=world,width=0.7,palette="Set3",fliersize=5)

plt.xticks(rotation=90)

plt.title("GDP BY REGİON",color="black")















#DOT PLOT OF GDP VS BIRTHRATE

plt.figure(figsize = (10, 5))

sns.lmplot(x = "GDP ($ per capita)", y = "Birthrate", data = world)

plt.xlabel("GDP")

plt.ylabel("Birthrate")

plt.title("GDP vs Birthrate")

plt.show()

















#BAR GRAPH OF GDP VS COUNTRY

fig, ax = plt.subplots(figsize=(16,6))

top_gdp_countries = world.sort_values('GDP ($ per capita)',ascending=False).head(20)

mean = pd.DataFrame({'Country':['World mean'], 'GDP ($ per capita)':[world['GDP ($ per capita)'].mean()]})

gdps = pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']],mean],ignore_index=True)

sns.barplot(x='Country', y='GDP ($ per capita)', data=gdps, palette='Set1')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

plt.xticks(rotation=90)

plt.show()







#BAE GRAPH OF REGIONAL AVERAGE GDP PER CAPITA

fig = plt.figure(figsize=(12, 4))

world.groupby('Region')['GDP ($ per capita)'].mean().sort_values().plot(kind='bar')

plt.title('Regional Average GDP per Capita')

plt.xlabel("Region")

plt.ylabel('Average GDP per Capita')

plt.show()











#BAR GRAPH OF NUMBER OF COUNTRIES BY REGION

region = world.Region.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=region.index,y=region.values)

plt.xticks(rotation=45)

plt.ylabel('Number of countries')

plt.xlabel('Region')

plt.title('Number of Countries by REGİON',color = 'black',fontsize=20)



#PIE GRAPH OF CONTINENTS DISTRIBUTION

explode = (0, 0.1, 0, 0,0,0,0)

sizes=[15,10,25,5,30,5,10]

labels="ASIA","EASTERN EUROPE","NORTHERN AFRICA","OCEANIA","WESTERN EUROPE","SUB-SAHARAN AFRICA","NORTHERN AMERICA"

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels,explode=explode, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.show()







#BOX PLOT OF LITERACY

world.boxplot(column = ["Literacy (%)"])