import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

import warnings 

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/Chicago_Crimes_2012_to_2017.csv",error_bad_lines=False,warn_bad_lines=False)
data.info()
data_samples = data.sample(n=200000,random_state = 42)
data_samples.index = np.arange(0,len(data_samples))
data_samples.isna().sum()
data_samples.dropna(axis=0,inplace=True)
data_samples.drop(["Unnamed: 0","Case Number","Block","IUCR","FBI Code","X Coordinate","Y Coordinate"],axis=1,inplace=True)
data_samples.head()
LocDescList = data_samples["Location Description"].unique()

LocDescList2 = data_samples["Location Description"]

LocDescRatio = []



counted = Counter(LocDescList2) 

mostcommon = counted.most_common(10)

x,y = zip(*mostcommon)

x,y = list(x),list(y)

# %% 

plt.figure(figsize = (20,8))

ax = sns.barplot(x = x , y = y, palette = sns.cubehelix_palette(len(x)))

plt.xticks(rotation= 15)

plt.xlabel("Location Description")

plt.ylabel("Crimes")

plt.title("Most common 10 crime places ")

plt.show()
PrimaryType = data_samples["Primary Type"]

PrimaryTypeCounted = Counter(PrimaryType)

MostCommonPT = PrimaryTypeCounted.most_common(10)

x,y = zip(*MostCommonPT)

x,y  = list(x),list(y)



# %% 



plt.figure(figsize=(15,8))

ax = sns.barplot(x=x, y=y,palette = "deep")

plt.xticks(rotation= 15)

plt.xlabel("Primary Type")

plt.ylabel("Crimes")

plt.show()

Arrest = data_samples["Arrest"]

CountedArrest = Counter(Arrest)

MostCommonCA = CountedArrest.most_common()

x,y = zip(*MostCommonCA)

x,y = list(x),list(y)

x[1] = "Arrested"

x[0] = "Not Arrested"

# %% 

plt.figure(figsize=(10,4))

sns.barplot(x = x , y = y , palette = "dark")

plt.xlabel("Arresting")

plt.ylabel("Frequency")

plt.show()
Crimes = data.iloc[:, 3: ]

Crimes.head()
Crimes.index = Crimes.Date

Crimes.drop("Date",axis=1,inplace=True)



Crimes.index = pd.to_datetime(Crimes.index)
CrimeCounts = pd.DataFrame(Crimes.groupby("Primary Type").size().sort_values(ascending=False).rename("counter").reset_index())
CrimeCounts.head()
f, ax = plt.subplots(figsize=(6, 15))



sns.set_color_codes("pastel")

sns.barplot(x="counter", y="Primary Type", data=CrimeCounts.iloc[:10, :],

            label="Total", palette="Blues_d")



ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(ylabel="Type",

       xlabel="Crimes")

sns.despine(left=True, bottom=True)



plt.show()
crimes2012 = Crimes.loc["2012"]

crimes2013 = Crimes.loc["2013"]

crimes2014 = Crimes.loc["2014"]

crimes2015 = Crimes.loc["2015"]

crimes2016 = Crimes.loc["2016"]
crimes2012 = pd.DataFrame(crimes2012)

crimes2013 = pd.DataFrame(crimes2013)

crimes2014 = pd.DataFrame(crimes2014)

crimes2015 = pd.DataFrame(crimes2015)

crimes2016 = pd.DataFrame(crimes2016)
theft_2012 = pd.DataFrame(crimes2012[crimes2012['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])

theft_2013 = pd.DataFrame(crimes2013[crimes2013['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])

theft_2014 = pd.DataFrame(crimes2014[crimes2014['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])

theft_2015 = pd.DataFrame(crimes2015[crimes2015['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])

theft_2016 = pd.DataFrame(crimes2016[crimes2016['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])
grouper2012 = theft_2012.groupby([pd.TimeGrouper('M'), 'Primary Type'])

grouper2013 = theft_2013.groupby([pd.TimeGrouper('M'), 'Primary Type'])

grouper2014 = theft_2014.groupby([pd.TimeGrouper('M'), 'Primary Type'])

grouper2015 = theft_2015.groupby([pd.TimeGrouper('M'), 'Primary Type'])

grouper2016 = theft_2016.groupby([pd.TimeGrouper('M'), 'Primary Type'])
data_2012 = grouper2012['Primary Type'].count().unstack()

data_2013 = grouper2013['Primary Type'].count().unstack()

data_2014 = grouper2014['Primary Type'].count().unstack()

data_2015 = grouper2015['Primary Type'].count().unstack()

data_2016 = grouper2016['Primary Type'].count().unstack()
DataAll = pd.DataFrame()

DataAll=DataAll.append([data_2012,data_2013,data_2014,data_2015,data_2016])
DataAll.plot()

plt.show()
Crimes2 = pd.DataFrame(Crimes[Crimes['Location Description'].isin(['APARTMENT','RESIDENCE', 'STREET', 'SIDEWALK','OTHER','ALLEY','RESTAURANT','GAS STATION'])])

Crimes2.head()
plt.figure(figsize=(8,10))

Crimes2.groupby([Crimes2['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of crimes by Location')

plt.ylabel('Crime Location')

plt.xlabel('Number of crimes')

plt.show()
import folium



MapOfChicago = folium.Map(location=[41.881832, -87.623177],

                         zoom_start=11)
Crimes = Crimes.dropna()

Crimes = Crimes.drop(columns=['Block', 'IUCR','Domestic', 'Beat', 'District', 'Ward','X Coordinate', 'Y Coordinate','Updated On', 'FBI Code'], axis = 1)
Crimes = Crimes[Crimes["Primary Type"] == "THEFT"]

locations = Crimes.groupby('Community Area').first()

new_locations = locations.loc[:, ['Latitude', 'Longitude', 'Location Description', 'Arrest']]
for i in range(len(new_locations)):

    lat = new_locations.iloc[i][0]

    long = new_locations.iloc[i][1]

    popup_text = """Community Index : {}<br>

                Arrest : {}<br>

                Location Description : {}<br>"""

    popup_text = popup_text.format(new_locations.index[i],

                               new_locations.iloc[i][-1],

                               new_locations.iloc[i][-2]

                               )

    folium.CircleMarker(location = [lat, long], popup= popup_text, fill = True).add_to(MapOfChicago)

MapOfChicago