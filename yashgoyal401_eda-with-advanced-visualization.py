import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)

raw = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

raw_2 = pd.read_csv("../input/coronavirusdataset/route.csv")

raw_1 = pd.read_csv("../input/coronavirusdataset/time.csv")
raw.head()
raw.columns

raw.info()

raw.isnull().sum()

sns.heatmap(raw.isnull(),cmap="viridis")

sns.countplot(data=raw,y="sex")

plt.title("vitims distribution by their GENDER")

sns.countplot(data=raw,y="country")

plt.title("Affected countries")

sns.countplot(data=raw,y="infection_reason")

plt.title("Region")
sns.countplot(x=raw["state"].loc[(raw["infection_reason"]=="contact with patient")])

plt.title("State of victims who got affected by contact")

sns.countplot(x=raw["state"].loc[(raw["sex"]=="male")])

plt.title("State of male victims")

sns.countplot(x=raw["state"].loc[(raw["sex"]=="female")])

plt.title("State of female victims")

## Converting into DateTime format

raw[['confirmed_date', 'released_date',"deceased_date"]] = raw[['confirmed_date', 'released_date',"deceased_date"]].apply(pd.to_datetime)



raw["time_to_release_since_confirmed"] = (raw["released_date"] - raw["confirmed_date"]).dt.days



sns.countplot(data=raw,y="time_to_release_since_confirmed",order=raw["time_to_release_since_confirmed"].value_counts()[:20].index)

plt.title("Time taken to release(in days)")



raw["time_to_DEATH_since_confirmed"] = (raw["deceased_date"] - raw["confirmed_date"]).dt.days

sns.countplot(data=raw,x="time_to_DEATH_since_confirmed",order=raw["time_to_DEATH_since_confirmed"].value_counts().index)

## Converting birth year from str to int

raw["birth_year"] = raw["birth_year"].fillna(0.0).astype(int)

## Replacing 0.0 with nan

raw["birth_year"] = raw["birth_year"].map(lambda x: x if x>0 else np.nan)

## Calculating age

raw["age"] = 2020 - raw["birth_year"]

raw["age"].unique()

sns.countplot(data=raw,y="age",order=raw["age"].value_counts()[:15].index)



sns.kdeplot(data=raw["age"],shade=True)
raw["state_by_gender"] = raw["state"] + "_" + raw["sex"]

sns.countplot(y="state_by_gender",data=raw)

## Dataframe of Dead victims

Dead = raw[raw["state"]=="deceased"]

sns.kdeplot(data=Dead["age"],shade=True,label="Dead")

sns.kdeplot(data=raw["age"],shade=True,label="all people's age")

plt.title("Died_people_age_distribution Vs all people's age")

## Dead male victims

male_dead = Dead[Dead["sex"]=="male"]

## Dead female victims

female_dead = Dead[Dead["sex"]=="female"]

sns.kdeplot(data=male_dead["age"],shade=True,label="male")

sns.kdeplot(data=female_dead["age"],shade=True,label="female")

plt.title("Dead victims age distribution by age")

sns.distplot(a=male_dead['age'], label="Men", kde=False)

sns.distplot(a=female_dead['age'], label="Women", kde=False)

plt.title("Age distribution of the deceased by sex")

plt.legend()

sns.countplot(data=Dead,x="sex")

plt.title("Dead victims by gender")

## Creating age group

def age_buckets(x): 

    if x < 18: 

        return '0-17'

    elif x < 30:

        return '18-29'

    elif x < 40: 

        return '30-39' 

    elif x < 50: 

        return '40-49' 

    elif x < 60: 

        return '50-59' 

    elif x < 70:

        return '60-69' 

    elif x >=70: 

        return '70+' 

    else: return 'other'

raw["age_range"] = raw["age"].apply(age_buckets)

raw["age_range"].unique()









sns.countplot(data=raw,y="age_range")

plt.xlim(0,100)

plt.title("Age-group")

sns.countplot(y="age_range",data=raw,hue="state_by_gender")

facet = sns.FacetGrid(raw,hue="state_by_gender",aspect=4)

facet.map(sns.kdeplot,"age",shade=True)

facet.add_legend()

plt.title("Age distribuyion of state_by_gender")

facet = sns.FacetGrid(raw,hue="sex",aspect=4)

facet.map(sns.kdeplot,"age",shade=True)

facet.add_legend()

plt.title("Age distribuyion by sex")


facet = sns.FacetGrid(Dead,hue="country",aspect=4)

facet.map(sns.kdeplot,"age",shade=True)

facet.add_legend()

plt.title("age distribution by country")
## Data frame for released victims

released = raw[raw["state"]=="released"]

## Dataframe for Isolated victims

isolated = raw[raw["state"]=="isolated"]

sns.kdeplot(data = Dead["age"],label="dead",shade=True)

sns.kdeplot(data = released["age"],label="released",shade=True)

sns.kdeplot(data = isolated["age"],label="isolated",shade=True)

plt.xlim(0,100)

plt.title("age distribution by state")

## Days taken to Dead or release since confirmed

raw["duration_since_confirmed"] = raw[["time_to_release_since_confirmed", "time_to_DEATH_since_confirmed"]].min(axis=1)

sns.boxplot(x="state",y="duration_since_confirmed",order=['released','deceased'],data=raw)

### Route data set

raw_2.columns

raw_2.info()

raw_2.shape

sns.heatmap(raw_2.isnull())
## Basemap

import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')

for lat, lon,city in zip(raw_2['latitude'], raw_2['longitude'],raw_2['city']):

    folium.CircleMarker([lat, lon],radius=7,color='blue',popup =('City: ' + str(city) + '<br>'),fill_color='red',fill_opacity=0.7 ).add_to(southkorea_map)

display(southkorea_map)
sns.countplot(data=raw_2,y="city",order=raw_2["city"].value_counts()[:30].index)

plt.title("Affected cities")

sns.countplot(data=raw_2,y="province",order=raw_2["province"].value_counts().index)

plt.title("Affected province")
sns.countplot(data=raw_2,y="visit",order=raw_2["visit"].value_counts().index)

plt.title("Dangerous places")
## Clustering

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist



raw_2.columns

lat_lon = raw_2.drop(['id', 'date', 'province', 'city', 'visit'],axis=1)



k = list(range(2,15))

TWSS = []

for i in k:

        kmeans = KMeans(n_clusters = i)

        kmeans.fit(lat_lon)

        WSS = []

        for j in range(i):

            WSS.append(sum(cdist(lat_lon.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,lat_lon.shape[1]),"euclidean")))

        TWSS.append(sum(WSS)) 

# Scree plot 

plt.plot(k,TWSS, 'ro-')

plt.xlabel("No_of_Clusters")

plt.ylabel("total_within_SS")

plt.xticks(k)    
model = KMeans(n_clusters=4)

model.fit(lat_lon)

model.labels_

cc = model.cluster_centers_

mm = pd.Series(model.labels_)

lat_lon["group"] = mm



## Plot scatter by cluster / color, and centroids

colors = ["red", "green", "blue","black"]

lat_lon['color'] = lat_lon['group'].map(lambda p: colors[p])

lat_lon.plot(kind="scatter", x='latitude', y='longitude',figsize=(10,8),c = lat_lon['color'])

plt.scatter(cc[:, 0], cc[:, 1], c='pink', s=500, alpha=0.5, marker="*")

plt.title("areas which are mostly affected(by using lattitude and longitude)")

sns.scatterplot(x="date",y="acc_test",data=raw_1)

plt.xticks(rotation=90)

plt.title("acc_test")

sns.scatterplot(x="date",y="acc_confirmed",data=raw_1)

plt.xticks(rotation=90)

plt.title("acc_confirmed")
plt.plot(raw_1.date,raw_1.acc_test,zorder=1,label="acc_test")

plt.plot(raw_1.date,raw_1.acc_confirmed,zorder=1,color="orange")

plt.xticks(rotation=90)

plt.title("Tests done Vs positive tests")

plt.plot(raw_1.date,raw_1.new_test)

plt.plot(raw_1.date,raw_1.new_confirmed,color="orange")

plt.xticks(rotation=90)

plt.title("Tested done Vs positive tests")

##### Dtaframe for total number of vitims by date and their id

daily_count = raw.groupby('confirmed_date').id.count()

## Cumullative sum of victims

accumulated_count = daily_count.cumsum()

accumulated_count = pd.DataFrame(accumulated_count)

accumulated_count["ds"] = accumulated_count.index

accumulated_count = accumulated_count.reset_index(drop=True)

accumulated_count.columns =['y', 'ds']

accumulated_count = accumulated_count[["ds","y"]]

## Forecasting by using PROPHET

from fbprophet import Prophet

m = Prophet()

m.fit(accumulated_count)



future = m.make_future_dataframe(periods=30)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
import pandas as pd

Case = pd.read_csv("../input/coronavirusdataset/Case.csv")

PatientInfo = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

PatientRoute = pd.read_csv("../input/coronavirusdataset/PatientRoute.csv")

Region = pd.read_csv("../input/coronavirusdataset/Region.csv")

SearchTrend = pd.read_csv("../input/coronavirusdataset/SearchTrend.csv")

Time = pd.read_csv("../input/coronavirusdataset/Time.csv")

TimeAge = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")

TimeGender = pd.read_csv("../input/coronavirusdataset/TimeGender.csv")

TimeProvince = pd.read_csv("../input/coronavirusdataset/TimeProvince.csv")

Weather = pd.read_csv("../input/coronavirusdataset/Weather.csv")