import pandas as pd

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

print(df.shape)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# droping duplicate value along entire datastet

df.drop_duplicates(inplace =True,keep = "first")
df.columns = ["sno","firstupdate","state","country","lastupdate","confirmed","deaths","recovered"]

df[["confirmed","deaths","recovered"]] = df[["confirmed","deaths","recovered"]].astype(int)
### Let's see how many days recoreds this dataser has 

print("Starting From : ",df.lastupdate.min())

print("Till Now      : ",df.lastupdate.max())

print("This dataset as Approx 80 days of record world wide")
print(df.isnull().sum())

sns.heatmap(df.isnull(),yticklabels = False)

print("State feature has 3595 missing values")
df.country = [i.lower() for i in df.country]

#df.country.unique()

df.country = df.country.replace("mainland china","china")
df.state.fillna("unknown",inplace = True,axis = 0)

#df.head()
max_death_per_country = pd.DataFrame(df.groupby("country")["deaths"].max())

top_ten_death_rates = max_death_per_country.sort_values(by = "deaths",ascending = False).head(10)

top_ten_death_rates
#lets visualize it graphically 

countries = ["ITALY","CHINA","SPAIN","IRAN","FRANCE","UK","NETHERLANDS","US","GERMANY","BELGIUM"]

sns.barplot(top_ten_death_rates["deaths"],countries,orient = "h",alpha = 1)

plt.xlabel("Maximum death rates ")

plt.ylabel("Top ten countries")
max_confirmedcases_per_country = pd.DataFrame(df.groupby("country")["confirmed"].max())

top_ten_confirmed_countries = max_confirmedcases_per_country.sort_values(by = "confirmed",ascending = False).head(10)

top_ten_confirmed_countries
#lets visualize it graphically 

countries = ["ITALY","CHINA","SPAIN","GERMANY","US","IRAN","FRANCE","SWITZERLAND","SOUTH KOREA","UK"]

sns.barplot(top_ten_confirmed_countries["confirmed"],countries,orient = "h",alpha = 1)

plt.xlabel("Maximum confirmed rates ")

plt.ylabel("Top ten countries")
max_recovered_per_country = pd.DataFrame(df.groupby("country")["recovered"].max())

top_ten_recovered_countries = max_recovered_per_country.sort_values(by = "recovered",ascending = False).head(10)

top_ten_recovered_countries
countries  = ["CHINA","IRAN","ITALY","SPAIN","SOUTH KOREA","FRANCE","GERMANY","OTHERS","BELGIUM","JAPAN"]

sns.barplot(top_ten_recovered_countries["recovered"],countries ,orient = "h")

plt.xlabel("Maximum recovery rates ")

plt.ylabel("Top ten countries")
countries_df = df.groupby('country')[['confirmed','deaths','recovered']].max().sum().reset_index()

world_wide   = pd.DataFrame(countries_df) 

world_wide.rename(columns = {"index":"Cases",0:"Counts"},inplace =True)

print("Worl wide Cases  :")

world_wide
# using pie chart for visualiztion of different cases
plt.pie(world_wide["Counts"],labels = world_wide["Cases"],shadow = True,autopct = "%0.f%%",explode = [0.1,0.2,0.1],colors = ["orange","red","Green"])

plt.tight_layout()

plt.show()

#Getting active cases 

df["activecases"] = df["confirmed"] - (df["deaths"] + df["recovered"])
df.head()
world_wide_active_cases = df.groupby(["country"])["activecases"].max().sum()

print("Total Number of world_wide_active_cases are :",world_wide_active_cases)

world_wide_active_case = pd.Series(world_wide_active_cases)
world_wide = world_wide.append({"Cases":"active","Counts":278874},ignore_index = True)
plt.pie(world_wide["Counts"],labels = world_wide["Cases"],shadow = True,autopct = "%0.f%%",explode = [0,0,0,0.2])

plt.tight_layout()

plt.show()
idf = df[df["country"]=="india"].iloc[:,:]

idf.shape
idf.head()

idf[["confirmed","deaths","recovered","activecases"]] = idf[["confirmed","deaths","recovered","activecases"]].astype(int)
tot_deaths  = idf.deaths.max()

tot_confirm = idf.confirmed.max()

tot_recover = idf.recovered.max()

tot_active  = idf.activecases.max()

print("Total deaths cases   in india :",tot_deaths)

print("Total confirmed cases in india :",tot_confirm)

print("Total recovered cases in india :",tot_recover)

print("Total active cases    in india :",tot_active)
sns.barplot([10,536,40,486],["Death","Confirmed","recovered","active"],orient = "h")

plt.ylabel("Cases")

plt.xlabel("Number of People")