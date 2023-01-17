import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 
df=pd.read_csv("../input/time_series_2019-ncov-Confirmed.csv")

df=df.rename(columns={'Province/State':'State','Country/Region':'Country'})

df
x=df.copy()

ls=[]

for i in range(4,65):

    m=x.iloc[:,i].value_counts()

    if(m[0]>341):

        ls.append(i)

df.drop(df.columns[ls],axis=1,inplace=True)

df.shape
plt.barh(df['Country'],df['3/18/20'], color = "green")

plt.xlabel("Countries")

plt.xticks(fontsize = 10)

plt.yticks(fontsize = 3)

plt.ylabel("cases")

plt.title("Corona Cases")

plt.show()
x=list(set(df.columns)-set(['State','Country','Lat','Long']))

plt.xlabel("Cases")

plt.ylabel("Dates")

plt.title("Cases in Japan")

plt.plot(df.iloc[1,4:23],x)
df.dropna(axis=0,how='any',inplace=True)
df
af=pd.read_csv("../input/af_countries.csv")

asia=pd.read_csv("../input/as_countries.csv")

eu = pd.read_csv("../input/eu_countries.csv")

na = pd.read_csv("../input/na_countries.csv")

sa = pd.read_csv("../input/sa_countries.csv")

df=pd.read_csv("../input/time_series_2019-ncov-Confirmed.csv")

df=df.rename(columns={'Province/State':'State','Country/Region':'Country'})
as_countries = asia['name'].values.tolist()

eu_countries = eu['name'].values.tolist()

af_countries = af['name'].values.tolist()

na_countries = na['name'].values.tolist()

sa_countries = sa['name'].values.tolist()

date = "3/4/20"
cases = df[date]

countries = df["Country"]
ctr = 0

as_cases = 0

eu_cases = 0

af_cases = 0

na_cases = 0

sa_cases = 0

for i in countries:

    if(i in as_countries):

        as_cases+=cases[ctr]

    if(i in eu_countries):

        eu_cases+=cases[ctr]

    if(i in af_countries):

        af_cases+=cases[ctr]

    if(i in na_countries):

        na_cases+=cases[ctr]

    if(i in sa_countries):

        sa_cases+=cases[ctr]

    ctr+=1
cases=[as_cases, eu_cases, af_cases, na_cases, sa_cases]

continent = ['Asia', 'Europe', 'Africa', "North America", "South America"]

plt.barh(continent,cases,color="Green")

plt.show()