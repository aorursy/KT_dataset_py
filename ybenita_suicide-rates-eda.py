import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

from matplotlib import rcParams

%matplotlib inline
Suicide = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
Suicide.info()
Suicide.head()
print("Maximum number of samples per country:        ",Suicide.groupby('country')['country'].count().max(),Suicide.groupby('country')['country'].count().idxmax() )

print("Minimum number of samples per country:        ",Suicide.groupby('country')['country'].count().min(), Suicide.groupby('country')['country'].count().idxmin())

print("Total number of countries in the sample:      ",Suicide['country'].nunique())

print("Total number of unique rows (country + year) in the sample: ",Suicide['country-year'].nunique())
Suicide[Suicide['country'] == 'Mongolia']
import missingno

print(Suicide.isna().sum())

print("\n\nPrecentage of NULLs of the HDI for year: ", Suicide['HDI for year'].isna().sum()/Suicide.shape[0])

missingno.matrix(Suicide, figsize = (30,10))
nnull = Suicide[Suicide['HDI for year'].isna() == False]

nnull.shape[0]

rcParams['figure.figsize'] = 10,3

#sns.scatterplot(x="gdp_per_capita ($)", y="HDI for year", data=nnull)

sns.lmplot(x="gdp_per_capita ($)", y="HDI for year",data=nnull,logx=True,ci=None , scatter_kws={"s" : 70})
Suicide.describe()
sortSu = Suicide.groupby('country').mean()

sortSu['Country'] = sortSu.index

rcParams['figure.figsize'] = 10,3

sortSu = sortSu.sort_values('suicides/100k pop') 

dis = sns.barplot(x=sortSu['Country'][-20:], y=sortSu['suicides/100k pop'][-20:], palette="rocket")

plt.title('Average number of Suicide per Country - Top 20', fontsize=20)



rcParams['figure.figsize'] = 12,3



for item in dis.get_xticklabels():

    item.set_rotation(45)

    

dis.plot()
dis = sns.barplot(x=sortSu['Country'][0:20], y=sortSu['suicides/100k pop'][0:20], palette="rocket")

plt.title('Average number of Suicide per Country - bottom 20', fontsize=20)



rcParams['figure.figsize'] = 10,3



for item in dis.get_xticklabels():

    item.set_rotation(45)

    

dis.plot()
rcParams['figure.figsize'] = 8,3

x = sortSu[sortSu['suicides/100k pop'] > 16]

sns.distplot(x['suicides/100k pop'], bins=20)
pop = Suicide.groupby('year').mean()

pop['Year'] = pop.index

dis = sns.barplot(x=pop['Year'], y=pop['suicides/100k pop'], palette="rocket")

plt.title('Mean number of Suicide per Year', fontsize=20)



for item in dis.get_xticklabels():

    item.set_rotation(45)

rcParams['figure.figsize'] = 12,3    

dis.plot()

#pop['suicides/100k pop'].max()
gen = Suicide.groupby('generation').mean()

rcParams['figure.figsize'] = 12,3

x = ['Generation Z', 'Millenials', 'Generation X', 'Boomers' ,'Silent', 'G.I. Generation']

gen = gen.reindex(x)

gen['Gen'] = gen.index

dis = sns.barplot(x=gen['Gen'], y=gen['suicides/100k pop'], palette="rocket")

plt.title('Mean number of Suicide per generetion', fontsize=20)



for item in dis.get_xticklabels():

    item.set_rotation(45)

rcParams['figure.figsize'] = 10,3    

dis.plot()
dis = sns.barplot(x=gen['Gen'], y=gen['gdp_per_capita ($)'], palette="rocket")

plt.title('Mean size of GDP per capita per generetion', fontsize=20)



for item in dis.get_xticklabels():

    item.set_rotation(45)

rcParams['figure.figsize'] = 10,3    

dis.plot()
pop = Suicide.groupby('age').sum()

pop['Age'] = pop.index

tmp = ['5-14 years', '15-24 years', '25-34 years', '35-54 years' ,'55-74 years', '75+ years']

pop = pop.reindex(tmp)

dis = sns.barplot(x=pop['Age'], y=pop['suicides/100k pop'], color="blue")

plt.title('Age distribution across all countries', fontsize=20)

rcParams['figure.figsize'] = 10,5

for item in dis.get_xticklabels():

    item.set_rotation(45)

    

dis.plot()
print(pop['suicides/100k pop'])
g = Suicide.groupby('sex').sum()['suicides_no']

g = g.reset_index()

rcParams['figure.figsize'] = 5,5

dis = sns.barplot(x=g['sex'], y=g['suicides_no'], color="green")
gdp = Suicide.groupby('country').mean()

rcParams['figure.figsize'] = 15,3

f, axes = plt.subplots(1, 3)

sns.distplot(gdp['gdp_per_capita ($)'], bins=30,ax=axes[0]);

x = gdp[gdp['gdp_per_capita ($)'] < 9000 ]

sns.distplot(x['gdp_per_capita ($)'], bins=30,ax=axes[1]);

x1 = gdp[gdp['gdp_per_capita ($)'] > 9000 ]

sns.distplot(x1['gdp_per_capita ($)'], bins=30,ax=axes[2]);
Suicide['gdp'] = Suicide[' gdp_for_year ($) '].str.replace(',','')

Suicide['gdp'] = Suicide['gdp'].astype(int)

#f, axes = plt.subplots(2, 1)

rcParams['figure.figsize'] = 10,5

#old = Suicide[Suicide.age == "75+ years"]

sns.heatmap(Suicide.corr(),annot=True,linewidth = 0.5, cmap='coolwarm')

#sns.heatmap(Suicide.corr(method='spearman'),annot=True,linewidth = 0.5, cmap='coolwarm', ax=axes[1])
df = Suicide.groupby('country').mean()

#df = Suicide.groupby('year').mean()

rcParams['figure.figsize'] = 15,3

df = df[df['suicides/100k pop'] < 10]

sns.jointplot(x="gdp_per_capita ($)", y="suicides/100k pop", data=df)
rcParams['figure.figsize'] = 15,10

df = df[df["gdp_per_capita ($)"] < 5000 ]

sns.jointplot(x=df["gdp_per_capita ($)"], y=df["suicides/100k pop"], kind="hex", color="#4CB391")
rcParams['figure.figsize'] = 15,10

tmp = df[df["gdp_per_capita ($)"] < 9000 ]

sns.jointplot(x=tmp["gdp_per_capita ($)"], y=tmp["suicides/100k pop"], kind="hex", color="#4CB391")
df = Suicide[Suicide.age == "75+ years"]

rcParams['figure.figsize'] = 10,3

sns.scatterplot(x=df["gdp_per_capita ($)"], y=df["suicides/100k pop"], hue='sex', data=df)
df = Suicide[Suicide.age == "55-74 years"]

rcParams['figure.figsize'] = 10,3

sns.scatterplot(x="gdp_per_capita ($)", y="suicides/100k pop", hue='sex', data=df)
df = Suicide[Suicide.age == "35-54 years"]

rcParams['figure.figsize'] = 10,3

sns.scatterplot(x="gdp_per_capita ($)", y="suicides/100k pop", hue='sex', data=df)
df = Suicide[Suicide.age == "25-34 years"]

rcParams['figure.figsize'] = 10,3

sns.scatterplot(x="gdp_per_capita ($)", y="suicides/100k pop", hue='sex', data=df)
df = Suicide[Suicide.age == "15-24 years"]

rcParams['figure.figsize'] = 10,3

sns.scatterplot(x="gdp_per_capita ($)", y="suicides/100k pop", hue='sex', data=df)
df = Suicide[Suicide.age == "5-14 years"]

rcParams['figure.figsize'] = 10,3

sns.scatterplot(x="gdp_per_capita ($)", y="suicides/100k pop", hue='sex', data=df)
df =Suicide[(Suicide.age == "5-14 years")]

g = df.groupby('sex').sum()['suicides_no']

g = g.reset_index()

rcParams['figure.figsize'] = 5,5

dis = sns.barplot(x=g['sex'], y=g['suicides_no'], color="green")
rcParams['figure.figsize'] = 12,8

f, axes = plt.subplots(2,3)

df = Suicide.groupby('year').mean()

df['Year']  = df.index

sns.lineplot(x="Year", y='gdp_per_capita ($)', data=df, ax=axes[0,0])

sns.lineplot(x="Year", y="suicides/100k pop" , data=df, ax=axes[1,0])

sns.lineplot(x="Year", y="gdp" , data=df, ax=axes[0,2])

df = Suicide.groupby('year').count()

df.reset_index()

df['Year'] = df.index

df['Country'] = (df['country']/12).astype(int)

sns.lineplot(x="Year", y="Country", data=df, ax=axes[0,1])

df = Suicide.groupby('year').sum()

df.reset_index()

df['Year'] = df.index

sns.lineplot(x="Year", y="population" , data=df, ax=axes[1,1])
rcParams['figure.figsize'] = 12,8

f, axes = plt.subplots(2,3)

df = Suicide.groupby('year').mean()

df['Year']  = df.index

df = df[(df['Year'] < 2010) & (df['Year'] > 2000)]

sns.lineplot(x="Year", y='gdp_per_capita ($)', data=df, ax=axes[0,0])

sns.lineplot(x="Year", y="suicides/100k pop" , data=df, ax=axes[1,0])

sns.lineplot(x="Year", y="gdp" , data=df, ax=axes[0,2])

df = Suicide.groupby('year').count()

df.reset_index()

df['Year'] = df.index

df = df[(df['Year'] < 2010) & (df['Year'] > 2000)]

df['Country'] = (df['country']/12).astype(int)

sns.lineplot(x="Year", y="Country", data=df, ax=axes[0,1])

df = Suicide.groupby('year').sum()

df.reset_index()

df['Year'] = df.index

df = df[(df['Year'] < 2010) & (df['Year'] > 2000)]

sns.lineplot(x="Year", y="population" , data=df, ax=axes[1,1])
df2 = Suicide.groupby(['year','country','age']).mean()

df2 = df2.reset_index('country')

df2 = df2.reset_index('age')

tmp = df2[df2.index == 2001]

cset = set(tmp['country'])



for i in range(2002,2011):

  tmp = df2[df2.index == i]

  cset = cset & set(tmp['country'])

  

print("Number of coutries that apears in all years: ", len(cset))



df2 = df2[(df2.index > 1999) & (df2.index < 2011)]

df2 = df2[df2.country.isin(cset) ]
# ------------ Data Analysis --------

rcParams['figure.figsize'] = 10,10

f, axes = plt.subplots(2,1)

df7 = df2.reset_index('year')

df7 = df7.groupby(['year','country','age']).mean()

df7 = df7.reset_index('country')

df7 = df7.reset_index('age')

#df2 = df2[df2['age'] == "75+ years"]

df7['Year']  = df7.index

sns.lineplot(x="Year", y='gdp_per_capita ($)', data=df7, ax=axes[0])

sns.lineplot(x="Year", y="suicides/100k pop" ,data=df7, ax=axes[1])
f, axes = plt.subplots(2,1)

df = Suicide[(Suicide['country'] == 'Israel')]

sns.lineplot(x="year", y="suicides/100k pop" , hue='sex', data=df, ax=axes[0]).set_title("Trend analysis of Suicides in Israel")

sns.lineplot(x="year", y="gdp_per_capita ($)" , data=df, ax=axes[1]).set_title("Trend analysis of GDP per Capita in Israel")
f, axes = plt.subplots(2,1)

df = Suicide[(Suicide['country'] == 'United States')]

sns.lineplot(x="year", y="suicides/100k pop" , data=df, hue='sex',ax=axes[0]).set_title("Trend analysis of Suicides in United States")

sns.lineplot(x="year", y="gdp_per_capita ($)" , data=df, ax=axes[1]).set_title("Trend analysis of GDP per Capita in United States")
f, axes = plt.subplots(2,1)

df = Suicide[(Suicide['country'] == 'Finland') & (Suicide['year'] > 2000) & (Suicide['year'] < 2011)]

sns.lineplot(x="year", y="suicides/100k pop" , data=df, hue='sex',ax=axes[0]).set_title("Trend analysis of Suicides in Finland")

sns.lineplot(x="year", y="gdp_per_capita ($)" , data=df, ax=axes[1]).set_title("Trend analysis of GDP per Capita in Finland")
Gini = pd.read_csv('../input/gini-index/data_csv.csv', encoding = "ISO-8859-1")

Gini = Gini.rename(index=str, columns={"Country Name": "country", "Year": "year", "Value" : "gini_reported"})

Gini.head()
df1 = Gini.groupby(['year','country']).mean()

df1 = df1.reset_index('country')

df1['Year'] = df1.index

df1 = df1[df1.country.isin(cset) ]

df1['country'].nunique()

df1.head()
rcParams['figure.figsize'] = 20,5

df4 = df1.groupby('year').mean()

df4['Year'] = df4.index

dis = sns.barplot(x=df4['Year'], y=df4['gini_reported'], palette="rocket")

plt.title('Mean Gini Index per Year', fontsize=20)



for item in dis.get_xticklabels():

    item.set_rotation(45)

rcParams['figure.figsize'] = 10,3    

dis.plot()
df1.info()
df1.describe()
df1 = df1[(df1['Year'] > 1999) & (df1['Year'] < 2011)]

df2['gini_reproted'] = 0.0 

df2 = df2.reset_index()

for index, row in df2.iterrows():

  row_gini = df1.loc[( df1['Year'] == row['year']) & (df1['country'] == row['country'])]

  if row_gini['Year'].empty == False: 

    df2.at[index,'gini_reproted'] = row_gini['gini_reported']
df2  = df2[df2['gini_reproted'] != 0]
print(df2['country'].nunique())

df2.head()
#df3 = df2[df2['gini_reproted'] < 40]

sns.jointplot(x="gini_reproted", y="suicides/100k pop", data=df2)
rcParams['figure.figsize'] = 10,12

f, axes = plt.subplots(3,1)

df = df2[df2['country'] == 'Argentina']

sns.lineplot(x="year", y="suicides/100k pop" , data=df,ax=axes[0]).set_title("Trend analysis of Suicides in Argentina")

sns.lineplot(x="year", y="gdp_per_capita ($)" , data=df, ax=axes[1]).set_title("Trend analysis of GDP per Capita in Argentina")

sns.lineplot(x="year", y='gini_reproted' , data=df, ax=axes[2]).set_title("Trend analysis of Gini in Argentina")
rcParams['figure.figsize'] = 10,12

f, axes = plt.subplots(3,1)

df = df2[df2['country'] == 'United States']

sns.lineplot(x="year", y="suicides/100k pop" , data=df,ax=axes[0]).set_title("Trend analysis of Suicides in'United States")

sns.lineplot(x="year", y="gdp_per_capita ($)" , data=df, ax=axes[1]).set_title("Trend analysis of GDP per Capita in 'United States")

sns.lineplot(x="year", y='gini_reproted' , data=df, ax=axes[2]).set_title("Trend analysis of Gini in 'United States")
rcParams['figure.figsize'] = 10,10

f, axes = plt.subplots(3,1)

df3 = df2.groupby(['year','country']).mean()

df3 = df3.reset_index('country')

df3['Year']  = df3.index

#df3 = df3[df3['gini_reproted'] < 30]

sns.lineplot(x="Year", y='gini_reproted', data=df3, ax=axes[0])

sns.lineplot(x="Year", y="gdp_per_capita ($)"  , data=df3, ax=axes[1])

sns.lineplot(x="Year", y="suicides/100k pop" , data=df3, ax=axes[2])