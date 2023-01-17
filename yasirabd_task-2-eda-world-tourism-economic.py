# import library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn import preprocessing

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")
world_indicators = pd.read_excel("../input/world-tourism-economic/Dataset  dictionary.xlsx", sheet_name='World Indicators_csv')

dictionary = pd.read_excel("../input/world-tourism-economic/Dataset  dictionary.xlsx", sheet_name='Dictionary', names=['Feature Name', 'Description'])

categorized_data = pd.read_excel("../input/world-tourism-economic/Dataset  dictionary.xlsx", sheet_name='categorized_data', usecols=lambda x: 'Unnamed' not in x)
world_indicators.head()
dictionary.head()
categorized_data.head()
df = world_indicators.copy()
df.info()
df.isnull().sum().sort_values(ascending=False)
for col in df.columns:

    print("%s unique count: %d" % (col, df[col].nunique()))

    print(df[col].unique(), '\n')
df.describe(include='object')
df['Business Tax Rate'] = df['Business Tax Rate'].str.replace('%','').astype('float64')

df['GDP'] = df['GDP'].str.replace('$','')

df['GDP'] = df['GDP'].str.replace(',','', regex=True).astype('float64')

df['Health Exp/Capita'] = df['Health Exp/Capita'].str.replace('$','')

df['Health Exp/Capita'] = df['Health Exp/Capita'].str.replace(',','', regex=True).astype('float64')

df['Tourism Inbound'] = df['Tourism Inbound'].str.replace('$','')

df['Tourism Inbound'] = df['Tourism Inbound'].str.replace(',','', regex=True).astype('float64')

df['Tourism Outbound'] = df['Tourism Outbound'].str.replace('$','')

df['Tourism Outbound'] = df['Tourism Outbound'].str.replace(',','', regex=True).astype('float64')
df['Year'] = df['Year'].dt.strftime('%Y')
le_country = preprocessing.LabelEncoder()

le_region = preprocessing.LabelEncoder()

le_year = preprocessing.LabelEncoder()



df['Country'] = le_country.fit_transform(df['Country'])

df['Region'] = le_region.fit_transform(df['Region'])

df['Year'] = le_year.fit_transform(df['Year'])
mice_impute = IterativeImputer(min_value=0)

data_impute = mice_impute.fit_transform(df)



# convert to dataframe

df_complete = pd.DataFrame(data_impute, columns=df.columns)
# pembulatan dan perbaiki data tipe atribut

cols_round3 = ['Birth Rate', 'Health Exp % GDP', 'Infant Mortality Rate', 'Lending Interest', 'Population 0-14',

               'Population 15-64', 'Population 65+', 'Population Urban']

cols_round1 = ['Business Tax Rate', 'Internet Usage', 'Mobile Phone Usage']

cols_round0 = ['CO2 Emissions', 'Country', 'Days to Start Business', 'Ease of Business', 'Energy Usage', 

               'GDP', 'Health Exp/Capita', 'Hours to do Tax', 'Life Expectancy Female', 'Life Expectancy Male', 

               'Population Total', 'Region', 'Tourism Inbound', 'Tourism Outbound', 'Year']



df_complete[cols_round3] = df_complete[cols_round3].round(3).astype('float64')

df_complete[cols_round1] = df_complete[cols_round1].round(1).astype('float64')

df_complete[cols_round0] = df_complete[cols_round0].round(0).astype('int64')
df_complete.info()
df_viz = df_complete.copy()
df_viz['Country'] = le_country.inverse_transform(df_viz['Country'])

df_viz['Region'] = le_region.inverse_transform(df_viz['Region'])

df_viz['Year'] = le_year.inverse_transform(df_viz['Year'])
df_viz.head()
df_viz.describe().T
df_viz.describe(include='object')
corr = df_viz.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr)
corr['GDP'].sort_values(ascending=False)
plt.figure(figsize=(7,5), dpi=80)

sns.barplot(x=df_viz.Region, y=df_viz.GDP)
plt.figure(figsize=(7,5), dpi=80)

sns.barplot(x=df_viz.Region, y=df_viz['Tourism Inbound'])
plt.figure(figsize=(7,5), dpi=80)

sns.barplot(x=df_viz.Region, y=df_viz['Tourism Outbound'])
# ambil GDP terbesar dari tiap Country

df_gdp_country = df_viz.groupby('Country', group_keys=False).apply(lambda x: x.loc[x.GDP.idxmax()])



# df_gdp_country

top40 = df_gdp_country['GDP'].sort_values(ascending=False)[:40]

bot40 = df_gdp_country['GDP'].sort_values()[:40]



plt.figure(figsize=(20,8), dpi=80)

top = sns.barplot(x=top40, y=top40.index, log=True)



plt.figure(figsize=(20,8), dpi=80)

bot = sns.barplot(x=bot40, y=bot40.index, log=True)
# ambil Tourism Inbound terbesar dari tiap Country

df_ti_country = df_viz.groupby('Country', group_keys=False).apply(lambda x: x.loc[x['Tourism Inbound'].idxmax()])



# df_gdp_country

top40 = df_ti_country['Tourism Inbound'].sort_values(ascending=False)[:40]

bot40 = df_ti_country['Tourism Inbound'].sort_values()[:40]



plt.figure(figsize=(20,8), dpi=80)

sns.barplot(x=top40, y=top40.index, log=True)



plt.figure(figsize=(20,8), dpi=80)

sns.barplot(x=bot40, y=bot40.index, log=True)
# ambil Tourism Outbond terbesar dari tiap Country

df_to_country = df_viz.groupby('Country', group_keys=False).apply(lambda x: x.loc[x['Tourism Outbound'].idxmax()])



# df_gdp_country

top40 = df_to_country['Tourism Outbound'].sort_values(ascending=False)[:40]

bot40 = df_to_country['Tourism Outbound'].sort_values()[:40]



plt.figure(figsize=(20,8), dpi=80)

top_bp = sns.barplot(x=top40, y=top40.index, log=True)



plt.figure(figsize=(20,8), dpi=80)

bot_bp = sns.barplot(x=bot40, y=bot40.index, log=True)
# ambil Energy Usage terbesar dari tiap Country

df_eu_country = df_viz.groupby('Country', group_keys=False).apply(lambda x: x.loc[x['Energy Usage'].idxmax()])



# df_gdp_country

top40 = df_eu_country['Energy Usage'].sort_values(ascending=False)[:40]

bot40 = df_eu_country['Energy Usage'].sort_values()[:40]



plt.figure(figsize=(20,8), dpi=80)

top_bp = sns.barplot(x=top40, y=top40.index, log=True)



plt.figure(figsize=(20,8), dpi=80)

bot_bp = sns.barplot(x=bot40, y=bot40.index, log=True)
# ambil Energy Usage terbesar dari tiap Country

df_ce_country = df_viz.groupby('Country', group_keys=False).apply(lambda x: x.loc[x['CO2 Emissions'].idxmax()])



# df_gdp_country

top40 = df_ce_country['CO2 Emissions'].sort_values(ascending=False)[:40]

bot40 = df_ce_country['CO2 Emissions'].sort_values()[:40]



plt.figure(figsize=(20,8), dpi=80)

top_bp = sns.barplot(x=top40, y=top40.index, log=True)



plt.figure(figsize=(20,8), dpi=80)

bot_bp = sns.barplot(x=bot40, y=bot40.index, log=True)
sns.lmplot(x='Tourism Inbound', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
sns.lmplot(x='Tourism Outbound', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
sns.lmplot(x='Energy Usage', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
sns.lmplot(x='CO2 Emissions', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
sns.lmplot(x='Infant Mortality Rate', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
sns.lmplot(x='Birth Rate', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
sns.lmplot(x='Population 0-14', y='GDP', hue='Region', data=df_viz, height=6, aspect=1.3)
# get data Indonesia

df_indonesia = df_viz[df_viz.Country.isin(['Indonesia'])]

df_indonesia
fig, axes = plt.subplots(3,1,figsize=(8,15), dpi=80)

sns.barplot(x=df_indonesia.Year, y=df_indonesia['GDP'], ax=axes[0], log=True)

sns.barplot(x=df_indonesia.Year, y=df_indonesia['Tourism Inbound'], ax=axes[1], log=True)

sns.barplot(x=df_indonesia.Year, y=df_indonesia['Tourism Outbound'], ax=axes[2], log=True)
fig, axes = plt.subplots(figsize=(8,5), dpi=80)

sns.barplot(x=df_indonesia.Year, y=df_indonesia['CO2 Emissions'],log=True)
fig, axes = plt.subplots(figsize=(8,5), dpi=80)

sns.barplot(x=df_indonesia.Year, y=df_indonesia['Energy Usage'],log=True)
# get data Indonesia dan malaysia

df_idmy = df_viz[df_viz.Country.isin(['Indonesia', 'Malaysia'])]

df_idmy.head()
plt.figure(figsize=(12,7))

sns.barplot(x='Year', y='GDP', hue='Country', data=df_idmy, log=True)
plt.figure(figsize=(12,7))

sns.barplot(x='Year', y='Tourism Inbound', hue='Country', data=df_idmy, log=True)
plt.figure(figsize=(12,7))

sns.barplot(x='Year', y='Tourism Outbound', hue='Country', data=df_idmy, log=True)
plt.figure(figsize=(12,7))

sns.barplot(x='Year', y='CO2 Emissions', hue='Country', data=df_idmy, log=True)