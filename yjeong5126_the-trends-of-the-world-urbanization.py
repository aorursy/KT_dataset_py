# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import geopandas as gpd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
# read the file of 'indicators'

data = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')
data = data.loc[data['IndicatorCode'].isin(['NY.GDP.PCAP.KD','EN.URB.LCTY.UR.ZS','SP.URB.TOTL.IN.ZS','EN.URB.LCTY.UR.ZS','EN.URB.MCTY.TL.ZS'])]
df = data.loc[(data['IndicatorCode']=='SP.URB.TOTL.IN.ZS')&(data['Year'].isin([1960,2010]))]

df = df.pivot_table('Value', ['CountryName', 'Year'], 'IndicatorName')

df.reset_index(inplace=True)

df['Year']=df['Year'].astype(int)

df.rename(columns={'CountryName':'country', 'Urban population (% of total)':'urban_percent','Year':'year'}, inplace=True)

df.head()
# read the file for the world map using GeoPandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
country_data = list(df['country'].unique())

country_geo = list(world['name'])

country_diff = [country for country in country_data if country not in country_geo]

country_diff
CountryChanged = pd.DataFrame(df['country'].replace({'Bahamas, The':'Bahamas','Bosnia and Herzegovina':'Bosnia and Herz.','Brunei Darussalam':'Brunei','Central African Republic':'Central African Rep.','Congo, Dem. Rep.':'Dem. Rep. Congo','Congo, Rep.':'Congo', "Cote d'Ivoire": "Côte d'Ivoire", 'Czech Republic': 'Czechia', 'Dominican Republic': 'Dominican Rep.', 'Egypt, Arab Rep.': 'Egypt', 'Equatorial Guinea': 'Eq. Guinea', 'Gambia, The': 'Gambia', 'Iran, Islamic Rep.': 'Iran', 'Korea, Dem. Rep.': 'North Korea', 'Korea, Rep.': 'South Korea', 'Kyrgyz Republic': 'Kyrgyzstan', 'Lao PDR': 'Laos', 'Macedonia, FYR': 'Macedonia', 'Mauritius': 'Mauritania', 'Russian Federation': 'Russia', 'Slovak Republic': 'Slovakia', 'Solomon Islands': 'Solomon Is.','South Sudan': 'S. Sudan', 'Syrian Arab Republic': 'Syria', 'United States': 'United States of America', 'Venezuela, RB': 'Venezuela', 'Yemen, Rep.': 'Yemen'}))

df['country'] =  CountryChanged
df1960 = df.loc[df['year'] == 1960]

df2010 = df.loc[df['year'] == 2010]
# combining two dataframe

mapped1960 = world.set_index('name').join(df1960.set_index('country')).reset_index()

mapped2010 = world.set_index('name').join(df2010.set_index('country')).reset_index()
#creating worldcolormap function

def Worldcolormap(VariableName, JointDF,TitleName):

    variable = VariableName

    fig,ax = plt.subplots(1, figsize=(15,15))

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="3%", pad=0.1)

    JointDF.dropna(subset=[VariableName]).plot(column=variable, cmap ='Blues',ax=ax,cax=cax, linewidth=0.8, edgecolors='0.8', legend=True)

    ax.set_title(TitleName, fontdict={'fontsize':20})

    ax.set_axis_off()
Worldcolormap('urban_percent', mapped1960, 'Urban Population (% of Total Population) in 1960')
Worldcolormap('urban_percent', mapped2010, 'Urban Population (% of Total Population) in 2010')
# GDP per capita for three groups of the countries

hic_gdpcapita =data[(data['CountryCode']=='OEC')&(data['IndicatorCode']=='NY.GDP.PCAP.KD')]

mic_gdpcapita =data[(data['CountryCode']=='MIC')&(data['IndicatorCode']=='NY.GDP.PCAP.KD')]

lic_gdpcapita =data[(data['CountryCode']=='LIC')&(data['IndicatorCode']=='NY.GDP.PCAP.KD')]
# Line Plot for GDP per capita over Time

width = 8

height = 7

plt.figure(figsize=(width,height))

    

plt.style.use('seaborn-darkgrid')

palette = plt.get_cmap('Set1')



plt.plot(hic_gdpcapita.Year, hic_gdpcapita.Value, marker='', color=palette(1), linewidth=1, alpha=0.9, label = 'High Income Countries')

plt.plot(mic_gdpcapita.Year, mic_gdpcapita.Value, marker='', color=palette(2), linewidth=1, alpha=0.9, label = 'Middle Income Countries')

plt.plot(lic_gdpcapita.Year, lic_gdpcapita.Value, marker='', color=palette(3), linewidth=1, alpha=0.9, label = 'Low Income Countries')



plt.legend(loc=2, ncol=1, fontsize=15)

plt.title('GDP per capita', loc='center', fontsize=15, fontweight=0)

plt.xlabel("Year", fontsize=15)

plt.ylabel("constant 2005 US$", fontsize=15)
# urban population (% of total population) for each income level

hic_city =data[(data['CountryCode']=='OEC')&(data['IndicatorCode']=='SP.URB.TOTL.IN.ZS')]

mic_city =data[(data['CountryCode']=='MIC')&(data['IndicatorCode']=='SP.URB.TOTL.IN.ZS')]

lic_city =data[(data['CountryCode']=='LIC')&(data['IndicatorCode']=='SP.URB.TOTL.IN.ZS')]
# population in urban agglomerations of more than 1 million (% of total population) for each income level

hic_large =data[(data['CountryCode']=='OEC')&(data['IndicatorCode']=='EN.URB.MCTY.TL.ZS')]

mic_large =data[(data['CountryCode']=='MIC')&(data['IndicatorCode']=='EN.URB.MCTY.TL.ZS')]

lic_large =data[(data['CountryCode']=='LIC')&(data['IndicatorCode']=='EN.URB.MCTY.TL.ZS')]
# population in the largest city (% of urban population) for each income level

hic_largest =data[(data['CountryCode']=='OEC')&(data['IndicatorCode']=='EN.URB.LCTY.UR.ZS')]

mic_largest =data[(data['CountryCode']=='MIC')&(data['IndicatorCode']=='EN.URB.LCTY.UR.ZS')]

lic_largest =data[(data['CountryCode']=='LIC')&(data['IndicatorCode']=='EN.URB.LCTY.UR.ZS')]
# Creating 'population in the largest city (% of total population)' for high income countries

hic_largest = hic_largest.rename(columns={'Value':'LargestUrban'})

hic_largest = pd.merge(hic_largest, hic_city, how='inner', on=['CountryName','CountryCode','Year'])

hic_largest['LargestTotal'] = (hic_largest['LargestUrban']*hic_largest['Value'])/100

hic_largest = hic_largest.drop(columns=['IndicatorName_x','IndicatorCode_x','IndicatorName_y','IndicatorCode_y','LargestUrban','Value'])

hic_largest = hic_largest.rename(columns={'LargestTotal':'Value'})



# Creating 'population in the largest city (% of total population)' for middle income countries

mic_largest = mic_largest.rename(columns={'Value':'LargestUrban'})

mic_largest = pd.merge(mic_largest, mic_city, how='inner', on=['CountryName','CountryCode','Year'])

mic_largest['LargestTotal'] = (mic_largest['LargestUrban']*mic_largest['Value'])/100

mic_largest = mic_largest.drop(columns=['IndicatorName_x','IndicatorCode_x','IndicatorName_y','IndicatorCode_y','LargestUrban','Value'])

mic_largest = mic_largest.rename(columns={'LargestTotal':'Value'})



# Creating 'population in the largest city (% of total population)' for low income countries

lic_largest = lic_largest.rename(columns={'Value':'LargestUrban'})

lic_largest = pd.merge(lic_largest, lic_city, how='inner', on=['CountryName','CountryCode','Year'])

lic_largest['LargestTotal'] = (lic_largest['LargestUrban']*lic_largest['Value'])/100

lic_largest = lic_largest.drop(columns=['IndicatorName_x','IndicatorCode_x','IndicatorName_y','IndicatorCode_y','LargestUrban','Value'])

lic_largest = lic_largest.rename(columns={'LargestTotal':'Value'})
def Urbantrends(TitleName, Highincome, Middleincome, Lowincome):

    

    plt.plot(Highincome.Year, Highincome.Value, marker='', color=palette(1), linewidth=1, alpha=0.9, label = 'High Income Countries')

    plt.plot(Middleincome.Year, Middleincome.Value, marker='', color=palette(2), linewidth=1, alpha=0.9, label = 'Middle Income Countries')

    plt.plot(Lowincome.Year, Lowincome.Value, marker='', color=palette(3), linewidth=1, alpha=0.9, label = 'Low Income Countries')

    plt.legend(loc=2, ncol=1, fontsize=10)

    plt.title(TitleName, loc='center', fontsize=15, fontweight=0)

    plt.xlabel("Year", fontsize=15)

    plt.ylabel("% of Total Population", fontsize=15)

    plt.ylim(0,100)
# Graphs for Urbanization

width = 18

height = 6

plt.figure(figsize=(width,height))

    

plt.style.use('seaborn-darkgrid')

palette = plt.get_cmap('Set1')

    

plt.subplot(131)

Urbantrends('Population in Any Cities', hic_city, mic_city, lic_city)

    

plt.subplot(132)

Urbantrends('Population in Large Cities', hic_large, mic_large, lic_large)



plt.subplot(133)

Urbantrends('Population in the Largest City', hic_largest, mic_largest, lic_largest)
def Urbantrends(TitleName, Largest, Large, Any):

    

    plt.plot(Largest.Year, Largest.Value, marker='', color=palette(1), linewidth=1, alpha=0.9, label = 'Population in the Largest City')

    plt.plot(Large.Year, Large.Value, marker='', color=palette(2), linewidth=1, alpha=0.9, label = 'Population in Large Cities')

    plt.plot(Any.Year, Any.Value, marker='', color=palette(3), linewidth=1, alpha=0.9, label = 'Population in Any Cities')

    plt.legend(loc=2, ncol=1, fontsize=10)

    plt.title(TitleName, loc='center', fontsize=15, fontweight=0)

    plt.xlabel("Year", fontsize=15)

    plt.ylabel("% of Total Population", fontsize=15)

    plt.ylim(0,100)
# Size

width = 18

height = 6

plt.figure(figsize=(width,height))

    

# Style

plt.style.use('seaborn-darkgrid')

    

# Create a color palette

palette = plt.get_cmap('Set1')

    

plt.subplot(131)

Urbantrends('High Income Countries', hic_largest, hic_large, hic_city)

    

plt.subplot(132)

Urbantrends('Middle Income Countries', mic_largest, mic_large, mic_city)



plt.subplot(133)

Urbantrends('Low Income Countries', lic_largest, lic_large, lic_city)
# high income countries

hic_city = hic_city.rename(columns={'Value':'pop_any'})

hic_large = hic_large.rename(columns={'Value':'pop_large'})

hic_city = pd.merge(hic_city, hic_large, how='inner', on=['CountryName','CountryCode','Year'])

hic_city['Medium or Small'] = hic_city['pop_any']-hic_city['pop_large']



hic_largest = hic_largest.rename(columns={'Value':'Largest'})

hic_large = pd.merge(hic_large, hic_largest, how='inner', on=['CountryName','CountryCode','Year'])

hic_large['Large'] = hic_large['pop_large']-hic_large['Largest']



hic_large = pd.merge(hic_large, hic_city, how='inner', on=['CountryName','CountryCode','Year'])

hic = hic_large[['CountryName', 'Year', 'Largest','Large', 'Medium or Small']]

hic = hic.loc[hic['Year'].isin([1960,1970,1980,1990,2000,2010])]



# middle income countries

mic_city = mic_city.rename(columns={'Value':'pop_any'})

mic_large = mic_large.rename(columns={'Value':'pop_large'})

mic_city = pd.merge(mic_city, mic_large, how='inner', on=['CountryName','CountryCode','Year'])

mic_city['Medium or Small'] = mic_city['pop_any']-mic_city['pop_large']



mic_largest = mic_largest.rename(columns={'Value':'Largest'})

mic_large = pd.merge(mic_large, mic_largest, how='inner', on=['CountryName','CountryCode','Year'])

mic_large['Large'] = mic_large['pop_large']-mic_large['Largest']



mic_large = pd.merge(mic_large, mic_city, how='inner', on=['CountryName','CountryCode','Year'])

mic = mic_large[['CountryName', 'Year', 'Largest','Large', 'Medium or Small']]

mic = mic.loc[mic['Year'].isin([1960,1970,1980,1990,2000,2010])]



# low income countries

lic_city = lic_city.rename(columns={'Value':'pop_any'})

lic_large = lic_large.rename(columns={'Value':'pop_large'})

lic_city = pd.merge(lic_city, lic_large, how='inner', on=['CountryName','CountryCode','Year'])

lic_city['Medium or Small'] = lic_city['pop_any']-lic_city['pop_large']



lic_largest = lic_largest.rename(columns={'Value':'Largest'})

lic_large = pd.merge(lic_large, lic_largest, how='inner', on=['CountryName','CountryCode','Year'])

lic_large['Large'] = lic_large['pop_large']-lic_large['Largest']



lic_large = pd.merge(lic_large, lic_city, how='inner', on=['CountryName','CountryCode','Year'])

lic = lic_large[['CountryName', 'Year', 'Largest','Large', 'Medium or Small']]

lic = lic.loc[lic['Year'].isin([1960,1970,1980,1990,2000,2010])]
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.set_size_inches(18,6)



hic.plot.bar(x='Year', ax=ax1).legend(loc='upper left')

ax1.set_ylim((0,50))

ax1.set_title("High Income Countries", fontsize = 15)

ax1.set_ylabel("% of Total Population", fontsize = 15)



mic.plot.bar(x='Year', ax=ax2).legend(loc='upper left')

ax2.set_ylim((0,50))

ax2.set_title("Middle Income Countries", fontsize = 15)



lic.plot.bar(x='Year', ax=ax3).legend(loc='upper left')

ax3.set_ylim((0,50))

ax3.set_title("Low Income Countries", fontsize = 15)