import sys

import pandas as pd

import numpy

import matplotlib

df= pd.read_csv("../input/washdataset/washdash-download.csv")
df.head()
df['Service Type']
waterdf = df[df['Service Type'] == 'Drinking water']
waterdf.head()
waterdf_total = waterdf[waterdf['Residence Type'] == 'total']
waterdf_total.head()
waterdf_total = waterdf_total.drop(['Population', 'Type', 'Residence Type', 'Service Type'], axis =1)

waterdf_total.head()
waterdf_spread = pd.pivot_table(waterdf_total, values = 'Coverage', index=['Region','Year'], columns = 'Service level').reset_index()

waterdf_spread.head()
waterdf_spread.loc[waterdf_spread['Basic service'].isnull(), 'Basic service'] = waterdf_spread['At least basic']

waterdf_spread = waterdf_spread.drop(['At least basic'], axis = 1)

waterdf=waterdf_spread

waterdf.head()
waterdf['Region'].unique()
waterdf_aus = waterdf[waterdf['Region'] == 'Australia and New Zealand']

waterdf_csa = waterdf[waterdf['Region'] == 'Central and Southern Asia']

waterdf_lac = waterdf[waterdf['Region'] == 'Latin America and the Caribbean']

waterdf_nawa = waterdf[waterdf['Region'] == 'Northern Africa and Western Asia']

waterdf_oce = waterdf[waterdf['Region'] == 'Oceania']

waterdf_ssa = waterdf[waterdf['Region'] == 'Sub-Saharan Africa']
import matplotlib.pyplot as plt
waterdf_aus.head()
waterdf_aus
from matplotlib import pyplot

sdg_regions = [waterdf_aus, waterdf_csa, waterdf_lac, waterdf_nawa, waterdf_oce, waterdf_ssa]

sdg_region_title = ['Australia and New Zealand', 

                   'Central and Southern Asia',

                   'Latin America and the Caribbean', 

                   'Northern Africa and Western Asia', 

                   'Oceania', 

                   'Sub-Saharan Africa']

index = 0

for region in sdg_regions: 

        title = sdg_region_title[index]

        

        plt.plot(region['Year'], region['Surface water'], fillstyle = 'full', color = 'yellowgreen', marker = '.', label = "Surface water")

        plt.plot(region['Year'], region['Basic service'], fillstyle = 'full',color = 'lightseagreen', marker = '^', label = "Basic")

        plt.plot(region['Year'], region['Limited service'], fillstyle = 'full',color = 'royalblue', marker = '^', label = 'Limited')

        plt.plot(region['Year'], region['Unimproved'],fillstyle = 'full', color = 'mediumpurple', marker ='.', label = 'Unimproved')

        plt.plot(region['Year'], region['Safely managed service'], fillstyle = 'full',color = 'black', marker = '.', label = "Safely managed service")

        

        plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

        plt.xticks(range(2000,2017,16))

        plt.xlabel('Year')

        plt.ylabel('Percent Coverage')

        plt.title(title)

        plt.show()

        

        image_title =  'Plot of Water Service Levels Across the world'

        

        index += 1
import numpy as np
from matplotlib import pyplot

sdg_regions = [waterdf_aus,waterdf_nawa, waterdf_oce]

sdg_region_title = ['Australia and New Zealand', 

                   'Northern Africa and Western Asia', 

                   'Oceania']

index = 0

for region in sdg_regions: 

    title = sdg_region_title[index]

    x = region['Year']

    y2 = region['Basic service']

    y3 = region['Limited service']

    y4 = region['Unimproved']

    y5 = region['Surface water']

    colors=['#FF9070','#FCD17A','#FDF3BF', '#4CC4E7','#4D8CBF']

        

    y = np.vstack([y2,y3,y4,y5])



    labels = ["Basic service", "Limited service", "Unimproved", "Surface water"]

    fig, ax = plt.subplots()

    ax.stackplot(x,y2,y3,y4,y5, labels=labels, colors=colors)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.xticks(range(2000,2017,16))

    plt.xlabel('Year')

    plt.ylabel('Percent Coverage')

    plt.title(title)

    plt.show()

    index += 1
from matplotlib import pyplot

sdg_regions = [waterdf_csa, waterdf_lac, waterdf_ssa]

sdg_region_title = [

                   'Central and Southern Asia',

                   'Latin America and the Caribbean',

                   'Sub-Saharan Africa']

index = 0

for region in sdg_regions: 

    title = sdg_region_title[index]

    x = region['Year']

    y1 = region['Safely managed service']

    y2 = region['Basic service']

    y3 = region['Limited service']

    y4 = region['Unimproved']

    y5 = region['Surface water']



    y = np.vstack([y1,y2,y3,y4,y5])



    labels = ["Safely managed service ", "Basic service", "Limited service", "Unimproved", 

        "Surface water"]    



    colors=['#FF9070','#FCD17A','#FDF3BF', '#4CC4E7','#4D8CBF']

    fig, ax = plt.subplots()

    ax.stackplot(x,y1,y2,y3,y4,y5, labels=labels, colors=colors)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.xticks(range(2000,2017,16))

    plt.xlabel('Year')

    plt.ylabel('Percent Coverage')

    plt.title(title)

    plt.show()

    index += 1
plt.style.use('dark_background')



waterdf.groupby('Year')['Safely managed service'].mean().plot(kind='bar', figsize=(10, 6))

plt.suptitle('Mean of the Safely Managed Service by Year', fontsize=14)

plt.ylabel('Safely Managed Service Population')

plt.xlabel('Year')
waterdf.groupby('Year')['Unimproved'].median().plot(kind='bar', figsize=(10, 6))

plt.suptitle('Mean of the Unimproved Water Service by Year', fontsize=14)

plt.ylabel('Unimproved Population')

plt.xlabel('Year')
waterdf.groupby('Year')['Safely managed service', 'Basic service', 'Limited service','Unimproved', 'Surface water'].median().plot(kind='bar', figsize=(10, 6))

plt.suptitle('Medians of different water services by Year', fontsize=14)

plt.ylabel('Unimproved Population')

plt.xlabel('Year')
waterdf.columns=['Region','Year','Basic Water','Limited Water','Safely Managed Water','Surface water','Unimproved water']
## Transforming Hygiene and Sanitation data to merge them to Drinking Water and look at the trends 
import pandas as pd

hygiene = pd.read_csv("../input/hygienecleaned/hygiene_cleaned.csv")

hygiene=hygiene.drop(['Unnamed: 0'], axis=1)

hygiene.columns=['Region', 'Year', 'Basic Hygiene','Limited Hygiene','No handwashing facility']

sanitation=pd.read_csv("../input/sanitationcleaned/sanitation_data(new).csv")

sanitation=sanitation.drop(['Unnamed: 0'], axis=1)

sanitation.columns=['Region', 'Year', 'Basic Sanitation','Limited Sanitation','Open Defecation','Safely Managed Sanitation', 'Unimproved Sanitation']



newdf=pd.merge(sanitation, waterdf, how='left', left_on=['Year','Region'], right_on = ['Year', 'Region'])
masterdf=pd.merge(newdf, hygiene, how='left', left_on=['Year', 'Region'],right_on=['Year', 'Region'])
masterdf.head()
masterdf.groupby('Year')['Basic Hygiene', 'Basic Water','Basic Sanitation'].median().plot(kind='bar', figsize=(10, 6))

plt.suptitle('Medians For Basic Hygiene, Basic Water Service and Basic Sanitation around the world', fontsize=14)

plt.ylabel('Percentage Coverage')

plt.xlabel('Year')
masterdf.groupby('Year')['Unimproved Sanitation','Unimproved water', 'No handwashing facility'].median().plot(kind='bar', figsize=(10, 6))

plt.suptitle('Medians For No Handwashing Facility and No water service around the world', fontsize=14)

plt.ylabel('Percentage Coverage')

plt.xlabel('Year')
masterdf.groupby('Year')['Limited Hygiene', 'Limited Water','Limited Sanitation'].median().plot(kind='bar', figsize=(10, 6))

plt.suptitle('Medians For Limited Hygiene , Limited Sanitation and Limited Water Service around the world', fontsize=14)

plt.ylabel('Percentage Coverage')

plt.xlabel('Year')
masterdf_aus = masterdf[masterdf['Region'] == 'Australia and New Zealand']

masterdf_csa = masterdf[masterdf['Region'] == 'Central and Southern Asia']

masterdf_lac = masterdf[masterdf['Region'] == 'Latin America and the Caribbean']

masterdf_nawa = masterdf[masterdf['Region'] == 'Northern Africa and Western Asia']

masterdf_oce = masterdf[masterdf['Region'] == 'Oceania']

masterdf_ssa = masterdf[masterdf['Region'] == 'Sub-Saharan Africa']
from matplotlib import pyplot

sdg_regions = [masterdf_aus, masterdf_csa, masterdf_lac,masterdf_nawa,masterdf_oce,masterdf_ssa]

sdg_region_title = ['Australia',

                   'Central and Southern Asia','Latin America and Caribbean','Northern Africa and Western Asia','Oceania',

                   'Sub-Saharan Africa']

index = 0

for region in sdg_regions: 

    title = sdg_region_title[index]

    region.groupby('Year')['Basic Hygiene', 'Basic Water','Basic Sanitation'].max().plot(kind='bar', figsize=(10, 6))

    plt.ylabel('Percentage Coverage')

    plt.title(title)

    plt.xlabel('Year')

    index += 1
#  Correlations provide evidence of association, not causation.

# Positive r values indicate positive association between the variables, 

# and negative r values indicate negative associations.



from pandas import DataFrame

import seaborn as sn





df = DataFrame(masterdf,columns=['Basic water','Basic Hygiene','Basic Sanitation','Limited Sanitation',

                           'Limited Hygiene','Limited Water', 'Open Defecation','Unimproved Sanitation','Limited Water','No handwashing facility','Safely Managed Water','Surface water'])



corrMatrix = df.corr()

sn.heatmap(corrMatrix, annot=True)