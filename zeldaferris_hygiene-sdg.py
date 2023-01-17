import pandas as pd

import matplotlib

import numpy

import os
os.getcwd()
import sys

!{sys.executable} -m pip install matplotlib
df= pd.read_csv("../input/washdataset/washdash-download.csv")
import os

import matplotlib.pyplot as plt

import pandas
# Cleaning Dataframe
#df = pd.read_csv('Raw_Data/washdash-download.csv')

df_hygiene = df[df['Service Type'] == 'Hygiene']

df_hygiene_total = df_hygiene[df_hygiene['Residence Type'] == 'total']

#df_hygiene_total.to_csv('Raw_Data/hygiene_data.csv')

df_hygiene_total = df_hygiene_total.drop(['Population', 'Type', 'Residence Type', 'Service Type'], axis =1)

df_hygiene_total.head()
df_hygiene_spread = pd.pivot_table(df_hygiene_total, values = 'Coverage', index=['Region','Year'], columns = 'Service level').reset_index()

df_hygiene_spread.head()
df_hygiene['Region'].unique()
# Annalyize Trends
df_csa = df_hygiene[df_hygiene['Region'] == 'Central and Southern Asia']

df_lac = df_hygiene[df_hygiene['Region'] == 'Latin America and the Caribbean']

df_nawa = df_hygiene[df_hygiene['Region'] == 'Northern Africa and Western Asia']

df_ssa = df_hygiene[df_hygiene['Region'] == 'Sub-Saharan Africa']
from matplotlib import pyplot as plt
sdg_regions = [df_csa, df_lac, df_nawa, df_ssa]

sdg_region_title = ['Central and Southern Asia',

                   'Latin America and the Caribbean', 

                   'Northern Africa and Western Asia', 

                   'Sub-Saharan Africa']

index = 0

for region in sdg_regions: 

        title = sdg_region_title[index]



        plt.plot(region['Year'], region['Basic service'], color = 'lightseagreen', marker = '^', label = "Basic")

        plt.plot(region['Year'], region['Limited service'], color = 'royalblue', marker = '^', label = 'Limited')

        plt.plot(region['Year'], region['No handwashing facility'], color = 'mediumpurple', marker ='.', label = 'No handwashing facility')



        plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

        plt.xticks(range(2000,2017,2))

        plt.xlabel('Year')

        plt.ylabel('Percent Coverage')

        plt.title(title)

        

        image_title = sdg_region_title[index].replace(" ", "") + "_coverage.png" 

        plt.savefig('Hygiene/' + image_title)

        plt.show()

        index += 1
from matplotlib import pyplot

sdg_regions = [df_csa, df_lac, df_nawa, df_ssa]

sdg_region_title = ['Central and Southern Asia',

                   'Latin America and the Caribbean', 

                   'Northern Africa and Western Asia', 

                   'Sub-Saharan Africa']

index = 0

for region in sdg_regions: 

    title = sdg_region_title[index]

    x = region['Year']

    y1 = region['Basic service']

    y2 = region['Limited service']

    y3 = region['No handwashing facility']

    colors=['#FF9070','#FCD17A','#FDF3BF' ]

        

    y = np.vstack([y2,y3])



    labels = ["No handwashing facility", "Limited service", "Basic service"]

    fig, ax = plt.subplots()

    ax.stackplot(x,y2,y3, labels=labels, colors=colors)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.xticks(range(2000,2017,16))

    plt.xlabel('Year')

    plt.ylabel('Percent Coverage')

    plt.title(title)

    plt.show()

    index += 1