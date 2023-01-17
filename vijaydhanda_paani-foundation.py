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



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import xticks
import pandas as pd

ListOfTalukas = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv")

MarkingSystem = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/MarkingSystem.csv")

StateLevelWinners = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv")

VillagesSupportedByDonationsWaterCup2019 = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/VillagesSupportedByDonationsWaterCup2019.csv")
## Checking Shape and exploring data

print(ListOfTalukas.shape)

ListOfTalukas.head()
ListOfTalukas = ListOfTalukas.set_index('Unnamed: 0')

ListOfTalukas.columns


for col in ListOfTalukas.select_dtypes(include=['object']).columns:

    print(col," has " , len(ListOfTalukas[col].unique()),"Unique values are ---",ListOfTalukas[col].unique())
# Checking missing values

missing_null_values = round(100*(ListOfTalukas.isnull().sum()/len(ListOfTalukas.index)), 2).sort_values(ascending = False)

print(missing_null_values)
## Checking Shape and exploring data

print(MarkingSystem.shape)

MarkingSystem.head()
for col in MarkingSystem.select_dtypes(include=['object']).columns:

    print(col," has " , len(MarkingSystem[col].unique()),"Unique values are ---",MarkingSystem[col].unique())
MarkingSystem = MarkingSystem.set_index('Sr. No.')

MarkingSystem.columns
# Checking missing values

missing_null_values = round(100*(MarkingSystem.isnull().sum()/len(MarkingSystem.index)), 2).sort_values(ascending = False)

print(missing_null_values)
## Checking Shape and exploring data

print(StateLevelWinners.shape)

StateLevelWinners.head()

for col in StateLevelWinners.select_dtypes(include=['object']).columns:

    print(col," has " , len(StateLevelWinners[col].unique()),"Unique values are ---",StateLevelWinners[col].unique())
## Checking Shape and exploring data

print(VillagesSupportedByDonationsWaterCup2019.shape)

VillagesSupportedByDonationsWaterCup2019.head()
# Set Sr No as index 

VillagesSupportedByDonationsWaterCup2019.set_index('Sr. No.',inplace=True)

VillagesSupportedByDonationsWaterCup2019.columns
for col in VillagesSupportedByDonationsWaterCup2019.select_dtypes(include=['object']).columns:

    print(col," has " , len(VillagesSupportedByDonationsWaterCup2019[col].unique()),"Unique values are ---",VillagesSupportedByDonationsWaterCup2019[col].unique())
# Plotting count plot of yearwise participation

plt.figure(figsize=[10,5])

new_districtwise = ListOfTalukas.drop(columns='District').drop_duplicates()

plot = sns.countplot(x='Region',hue='Year', data=new_districtwise)

xticks(rotation = 90)

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.5,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.show()



#Index(['Region', 'District', 'Taluka', 'Year'], dtype='object')
# Plotting count plot of yearwise participation

plt.figure(figsize=[10,5])

new_districtwise = ListOfTalukas.drop(columns='Taluka').drop_duplicates()

plot = sns.countplot(x='Region',hue='Year', data=new_districtwise)

xticks(rotation = 90)

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.5,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.show()
# Plotting count plot of yearwise participation

plt.figure(figsize=[10,5])

new_districtwise = ListOfTalukas.drop(columns=['Taluka','Region']).drop_duplicates()

plot = sns.countplot(x='Year', data=new_districtwise)

xticks(rotation = 90)

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.5,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.show()
# Plotting count plot of yearwise participation

plt.figure(figsize=[10,5])

new_districtwise = ListOfTalukas.drop(columns=['District','Region']).drop_duplicates()

plot = sns.countplot(x='Year', data=new_districtwise)

sns.despine(left=True,bottom=True)

xticks(rotation = 90)

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.5,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.show()
total_districts=36

total_talukas = 358



# Number of Districts participated per year

new_districtwise = ListOfTalukas.drop(columns=['Taluka','Region']).drop_duplicates()

new_districtwise = new_districtwise['Year'].value_counts().rename_axis('Year').reset_index(name='Number of District')

new_districtwise['percent_dist'] = round((new_districtwise['Number of District']/36)*100,2)



# Number of Talukas participated per year

new_districtwise2 = ListOfTalukas.drop(columns=['District','Region']).drop_duplicates()

new_districtwise2 = new_districtwise2['Year'].value_counts().rename_axis('Year').reset_index(name='Number of Talukas')

new_districtwise2['percent_dist'] = round((new_districtwise2['Number of Talukas']/358)*100,2)

new_districtwise2.head()
fig, ax = plt.subplots(figsize=(12,12))

MarkingSystem.set_index('Component').plot.pie(y='Maximum Marks',legend=False, ax=ax)
sns.set(style='darkgrid')

sns.lineplot(y='percent_dist', x='Year', data=new_districtwise,marker='o',color='darkblue')

plt.xticks([2016,2017,2018,2019],['2016','2017','2018','2019'])

plt.show()
sns.set(style='darkgrid')

sns.lineplot(y='percent_dist', x='Year', data=new_districtwise2,marker='o',color='darkblue' )

plt.xticks([2016,2017,2018,2019],['2016','2017','2018','2019'])

plt.show()