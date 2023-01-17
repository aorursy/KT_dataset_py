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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
xdf = pd.read_csv('/kaggle/input/icc-odi-batting-figures-1971-to-2019/ICC ODI Batting 2589.csv', engine = 'python')

xdf.head()
split1 = xdf.Player.str.split('([()])',expand=True)

split1 = pd.DataFrame(split1)

split1.columns = ['Name','z','Country','c','v','b','n','m','l']

split1 = split1[['Name', 'Country']]

xdf['Name'] = split1.Name

xdf['Country'] = split1.Country
# xdf['Name'] = xdf['Name'].replace(u'\xa0', u'')

# xdf.head()
xdf = xdf[(xdf[['Inns', 'NO', 'Runs', 'HS', 'Ave','BF', 'SR', '100', '50', '0']] != '-').all(axis=1)]

xdf.head()
xdf['Country'].unique()
xdf[xdf['Country'].str.contains('1')]
xdf[xdf['Country'].str.contains('3')]
xdf['Country'] = xdf['Country'].replace('1','PAK',regex=True)

xdf['Country'] = xdf['Country'].replace('3','UAE',regex=True)
xdf['Country'].unique()
ctry_name = { 'INDIA' : 'India', 'Asia/ICC/INDIA' : 'India', 'Asia/INDIA' : 'India', 

              'Asia/ICC/SL' : 'Sri Lanka', 'SL' : 'Sri Lanka', 'Asia/SL' : 'Sri Lanka', 

              'AUS/ICC' : 'Australia', 'AUS' : 'Australia', 

              'Asia/PAK' : 'Pakistan', 'PAK' : 'Pakistan', 'Asia/ICC/PAK': 'Pakistan', 3 : 'Pakistan',

              'Afr/ICC/SA' : 'South Africa', 'Afr/SA' : 'South Africa', 'SA' : 'South Africa','ICC/SA': 'South Africa',

              'ICC/WI' : 'West Indies', 'WI' : 'West Indies', 

              'NZ' : 'New Zeland', 'ICC/NZ' : 'New Zeland',

              'BDESH' : 'Bangladesh', 'Asia/BDESH' : 'Bangladesh', 

              'ENG' : 'England',

              'ENG/ICC' : 'England',

              'Afr/ZIM' : 'Zimbwabe', 'ZIM' : 'Zimbwabe', 

              'Afr/KENYA' : 'Kenya', 'KENYA' : 'Kenya',

              'IRE' : 'Ireland',

              'AFG' : 'Afghanistan',

              'SCOT' : 'Scotland', 

              'CAN' : 'Canada',

              'NL' : 'Netherlands',

              'HKG' : 'Hongkong', 

              'BMUDA' : 'Bermuda',

              'PNG' : 'Papua New Guinea',

              'OMAN' : 'Oman',

              'NAM' : 'Namibia', 

              'NEPAL' : 'Nepal',

              'USA' : 'USA', 

              'UAE' : 'UAE', 1 : 'UAE',

              'EAf' : 'East Africa',

              'ENG/IRE' : 'ENG/IRE', 

              'AUS/NZ' : 'AUS/NZ', 

              'AUS/SA' : 'AUS/SA', 

              'USA/WI' : 'USA/WI', 

              'ENG/PNG' : 'ENG/PNG', 

              'ENG/SCOT' : 'ENG/SCOT', 

              'CAN/WI' : 'CAN/WI', 

              'HKG/NZ' : 'HKG/NZ',

              'NL/SA' : 'NL/SA',

              'SA/USA' : 'SA/USA'

             }
ctnt_name = { 'India': 'Asia', 

              'Sri Lanka': 'Asia', 

              'Australia' : 'Australia', 

              'Pakistan': 'Asia',

              'South Africa' : 'Africa',

              'West Indies': 'America', 

              'New Zeland': 'Australia',

              'Bangladesh': 'Asia', 

              'England': 'Europe',

              'Zimbwabe': 'Africa', 

              'Kenya': 'Africa',

              'Ireland': 'Europe',

              'Afghanistan': 'Asia',

              'Scotland': 'Europe', 

              'Canada': 'America',

              'Netherlands': 'Europe',

              'Hongkong': 'Asia', 

              'Bermuda': 'America',

              'Papua New Guinea': 'Australia',

              'Oman': 'Asia',

              'Namibia' : 'Africa', 

              'Nepal': 'Asia',

              'USA': 'America', 

              'UAE': 'Asia',

              'East Africa': 'Africa',

              'ENG/IRE': 'Europe', 

              'AUS/NZ': 'Australia', 

              'AUS/SA': 'Mixed', 

              'USA/WI': 'America', 

              'ENG/PNG': 'Mixed', 

              'ENG/SCOT': 'Europe', 

              'CAN/WI': 'America', 

              'HKG/NZ': 'Mixed',

              'NL/SA': 'Mixed',

              'SA/USA': 'Mixed'

             }
xdf['Nation'] = xdf['Country'].map(ctry_name)

xdf['Continent'] = xdf['Nation'].map(ctnt_name)
xdf.isnull().sum()
xdf = xdf[['Name', 'Nation', 'Continent', 'Span', 'Mat', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', ]]

xdf.head()


fig = plt.figure()

sns.set(style='darkgrid')

sns.set(rc={'figure.figsize':(20.7, 15.27)})

sns.countplot(x = 'Nation', data = xdf, order = xdf['Nation'].value_counts().index)

plt.xticks(rotation = 90)

plt.show()
fig = plt.figure()

sns.set(style='darkgrid')

sns.set(rc={'figure.figsize':(16.7, 12.27)})

sns.countplot(x = 'Continent', data = xdf, order = xdf['Continent'].value_counts().index)

plt.xticks(rotation = 90)

plt.show()
xdf.iloc[0:1, :1]
