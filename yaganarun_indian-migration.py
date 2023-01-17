# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go
FILE_PATH = '../input/indian-migration-history/IndianMigrationHistory.csv'

df = pd.read_csv(FILE_PATH)
df.info()
# Checking for null data 

1 - df.count() / len(df) 
# Checking for duplicated rows

df[df.duplicated()].count()
df.head()

rename_columns = {'1960 [1960]':'1960' ,'1970 [1970]':'1970','1980 [1980]':'1980','1990 [1990]':'1990',

                  '2000 [2000]':'2000','Migration by Gender Name': 'Gender','Country Dest Name':'Country'

                 }

df.rename(columns = rename_columns , inplace = True)
total_data = df[ df['Gender'] == 'Total'] # gender = Total

df = df.iloc[:26 , : ] # gender = M or F
migrated_population = df.groupby(['Country']).sum()

migrated_population['Total'] = migrated_population.sum(axis = 1)

migrated_population['Percent'] = round((migrated_population['Total'] / migrated_population['Total'].sum())*100 , 2)

migrated_population
def create_percent(data_frame , years = []):

    new_frame = pd.DataFrame()    

    for year in years:

        new_frame[year] = round( ( data_frame[year]/data_frame[year].sum() ) * 100 , 3)

    new_frame['country'] = data_frame.index

    return new_frame

migrated_pop_percnt = create_percent(migrated_population , ['1960' , '1970' , '1980' , '1990' , '2000'])
header = dict(values = ['country' , '1960' , '1970' , '1980' , '1990' , '2000'])

cells = dict(values = [ migrated_pop_percnt['country'] ,migrated_pop_percnt['1960'] ,migrated_pop_percnt['1970'],migrated_pop_percnt['1980'] ,

                                migrated_pop_percnt['1990'] , migrated_pop_percnt['2000'] ])

data = go.Table(header = header , cells = cells)

go.Figure(data , layout = go.Layout(title = 'Percentage of population migrated to different countries for given years'))
px.bar(migrated_population['1960'] , labels = dict(value = 'Migrated Population')).show()

px.bar(migrated_population['2000'] , labels = dict(value = 'Migrated Population')).show()
data = df.groupby(['Gender' , 'Country']).sum()

data['Total'] = data.sum(axis = 1)

female_mig = data.loc['Female'][  data.loc['Female']['Total'] == data.loc['Female']['Total'].max() ]

male_mig = data.loc['Male'][  data.loc['Male']['Total'] == data.loc['Male']['Total'].max() ]



header = dict(values = ['Gender' , 'Source' , 'Destination' , 'Count'])

cells =  dict(values = [ 

    ['Female' , 'Male'] , ['India' , 'India'] , [female_mig.index[0] , male_mig.index[0]] ,

    [female_mig['Total'][0],male_mig['Total'][0]] ])



go.Figure(data = go.Table(header = header , cells = cells) , layout = go.Layout(title = 'Preferable choice of country to migrate according to female and male'))
max_ = migrated_population[ migrated_population['Total'] == migrated_population['Total'].max() ]

min_ = migrated_population[ migrated_population['Total'] == migrated_population['Total'].min() ]

head = dict(values = ['Migrated The Most' , 'Count' , 'Percent' , 'Migrated the least' , 'Count' , 'Percent'])

cells = dict(values = [

        [max_.index[0]] , [max_['Total'][0]] , [max_['Percent'][0]],

        [min_.index[0]] , [min_['Total'][0]] , [min_['Percent'][0]],

])

data = go.Table(header = head , cells = cells)

go.Figure(data , layout = go.Layout(title = 'Most and Least Migrated Country '))