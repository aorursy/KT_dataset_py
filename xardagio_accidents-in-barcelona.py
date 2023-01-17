# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium import plugins

from plotly import __version__

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
data = pd.read_csv('../input/accidents_2017.csv')
data.head()
data[['Mild injuries','Serious injuries','Victims','Vehicles involved']].describe()
data['Id'].apply(lambda x: x[:4]).unique()
data.columns
data.info()
part_day_data = data[['Vehicles involved','Victims','Part of the day','Mild injuries','Serious injuries',]]

part_day_data = part_day_data.set_index('Part of the day')

part_day_data = part_day_data.groupby(level=[0]).sum()

part_day_data = part_day_data.sort_index()

part_day_data = part_day_data.reindex(['Morning','Afternoon','Night'])
layout = dict(title='Accidents by part of the day',geo=dict(showframe=False))

part_day_data.iplot(kind='bar',layout=layout)
hour_day_data = data[['Hour','Vehicles involved','Victims','Mild injuries','Serious injuries',]]

hour_day_data = hour_day_data.set_index('Hour')

hour_day_data = hour_day_data.groupby(level=[0]).sum()

hour_day_data = hour_day_data.sort_index()

hour_day_data.head()
layout = dict(title='Accidents by hour of the day',geo=dict(showframe=False))

hour_day_data.iplot(kind='bar',layout=layout)
weekday_data = data[['Weekday','Vehicles involved','Victims','Mild injuries','Serious injuries',]]

weekday_data = weekday_data.set_index('Weekday')

weekday_data = weekday_data.groupby(level=[0]).sum()

weekday_data.head(7)

weekday_data = weekday_data.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
layout = dict(title='Accidents by weekday',geo=dict(showframe=False))

weekday_data.iplot(kind='bar',layout=layout)
month_data = data[['Month','Vehicles involved','Victims','Mild injuries','Serious injuries',]]

month_data = month_data.set_index('Month')

month_data = month_data.groupby(level=[0]).sum()

month_data = month_data.sort_index()

month_data = month_data.reindex(['January',

                                 'February',

                                 'March',

                                 'April',

                                 'May',

                                 'June',

                                 'July',

                                 'August',

                                 'September',

                                 'October',

                                 'November',

                                 'December'])
layout = dict(title='Accidents by month',geo=dict(showframe=False))

month_data.iplot(kind='bar',layout=layout)
barcelona_map = folium.Map(location=[41.395425, 2.169141],zoom_start = 12) 



coordinates_data = data[['Latitude', 'Longitude']]

coordinates_data['Weight'] = data['Hour']

coordinates_data['Weight'] = coordinates_data['Weight'].astype(float)

coordinates_data = coordinates_data.dropna(axis=0, subset=['Latitude','Longitude', 'Weight'])



coordinates_list = [[[row['Latitude'],row['Longitude']] for index, row in coordinates_data[coordinates_data['Weight'] == i].iterrows()] for i in range(0,24)]



hm = plugins.HeatMapWithTime(coordinates_list,auto_play=True,max_opacity=0.8)

hm.add_to(barcelona_map)



barcelona_map