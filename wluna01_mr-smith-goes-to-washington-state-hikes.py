# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import missingno as mno
# Any results you write to the current directory are saved as output.

weather_filepath = "../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv"

weather_data = pd.read_csv(weather_filepath, index_col=0, encoding="latin-1")

weather_data.head()
climbing_filepath = "../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv"

climbing_data = pd.read_csv(climbing_filepath, index_col=0, encoding="latin-1")

climbing_data.head()
result = pd.merge(weather_data, 

                  climbing_data[['Route','Success Percentage']],

                  on='Date',

                 how='left')

result.head()
result.tail()
result = pd.merge(weather_data, 

                  climbing_data[['Route','Success Percentage']],

                  on='Date',

                 how='inner')

result.tail()
TempGraph = result.filter(['Temperature AVG', 'Success Percentage'], axis=1)

TempGraph.tail()
sns.scatterplot(x="Success Percentage", y="Temperature AVG", data=TempGraph)
quantile = TempGraph["Success Percentage"].quantile(0.99)

TempGraph = TempGraph[TempGraph["Success Percentage"] < quantile]
sns.jointplot(x='Success Percentage', y='Temperature AVG', data=TempGraph, kind='reg')
sns.jointplot(x='Success Percentage', y='Temperature AVG', data=TempGraph, kind='hex')
result = result[result['Success Percentage'] < result['Success Percentage'].quantile(0.99)]

sns.lmplot(x='Success Percentage', y='Temperature AVG', hue = 'Route', data=result)
result['Route'] = np.where((result['Route'] != "Disappointment Cleaver"), "Other", result['Route'])

sns.lmplot(x='Success Percentage', y='Temperature AVG', hue = 'Route', data=result)
result.corr()