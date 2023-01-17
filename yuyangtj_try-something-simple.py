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
train = pd.read_csv('/kaggle/input/learn-together/train.csv')
train.describe()

elevation = train.Elevation.to_numpy()

features = train.columns[1:-1]

num_of_features = len(features)
import matplotlib.pyplot as plt

%matplotlib inline



fig, axes = plt.subplots(5, 2, figsize=(12, 30))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



for i in range(10): 

    train.plot.scatter(ax = axes[i//2][i%2], y=features[i], x = 'Cover_Type')

# distribution of cover_type in each wildness area



fig, axes = plt.subplots(2, 2, figsize=(12, 12))

plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(4):

    temp_df = train.filter(['Wilderness_Area%s'%(i+1), 'Cover_Type'])

    temp_df.groupby('Cover_Type').sum().plot.pie(subplots=True, ax = axes[i//2][i%2])
wildnerness_areas = []

for i in range(4):

    wildnerness_areas.append('Wilderness_Area%s'%(i+1))

                             

wildnerness_areas.append('Cover_Type') 

temp_df = train.filter(wildnerness_areas)
num_cover_type = len(set(train['Cover_Type']))



fig, axes = plt.subplots(4, 2, figsize=(12, 24))

plt.subplots_adjust(wspace=0.4, hspace=0.3)

for i in range(num_cover_type):

    temp_df.groupby('Cover_Type').sum().iloc[i].plot.pie(subplots=True, ax = axes[i//2][i%2])

num_soil_type = 40

soil_types = []



for i in range(num_soil_type):

    soil_types.append('Soil_Type%s'%(i+1))

    

soil_types.append('Cover_Type')

temp_df = train.filter(soil_types)

fig, axes = plt.subplots(4, 2, figsize=(12, 24))

plt.subplots_adjust(wspace=0.4, hspace=0.3)

for i in range(num_cover_type):

    temp_df.groupby('Cover_Type').sum().iloc[i].plot.pie(subplots=True, ax = axes[i//2][i%2])