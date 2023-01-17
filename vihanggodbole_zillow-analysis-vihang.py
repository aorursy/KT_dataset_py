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

# Get all data

dfCountyCrossWalk = pd.read_csv('/kaggle/input/zecon/CountyCrossWalk_Zillow.csv')

dfNeighborhoodTimeSeries = pd.read_csv('/kaggle/input/zecon/Neighborhood_time_series.csv')

dfZipTimeSeries = pd.read_csv('/kaggle/input/zecon/Zip_time_series.csv')

dfCountyTimeSeries = pd.read_csv('/kaggle/input/zecon/County_time_series.csv')

dfMetroTimeSeries = pd.read_csv('/kaggle/input/zecon/Metro_time_series.csv')

dfCitiesCrosswalk = pd.read_csv('/kaggle/input/zecon/cities_crosswalk.csv')

dfStateTimeSeries = pd.read_csv('/kaggle/input/zecon/State_time_series.csv')

dfCityTimeSeries = pd.read_csv('/kaggle/input/zecon/City_time_series.csv')
dfCityTimeSeries.head()