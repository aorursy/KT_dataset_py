# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import plotly.express as px

import plotly.graph_objs as pltgo

import statistics

from statistics import mean



import sklearn

from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
emission_data = pd.read_csv('/kaggle/input/emission-data-cleaned/emission_data_cleaned.csv')

emission_data.info()

emission_data.head(5)
world_pollution_2017 = emission_data[['Country','2017']]

world_pollution_2017.head(5)
iso_alpha_codes = pd.read_csv('/kaggle/input//iso-codes/iso_alpha.csv')

iso_alpha_codes.head(5)

world_pollution_2017_iso = pd.merge(left=world_pollution_2017,right=iso_alpha_codes,left_on='Country',right_on='Name',how='left')

world_pollution_2017_cleaned = world_pollution_2017_iso.dropna(subset=['Name'])

world_pollution_2017_cleaned.nlargest(7, '2017')
fig_map_2018 = pltgo.Figure(data=pltgo.Choropleth(locations=world_pollution_2017_cleaned['Country'],z=world_pollution_2017_cleaned['2017'],locationmode='country names',colorscale='Rainbow',colorbar_title='Tonnes'))

fig_map_2018.update_layout(title_text='Emission of Green House Gases 2017 (In Tonnes)')

fig_map_2018.show()
india_emission = emission_data[emission_data.Country=='India'].iloc[:,248:269].values.tolist()

india_emission_array = []

for col in india_emission:

    for ele in col:

        india_emission_array.append(ele)



        

us_emission = emission_data[emission_data.Country=='United States'].iloc[:,248:269].values.tolist()

us_emission_array = []

for col in us_emission:

    for ele in col:

        us_emission_array.append(ele)



eu_emission = emission_data[emission_data.Country=='EU-28'].iloc[:,248:269].values.tolist()

eu_emission_array = []

for col in eu_emission:

    for ele in col:

        eu_emission_array.append(ele)



russia_emission = emission_data[emission_data.Country=='Russian Federation'].iloc[:,248:269].values.tolist()

russia_emission_array = []

for col in russia_emission:

    for ele in col:

        russia_emission_array.append(ele)

        

japan_emission = emission_data[emission_data.Country=='Japan'].iloc[:,248:269].values.tolist()

japan_emission_array = []

for col in japan_emission:

    for ele in col:

        japan_emission_array.append(ele)



china_emission = emission_data[emission_data.Country=='China'].iloc[:,248:269].values.tolist()

china_emission_array = []

for col in china_emission:

    for ele in col:

        china_emission_array.append(ele)



row_emission = emission_data[~emission_data.Country.isin(['India','Japan','United States','EU-28','Russia'])].iloc[:,248:269].values.tolist()

row_emission_array = []

for col in row_emission:

    for ele in col:

        row_emission_array.append(ele)
world_emission_time_series_1998_2017_data = {'Year':range(1998,2018), 'India_Emission':india_emission_array, 'US_Emission':us_emission_array, 'Russia_emission':russia_emission_array,

                                            'Japan_Emission':japan_emission_array, 'EU_emission':eu_emission_array, 'China_emission':china_emission_array}

print(len(range(1998,2018)),len(us_emission_array),len(russia_emission_array),len(japan_emission_array),len(china_emission_array),len(eu_emission_array),len(india_emission_array))

world_emission_time_series_1998_2017 = pd.DataFrame(world_emission_time_series_1998_2017_data)
fig_time_series = pltgo.Figure([pltgo.Scatter(x=world_emission_time_series_1998_2017.Year,y=world_emission_time_series_1998_2017.India_Emission,name='India',mode='lines+markers')])

fig_time_series.add_trace(pltgo.Scatter(x=world_emission_time_series_1998_2017.Year,y=world_emission_time_series_1998_2017.US_Emission,name='US',mode='lines+markers'))

fig_time_series.add_trace(pltgo.Scatter(x=world_emission_time_series_1998_2017.Year,y=world_emission_time_series_1998_2017.Japan_Emission,name='Japan',mode='lines+markers'))

fig_time_series.add_trace(pltgo.Scatter(x=world_emission_time_series_1998_2017.Year,y=world_emission_time_series_1998_2017.Russia_emission,name='Russia',mode='lines+markers'))

fig_time_series.add_trace(pltgo.Scatter(x=world_emission_time_series_1998_2017.Year,y=world_emission_time_series_1998_2017.EU_emission,name='EU',mode='lines+markers'))

fig_time_series.add_trace(pltgo.Scatter(x=world_emission_time_series_1998_2017.Year,y=world_emission_time_series_1998_2017.China_emission,name='China',mode='lines+markers'))

fig_time_series.update_layout(title_text='Green House Gas Emission 1998-2017')

fig_time_series.show()