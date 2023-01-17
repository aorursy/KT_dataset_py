import pandas as pd
import numpy as np
!pip install --quiet pycountry_convert
from pycountry_convert import country_name_to_country_alpha3
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import os
data = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
data[:5]
data.info()
data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
data.rename({' Rocket': 'Cost'}, axis=1, inplace=True)
data.Cost.unique()
data['Cost'].loc[data['Cost'] == '5,000.0 '] = '5.0'
data['Cost'].loc[data['Cost'] == '1,160.0 '] = '1.16'
data.Cost = data.Cost.astype(float).fillna(0.0)
data['date'] = pd.to_datetime(data['Datum'])
data['year'] = data['date'].apply(lambda datetime: datetime.year)
data['country'] = data['Location'].str.split(', ').str[-1]
data['country'].loc[data['country'] == 'Shahrud Missile Test Site'] = "Iran"
data['country'].loc[data['country'] == 'New Mexico'] = 'USA'
data['country'].loc[data['country'] == 'Yellow Sea'] = "China"
data['country'].loc[data['country'] == 'Pacific Missile Range Facility'] = "USA"
data['country'].loc[data['country'] == 'Pacific Ocean'] = "USA"
data['country'].loc[data['country'] == 'Barents Sea'] = 'Russia'
data['country'].loc[data['country'] == 'Gran Canaria'] = 'USA'
def get_iso(col):
    try:
        iso_3 = country_name_to_country_alpha3(col)
    except:
        iso_3 = 'Unknown'
    return iso_3

data['iso_alpha'] = data['country'].apply(lambda x: get_iso(x))
data_map = pd.DataFrame(data.groupby(['country', 'iso_alpha'])['iso_alpha'].agg(Missions='count')).reset_index()
fig = px.scatter_geo(data_map, locations="iso_alpha",
                     color="country",
                     hover_name="country",
                     size="Missions",
                     projection="natural earth")
fig.show()
table = pd.pivot_table(data, values='year', index='country',
                    columns='Status Mission', aggfunc='count', fill_value=0)
table['Success (in prc)'] = table['Success'] / table.sum(axis=1)
table.style.format({'Success (in prc)' : '{:.2%}'})\
           .background_gradient(cmap='Reds')\
           .background_gradient(cmap='Blues',subset=["Success"])\
           .background_gradient(cmap='YlOrBr',subset=["Success (in prc)"])
data_r = data.Detail.value_counts()
data_r = data_r[data_r > 1]
table = data[data.Detail.isin(data_r.index)].groupby(['country', 'Company Name', 'Detail', 'Status Mission'])['Status Mission'].agg(Count='count')
table.style.background_gradient(cmap='YlOrBr',subset=["Count"])
data_fail = data[(data['Status Mission'] != 'Success') & (data['Cost'] != 0)]
fig = px.bar(data_fail, x='Company Name', y='Cost', color='country', title = 'Failure costs')
fig.show()
fig = px.sunburst(data_fail, path=['country', 'Company Name'], values='Cost', color='country',
                  title='')
fig.update_layout(autosize=False, width=600, height=600)
fig.show()
data_fail.Cost.sum()