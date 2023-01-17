# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

iso_df = pd.read_csv('/kaggle/input/isocountrycodes/ISOCountryCodes.csv')
recovered.head()
confirmed.head()
deaths.head()
confirmed_df = pd.melt(confirmed, id_vars=['Province/State','Country/Region','Lat','Long'], var_name=['ObservationDate'])



# Convert RecordedDate to datetime type

confirmed_df['ObservationDate'] = confirmed_df['ObservationDate'].apply(pd.to_datetime)



#recovered_df_gt_zero = recovered_df['value'] > 0

confirmed_df = confirmed_df[confirmed_df['value'] > 0]



conf_merged = pd.merge(confirmed_df, iso_df, how='left', left_on='Country/Region', right_on='Country')



conf_merged['date'] = conf_merged['ObservationDate'].apply(lambda x: x.strftime('%Y%m%d'))

conf_merged['date'] = conf_merged['date'].apply(pd.to_numeric)



conf_merged = conf_merged.groupby(['Country/Region','ObservationDate','date','ISO3 Code'])['value'].sum().reset_index()



conf_merged = conf_merged.sort_values(by='ObservationDate', ascending=True)



conf_merged.head()
deaths_df = pd.melt(deaths, id_vars=['Province/State','Country/Region','Lat','Long'], var_name=['ObservationDate'])



# Convert RecordedDate to datetime type

deaths_df['ObservationDate'] = deaths_df['ObservationDate'].apply(pd.to_datetime)



#recovered_df_gt_zero = recovered_df['value'] > 0

deaths_df = deaths_df[deaths_df['value'] > 0]



deaths_merged = pd.merge(deaths_df, iso_df, how='left', left_on='Country/Region', right_on='Country')



deaths_merged['date'] = deaths_merged['ObservationDate'].apply(lambda x: x.strftime('%Y%m%d'))

deaths_merged['date'] = deaths_merged['date'].apply(pd.to_numeric)



deaths_merged = deaths_merged.groupby(['Country/Region','ObservationDate','date','ISO3 Code'])['value'].sum().reset_index()



deaths_merged = deaths_merged.sort_values(by='ObservationDate', ascending=True)



deaths_merged.head()
# Transform

recovered_df = pd.melt(recovered, id_vars=['Province/State','Country/Region','Lat','Long'], var_name=['ObservationDate'])



# Convert RecordedDate to datetime type

recovered_df['ObservationDate'] = recovered_df['ObservationDate'].apply(pd.to_datetime)



recovered_df = recovered_df[recovered_df['value'] > 0]



rec_merged = pd.merge(recovered_df, iso_df, how='left', left_on='Country/Region', right_on='Country')



rec_merged['date'] = rec_merged['ObservationDate'].apply(lambda x: x.strftime('%Y%m%d'))

rec_merged['date'] = rec_merged['date'].apply(pd.to_numeric)



rec_merged = rec_merged.groupby(['Country/Region','ObservationDate','date','ISO3 Code'])['value'].sum().reset_index()



rec_merged = rec_merged.sort_values(by='ObservationDate', ascending=True)



rec_merged.head()
rec_merged['Country/Region'].value_counts()
fig5 = px.scatter_geo(conf_merged, locations="ISO3 Code",

                     hover_name="Country/Region", size="value", size_max=45,

                     animation_frame="date", title="Confirmed cases over time (click on play button to visualize data)"

                     )

fig5.show()
fig6 = px.scatter_geo(deaths_merged, locations="ISO3 Code",

                     hover_name="Country/Region", size="value", size_max=45,

                     animation_frame="date", title="Deaths recorded over time (click on play button to visualize data)"

                     )

fig6.show()
fig1 = px.scatter_geo(rec_merged, locations="ISO3 Code",

                     hover_name="Country/Region", size="value", size_max=45,

                     animation_frame="date",title="Recovery trends over time (click on play button to visualize data)"

                     )

fig1.show()
fig3 = px.scatter_mapbox(deaths_df, lat="Lat", lon="Long", hover_name="Country/Region", hover_data=["Country/Region","Province/State", "value"],

                        color_discrete_sequence=["fuchsia"], zoom=3, height=800)

fig3.update_layout(mapbox_style="open-street-map")

fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig3.update_layout(title="Death counts by Country and Province")

fig3.show()