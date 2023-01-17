import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/airbnb-istanbul-dataset/AirbnbIstanbul.csv")

df.sample(5)
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="neighbourhood", hover_data=["price", "room_type"],

                              color_discrete_sequence=["fuchsia"], zoom=8, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()
x=pd.DataFrame(df.groupby(['neighbourhood'])[['price', 'calculated_host_listings_count', 'availability_365', 'number_of_reviews', 'longitude', 'latitude']].mean())

y=pd.DataFrame(df.groupby('neighbourhood')['room_type'].value_counts().unstack().fillna(0))

z=pd.concat([y, x.reindex(y.index)], axis=1)

z['total_houses']= z['Entire home/apt'] + z['Private room'] + z['Shared room']

z.sample(5)
z['price_score']=z.price/z.price.max()

z['host_score']=z.calculated_host_listings_count/z.calculated_host_listings_count.max()

z['availability_score']=z.availability_365/z.availability_365.max()

z['number_of_reviews_score']=z.number_of_reviews/z.number_of_reviews.max()

z['total_houses_score']=z.total_houses/z.total_houses.max()

z['total_score']=z['total_houses_score'] + z['number_of_reviews_score'] + z['availability_score'] + z['host_score'] + z['price_score']

z.sample(3)
fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["total_score"], color="total_score",

                         zoom=8, height=300, size='total_score')

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()
fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["host_score"], color="host_score",

                         zoom=8, height=300, size='host_score')

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()
fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["availability_score"], color="availability_score",

                         zoom=8, height=300, size='availability_score')

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()
fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["number_of_reviews_score"], color="number_of_reviews_score",

                         zoom=8, height=300, size='number_of_reviews_score')

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()
z['max_annual_income']=round(z['price']*z['availability_365'])

fig = px.scatter_mapbox(z, lat="latitude", lon="longitude",  hover_name=z.index, hover_data=["max_annual_income"], color="max_annual_income",

                         zoom=8, height=300, size='max_annual_income')

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()