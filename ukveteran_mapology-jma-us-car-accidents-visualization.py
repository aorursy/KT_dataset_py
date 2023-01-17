import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import matplotlib.ticker as ticker





%matplotlib inline
df = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')

df.head()
df.describe().T
df.info()
df['Source'].unique()
df_st_ct = pd.value_counts(df['State'])



fig = go.Figure(data=go.Choropleth(

    locations=df_st_ct.index,

    z = df_st_ct.values.astype(float),  # Data to be color-coded

    locationmode = 'USA-states',     # set of locations match entries in `locations`

    colorscale = 'YlOrRd',

    colorbar_title = "Count",

))



fig.update_layout(

    title_text = 'US Accidents by State',

    geo_scope='usa', # limite map scope to USA

)



fig.show()