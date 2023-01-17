import os

import pandas as pd # for dataframes

import numpy as np



adsm = pd.read_excel('/kaggle/input/adsm-b2/ADSM_B2_Masked.xlsx')

adsm.head(5)
# Create a Bar chart 

import plotly.express as px



fig = px.histogram(adsm, x="Professional Experience", nbins=8)

fig.show()
# Count Students in every Work Location

adsm_location2 = adsm[['Work Location', 'Name']].groupby(['Work Location']).agg(['count'])



# Convert the Pivot to DataFrame

adsm_location2.columns = adsm_location2.columns.droplevel(0)

adsm_location2 = adsm_location2.reset_index().rename_axis(None, axis=1)



# Check the data

adsm_location2.head()

# Tryout map with data

fig_loc = px.scatter_geo(adsm_location2, locations="Work Location",

                     size="count", # size of markers, "pop" is one of the columns of gapminder

                     )

fig_loc.show()

# this did not work due to city.
# Histogram for Cities

figl = px.histogram(adsm_location2.sort_values(by='count', ascending=False), x="Work Location", y = 'count',  title="Student Location")

figl.show()