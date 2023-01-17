import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()





import os

import altair as alt

alt.renderers.enable('notebook')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
data.head()
inspection_score_daily = data.groupby('inspection_date').inspection_score.mean().reset_index()

inspection_score_daily['date'] = pd.to_datetime(inspection_score_daily['inspection_date'])

inspection_score_daily['year'] = inspection_score_daily['date'].dt.year
plot_data = [go.Scatter(x=inspection_score_daily.date, y=inspection_score_daily.inspection_score)]

# specify the layout of our figure

layout = dict(title = "Average Inspection Score",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = plot_data, layout = layout)

iplot(fig)
risk_category_count = data.pivot_table(columns='risk_category', index='inspection_type', 

                                       values='inspection_id', 

                                       aggfunc='count')
risk_category_count
plot_data = []

for t in list(risk_category_count.index):

    plot_data.append(go.Bar(x=list(risk_category_count.columns), 

                            y=list(risk_category_count.loc[t]),

                            name=t

                           )

                    )

    

layout = go.Layout(

    barmode='stack',

    title = "Count of Risk Category By Inspection Type"

)



# create and show our figure

fig = go.Figure(data=plot_data, layout=layout)

#fig = dict(data = plot_data, layout = layout)

iplot(fig)