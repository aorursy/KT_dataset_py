import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
%matplotlib inline


import plotly.tools as tls
import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

print(__version__) # requires version >= 1.9.0
cf.go_offline()
df = pd.read_csv('../input/avocado-prices-2020/avocado-updated-2020.csv')
df.head()
df_reg=df.groupby(['total_volume']).agg({'average_price':'sum'}).sort_values(["average_price"],ascending=False).reset_index()
df_reg.head(10)
fig = go.Figure(data=[go.Table(
    columnwidth = [50],
    header=dict(values=('total_volume', 'average_price'),
                fill_color='#104E8B',
                align='center',
                font_size=14,
                font_color='white',
                height=40),
    cells=dict(values=[df_reg['total_volume'].head(10), df_reg['average_price'].head(10)],
               fill=dict(color=['#509EEA', '#A4CEF8',]),
               align='right',
               font_size=12,
               height=30))
])

fig.show()
fig = px.pie(df_reg.head(10),
             values="average_price",
             title="Average Price",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()
fig = px.pie(df_reg.head(10),
             values="total_volume",
             title="Total Volume",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()
df_county=df.groupby(['date','total_volume']).agg({'average_price':'sum'}).sort_values(["average_price"],ascending=False)
df_county.head(10)
dfd = df_county.groupby('date').sum()
dfd.head()
dfd[['average_price']].iplot(title = 'Situation Over Time')