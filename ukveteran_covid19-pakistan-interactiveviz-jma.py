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
df = pd.read_excel("../input/corona-virus-pakistan-dataset-2020/COVID_FINAL_DATA.xlsx")
df.head()
df.head().style.background_gradient(cmap='Greens')
df_reg=df.groupby(['Region']).agg({'Discharged':'sum','Expired':'sum'}).sort_values(["Discharged"],ascending=False).reset_index()
df_reg.head(10)
fig = go.Figure(data=[go.Table(
    columnwidth = [50],
    header=dict(values=('Region', 'Discharged', 'Expired'),
                fill_color='#104E8B',
                align='center',
                font_size=14,
                font_color='white',
                height=40),
    cells=dict(values=[df_reg['Region'].head(10), df_reg['Discharged'].head(10), df_reg['Expired'].head(10)],
               fill=dict(color=['#509EEA', '#A4CEF8',]),
               align='right',
               font_size=12,
               height=30))
])

fig.show()
df_reg.iplot(kind='box')
fig = px.pie(df_reg.head(10),
             values="Discharged",
             names="Region",
             title="Discharged",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()
fig = px.pie(df_reg.head(10),
             values="Expired",
             names="Region",
             title="Expired",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()