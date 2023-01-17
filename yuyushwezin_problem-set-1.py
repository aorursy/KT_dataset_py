# datetime operations
from datetime import timedelta

# for numerical analyiss
import numpy as np

# to store and process data in dataframe
import pandas as pd

# basic visualization package
import matplotlib.pyplot as plt

# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# to interface with operating system
import os

# for offline ploting
import matplotlib.pyplot as plt

# interactive visualization
import plotly.express as px
import seaborn as sns; sns.set()

from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# for trendlines
import statsmodels


# color pallette
# Hexademical code RRGGBB (True Black #000000, True White #ffffff)
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
!ls ../input/corona-virus-report
country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')
country_wise = country_wise.replace('', np.nan).fillna(0)

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])

day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')
day_wise['Date'] = pd.to_datetime(day_wise['Date'])

selected = full_grouped ['Country/Region'].str.contains('Singapore')



Singapore = full_grouped[selected]
Singapore.tail(10)


# Index Date
Singapore.set_index('Date', inplace=True)
Singapore.tail(10)
Singapore.tail(1)
selected = full_grouped ['Country/Region'].str.contains('Singapore')

Singapore = full_grouped[selected]
Singapore.tail(1)

temp = Singapore[['Date','Confirmed','Deaths', 'Recovered', 'Active']].tail(1)
temp.head()
# Wide to Long form data
temp = temp.melt(id_vars="Date", value_vars=['Confirmed','Active', 'Deaths', 'Recovered'])
temp
# Plot
fig = px.treemap(temp, path=["variable"], values="value", height=225, 
                 color_discrete_sequence=[cnf,act, rec, dth])
fig.data[0].textinfo = 'label+text+value'
fig.show()
selected = full_grouped ['Country/Region'].str.contains('Singapore')



Singapore = full_grouped[selected]
Singapore.tail()

# Collapse Country, Date observations to Date observations and reindex
temp = Singapore.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp.head()
# Melt the data by the value_vars
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
temp.head()
# Plot
import plotly.express as px
fig = px.area(temp, x="Date", y="Count", color='Case', height=550, width=700,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.show()
import pandas as pd
Singapore["New active"] = Singapore["Active"].diff()

Singapore.tail(10)
temp= Singapore.melt(id_vars="Date", value_vars=['New cases'],
                 var_name='Case', value_name='Count')
Singapore.head()


fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=1000,
             title='New Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
!ls ../input/singapores-covid19-cases
import pandas as pd
import numpy as np
singapore = pd.read_csv('../input/singapores-covid19-cases/singapore_covid-19_sg_cases_updated-2020-04-14.csv')
singapore = singapore.replace('', np.nan).fillna(0)


singapore.tail(10)


#singapore["imported"] = singapore["imported"].fillna(0).astype(bool)
#singaporesingapore.tail(10)
temp = singapore.groupby('date')['imported'].sum().reset_index()
temp.tail()
singapore["cluster_local"] = singapore["cluster_local"].fillna(str).astype(bool)
singapore.tail(10)

temp2 = singapore.groupby('date')['cluster_local'].sum().reset_index()
temp2.tail()
singapore["link"] = singapore["link"].fillna(str).astype(bool)
singapore.tail(10)

temp3 = singapore.groupby('date')['link'].sum().reset_index()
temp3.tail()
import pandas as pd
baseline = pd.merge(temp,temp2,left_on='date', right_on="date")
baseline.tail()
baseline2= pd.merge(baseline,temp3,left_on='date', right_on="date") 
baseline2.tail()
import plotly.express as px

temp4 = baseline2.melt(id_vars="date", value_vars=['imported', 'cluster_local','link'],
                 var_name='Case', value_name='Count')
temp4.head()

fig = px.area(temp4, x="date", y="Count", color='Case', height=600, width=1200,
             title='Sources of covid cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()