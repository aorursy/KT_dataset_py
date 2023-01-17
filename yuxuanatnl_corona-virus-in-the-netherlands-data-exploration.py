# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/coronavirus-in-nederland-23days-cityhall-based/Coronavirus in Nederland per gemeente - Gemeenten_alfabetisch_2019.csv")

df.columns
col = ['GemeentecodeGM', 'Gemeentenaam', 'Grote gemeente', 'Inwonertal',

       'Provincie', '   Vandaag', 'Per 100k 1dag', 'Per 100k < 4 dgn',

       'Per 100k <7 dgn', 'Per  100k <14dgn', ' >1 dag', '> 4 dgn', '> 7 dgn',

       '> 14 dgn', 'Vandaag', ' >1 dag.1', '> 4 dgn.1', '> 7 dgn.1',

       '> 14 dgn.1', '29-Mar', '28-Mar', '27-Mar', '26-Mar', '25-Mar',

       '24-Mar', '23-Mar', '22-Mar', '21-Mar', '20-Mar', '19-Mar', '18-Mar',

       '17-Mar', '16-Mar', '15-Mar', '14-Mar', '13-Mar', '12-Mar', '11-Mar',

       '10-Mar', '9-Mar', '8-Mar', '7-Mar', '6-Mar', '5-Mar', '4-Mar', '3-Mar',

       '2-Mar', '1-Mar', '29-Feb', '28-Feb', '27-Feb']
df.set_index('Gemeentenaam')
#fill the null value with 0

df.fillna(0)
df["Provincie"].unique()
nb = df.loc[df["Provincie"] == "Noord-Brabant"]
nbl = len(nb["Gemeentenaam"].unique())

nbname = nb["Gemeentenaam"].unique()

print("There are {} cities in Noord-Brabant : {}".format(nbl, nbname.tolist()))
nb[["Gemeentenaam","29-Mar"]]
import plotly.graph_objects as go

import plotly.offline as pyo



# Set notebook mode to work in offline!

pyo.init_notebook_mode()



labels = nb["Gemeentenaam"]

values = nb["29-Mar"]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])





fig.update_traces(textposition='inside', textinfo='label + value')

fig.update_layout(legend_orientation="h",width=800,height=600)



fig.update_layout(showlegend=False, title="The total number of postive cases in Noord-Brabant until 29th March 2020",)

fig.show()
cityinfo =['Gemeentenaam', 'Grote gemeente', 'Inwonertal']

city_date = ['Gemeentenaam','29-Mar', '28-Mar', '27-Mar', '26-Mar', '25-Mar',

       '24-Mar', '23-Mar', '22-Mar', '21-Mar', '20-Mar', '19-Mar', '18-Mar',

       '17-Mar', '16-Mar', '15-Mar', '14-Mar', '13-Mar', '12-Mar', '11-Mar',

       '10-Mar', '9-Mar', '8-Mar', '7-Mar', '6-Mar', '5-Mar', '4-Mar', '3-Mar',

       '2-Mar', '1-Mar', '29-Feb', '28-Feb', '27-Feb']

date = ['29-Mar', '28-Mar', '27-Mar', '26-Mar', '25-Mar',

       '24-Mar', '23-Mar', '22-Mar', '21-Mar', '20-Mar', '19-Mar', '18-Mar',

       '17-Mar', '16-Mar', '15-Mar', '14-Mar', '13-Mar', '12-Mar', '11-Mar',

       '10-Mar', '9-Mar', '8-Mar', '7-Mar', '6-Mar', '5-Mar', '4-Mar', '3-Mar',

       '2-Mar', '1-Mar', '29-Feb', '28-Feb', '27-Feb']
print("Top 20 biggest city in Noord-Brabant")

nbinfo = nb[cityinfo].sort_values(by=['Inwonertal'],ascending=False).head(20)

nbinfo.rename(columns={"Gemeentenaam": "City", "Grote gemeente": "Big\middle\small","Inwonertal": "Registered population"})
nb =nb[city_date].fillna(0)

eindhoven = nb.loc[nb["Gemeentenaam"] =="Eindhoven"]

valuesperday = eindhoven[date].values[0][::-1] #[::-1] is for reverse the numberorder

dates = date[::-1]

fig = go.Figure(data=go.Scatter(x=dates, y=valuesperday))

fig.update_layout(showlegend=False, title="Eindhoven Trend",)

fig.show()
#Get the Netherlands all data as total. 

df = df[df['GemeentecodeGM'].notna()]

#then fill the null number with 0

df = df.fillna(0)

#then get the sum of each colum, store in total

df[dates] = df[dates].apply(pd.to_numeric)

total_value= df[dates].sum().values



################ put the netherlands line with eindhoven line together

nb =nb[city_date].fillna(0)

eindhoven = nb.loc[nb["Gemeentenaam"] =="Eindhoven"]

valuesperdayEindhoven = eindhoven[date].values[0][::-1] #[::-1] is for reverse the numberorder

dates = date[::-1]



fig = go.Figure(data=go.Scatter(x=dates, y=valuesperdayEindhoven))

fig.update_layout(showlegend=False, title="Eindhoven Trend",)



fig = go.Figure()

fig.add_trace(go.Scatter(x=dates, y=valuesperdayEindhoven,

                    mode='lines+markers',

                    name='Eindhoven'))

fig.add_trace(go.Scatter(x=dates, y=total_value,

                    mode='lines+markers',

                    name='The Netherlands'))

fig.update_layout(width=500,height=500,showlegend=True, title="NL with Eindhoven Trend")

fig.show()
pr = df.sort_values(by=["Provincie"])

protable = pr.groupby("Provincie").sum()

protable
#extracting value, take Drenthe as one example

protable.loc["Drenthe"][-32:].values[::-1]
pname = []

pro_value_32_days = []

for p in protable.index:

    v = protable.loc[p][-32:].values[::-1]

    pro_value_32_days.append(v)

    pname.append(p)
pname 

pro_value_32_days



fig = go.Figure()

for i in range(0,12):

    fig.add_trace(go.Scatter(x=dates, y=pro_value_32_days[i],

                    mode='lines+markers',

                    name=pname[i]))



fig.update_layout(title="Trend per provience in NL")

fig.show()