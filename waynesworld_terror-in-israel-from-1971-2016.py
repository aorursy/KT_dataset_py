import numpy as np
import pandas as pd
import folium
import warnings
warnings.filterwarnings('ignore')
import datetime
import calendar

import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

%matplotlib inline 

sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

from IPython.display import display, Markdown, Latex
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)

df = pd.read_csv("../input/globalterrorismdb_0617dist.csv" ,encoding='ISO-8859-1')
df_il = df[df.country_txt == "Israel"]
df_il.head()
data = df_il.groupby("iyear").nkill.sum()
data = data.reset_index()
data.columns = ["Year", "Number of fatalities"]

ax = data.plot(x="Year", y="Number of fatalities", legend=False)
ax2 = ax.twinx()

data = df_il.groupby("iyear").nkill.count()
data = data.reset_index()
data.columns = ["Year", "Number of attacks"]

data.plot(x="Year", y="Number of attacks", ax=ax2, legend=False, color="r")
ax.figure.legend(bbox_to_anchor=(0.05, 0.92), loc="upper left")
plt.tight_layout()
plt.show()
data = df_il.groupby("targtype1_txt").eventid.count().sort_values(ascending=False)[:10]
data = data.reset_index()
data.columns = ["Target", "Number of attacks"]
sns.barplot(data=data, x=data.columns[1], y=data.columns[0]);
for year in [[df_il.iyear.min(), 1980], [1980, 1990],
             [1990, 2000], [2000, 2010], [2010, df_il.iyear.max()]
            ]:
    
    m = folium.Map(
    location=[32.109333, 34.855499],
    zoom_start=7,
    tiles='Stamen Toner'
    )
    
    data = df_il.query("{} < iyear <= {}".format(year[0], year[1]))
    data = data.drop(data[data.iday < 1].index)
    data['weekday'] = [calendar.day_name[datetime.datetime(day.iyear, day.imonth, day.iday).weekday()] for i, day in data.iterrows()]
    data['date'] = [datetime.datetime(day.iyear, day.imonth, day.iday) for i, day in data.iterrows()]

    non_civ_target = ['Food or Water Supply', 'Government (Diplomatic)',
       'Government (General)', 'Journalists & Media', 'Other', 'Police', 'Telecommunication',
       'Terrorists/Non-State Militia', 'Utilities', 'Violent Political Party']
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        if row.targtype1_txt in non_civ_target:
            color = '#6b9cff'
        elif row.targtype1_txt == 'Unknown':
            color = "#e3b57e"
        else:
            color = '#9b5353'       
        
        desc = "Type: {}; Number fatalities: {}; Number wounded: {}; Year: {}".format(row.attacktype1_txt, row.nkill, row.nwound, row.iyear)
        if not pd.isnull(row.longitude):
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=row.nkill,
                popup=desc,
                color=color,
                fill=color

            ).add_to(m)
            
    display(Markdown("<center style='background: black;'><font color='white' size='12'>Terror attacks from {} to {}</font></center>".format(year[0], year[1])))
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15,3), facecolor='black')
    
    data_sub = data.groupby("weapsubtype1_txt").eventid.count().sort_values(ascending=False).iloc[:3]
    data_sub = data_sub.reset_index()
    data_sub.columns = ["Weapon Type", "Number of attacks"]
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax1)
    
    data_sub = data.groupby("targtype1_txt").eventid.count().sort_values(ascending=False).iloc[:3]
    data_sub = data_sub.reset_index()
    data_sub.columns = ["Target", "Number of attacks"]    
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax2)
    
    ax1.set_title('Most used weapons (top 3)')
    ax1.set_ylabel('')
    ax2.set_title('Most frequent targets (top 3)')
    ax2.set_ylabel('')   

    plt.tight_layout()
    plt.show()
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3), facecolor='black')
    
    data_sub = data.groupby("weekday").nkill.count().sort_values(ascending=False).iloc[:]
    data_sub = data_sub.reset_index()
    data_sub.columns = ["Weekday", "Number of attacks"]    
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax1)


    data_sub = data.groupby("gname").eventid.count().sort_values(ascending=False)
    data_sub = data_sub.reset_index()
    data_sub = data_sub.drop(data_sub[data_sub.gname == 'Unknown'].index)[:3]

    data_sub.columns = ["Attacker", "Number of attacks"]
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax2)
    
    ax1.set_title('Fatalities per weekday')
    ax1.set_ylabel('')
    ax2.set_title('Most active terror groups (top 3)')
    ax2.set_ylabel('')     
    
    plt.tight_layout()
    plt.show()
    
    display(m)
    display(Markdown("<hr></hr>"))

    
data = df_il.groupby(["gname", "targtype1_txt"])[['nkill', 'nwound']].sum()
data = data.reset_index()
data = data[data.targtype1_txt != 'Unknown']
data = data[data.gname != 'Unknown']
data = data[data.gname.isin(list(df_il.groupby("gname").nwound.sum().sort_values(ascending=False)[:10].index.tolist()))]
data = data[data.targtype1_txt.isin(list(df_il.groupby("targtype1_txt").nwound.sum().sort_values(ascending=False)[:10].index.tolist()))]


data = data.fillna(0)
data['nvictim'] = data.nkill + data.nwound
del data['nkill']
del data['nwound']
sns.heatmap(data.pivot('gname', 'targtype1_txt', 'nvictim'),square=True, linewidths=1, linecolor='white')
plt.ylabel('')
plt.xlabel('');