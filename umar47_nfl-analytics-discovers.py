import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization

import plotly.express as px #visualization

import chart_studio.plotly as py #visualization

import plotly.figure_factory as ff #visualization

import plotly.graph_objs as go #visualization

import matplotlib.pyplot as plt #visualization

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

Injurydf=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

Injurydf
Playdf=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

Playdf
Playdf['StadiumType']=Playdf['StadiumType'].replace(to_replace =['Outdoors', 'Outddors', 'Oudoors','Oudoor', 'Dome', 'Indoor'], 

                            value =["Outdoor", "Outdoor", "Outdoor","Outdoor",  "Indoors", "Indoors"])
plays=list(Injurydf['PlayerKey'].unique())

Playdf_Copy=Playdf[Playdf.PlayerKey.isin(plays)]

Playdf_Copy
last_row0=Playdf_Copy.groupby('PlayerKey').tail(1)

rp=last_row0['StadiumType'].value_counts()

x1=rp[:5].index

y1=rp[:5].values



st=last_row0['Position'].value_counts()

x2=st[:5].index

y2=st[:5].values



wt=last_row0['RosterPosition'].value_counts()

x3=wt[:5].index

y3=wt[:5].values



pt=last_row0['PlayType'].value_counts()

x4=pt[:5].index

y4=pt[:5].values

plt.subplots_adjust(left=0.005, bottom=0.01, right=2.5, top=2.1)

plt.subplot(2, 2, 1)

sns.barplot(x1, y1)

plt.title('StadiumType',fontsize=20)

plt.ylabel('Number')

plt.subplot(2, 2, 2)

plt.title('RosterPosition', fontsize=20)

sns.barplot(x3, y3)

plt.ylabel('Number')

plt.subplot(2, 2, 3)

sns.barplot(x2, y2)

plt.title('Position', fontsize=20)

plt.ylabel('Number')

plt.subplot(2, 2, 4)

plt.title('PlayType', fontsize=20)

sns.barplot(x4, y4)

plt.ylabel('Number')



plt.show()
Trackdf=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')

Trackdf_Copy=Trackdf.merge(Injurydf, how='inner')
Trackdf
Trackdf_Copy['day_missed']=(Trackdf_Copy['DM_M1']+Trackdf_Copy['DM_M7']+Trackdf_Copy['DM_M28']+Trackdf_Copy['DM_M42'])

Trackdf_Copy=Trackdf_Copy.drop(['DM_M1', 'DM_M7', 'DM_M28','DM_M42'], axis=1)

Trackdf_Copy
last_row1=Trackdf_Copy.groupby('PlayerKey').tail(1)#last row for each player
rp=last_row1['BodyPart'].where(Trackdf_Copy['Surface']=='Natural').value_counts()

st=last_row1['BodyPart'].where(Trackdf_Copy['Surface']=='Synthetic').value_counts()



fig = go.Figure()

fig.add_trace(go.Bar(x=st.index, y=st.values, name="Synthetic"))

fig.add_trace(go.Bar(x=rp.index, y=rp.values, name="Natural"))

fig.update_layout(title_text="BodyPart & Surface  ",

                   width=850,

                   height=600,

                  title_font_size=20)

fig.show()
rp=last_row1['day_missed'].where(Trackdf_Copy['Surface']=='Natural').value_counts()

st=last_row1['day_missed'].where(Trackdf_Copy['Surface']=='Synthetic').value_counts()

fig = go.Figure()

fig.add_trace(go.Bar(x=st.index, y=st.values, name="Synthetic"))

fig.add_trace(go.Bar(x=rp.index, y=rp.values, name="Natural"))

fig.update_layout(title_text="day_missed & Surface ",

                   width=850,

                   height=500,

                  title_font_size=20,

                 barmode='stack')

fig.show()
KN=last_row1['day_missed'].where(last_row1['BodyPart']=='Knee').value_counts()

FT=last_row1['day_missed'].where(last_row1['BodyPart']=='Foot').value_counts()

AN=last_row1['day_missed'].where(last_row1['BodyPart']=='Ankle').value_counts()

fig = go.Figure()

fig.add_trace(go.Bar(x=KN.index, y=KN.values, name="Knee"))

fig.add_trace(go.Bar(x=FT.index, y=FT.values, name="Foot"))

fig.add_trace(go.Bar(x=AN.index, y=AN.values, name="Ankle"))

fig.update_layout(title_text="day_missed & BodyPart",

                   width=850,

                   height=500,

                  title_font_size=20, 

                 barmode='stack')

fig.show()
last_row2=Trackdf_Copy.groupby('PlayerKey').tail(1)#last row for each inuried player

fig = px.scatter(last_row2, x="x", y="y", 

              title="The players last position  ")

fig.show()
fig = px.scatter(last_row2, x="x", y="y", color="Surface", 

                 labels='BodyPart', title="Surface types for injuried player on the field")

fig.show()
fig = px.scatter(last_row2, x="x", y="y", color="BodyPart", title="BodyPart distrubiton on the field")

fig.show()
fig = px.scatter(last_row2, x="x", y="y", color="event", title="last event's distrubiton on the field")

fig.show() 
event_dist=last_row2['event'].value_counts()

fig = px.bar(x=event_dist.values, y=event_dist.index, orientation='h', title="last event of injuried players")

fig.show()
last_row5=Trackdf_Copy.groupby('PlayerKey').head(2)

event_dist1=last_row5['event'].value_counts()

fig = px.bar(x=event_dist1.values, y=event_dist1.index, orientation='h', title="first event  injuried players")

fig.show()
players=list(Trackdf_Copy['PlayKey'].unique())

players_track=Trackdf[Trackdf.PlayKey.isin(players)]

players_track['PlayerKey']=players_track.PlayKey.str.split('-').str[0]

players_track.sample(4)
#Acceleration calculation found on stackoverflow

players_track['acceleration'] = (players_track['s'] - players_track['s'].shift(1)) / (players_track['time'] - players_track['time'].shift(1))
fig = px.line(players_track, x="x", y="y", color='PlayerKey',

              title="Movements of player on the field in the game ")

fig.show()
fig = px.line(Trackdf_Copy, x="x", y="y", color='PlayerKey',

              title="Movements of player on the field during injury ")

fig.show()
last_row3=players_track.groupby('PlayerKey').tail(15)#last row for each inuried player

fig = px.line(last_row3, x="x", y="y", color='PlayerKey',

              title="last moves of injuried players")

fig.show()
fig = px.line(players_track, x="time", y="acceleration", color="PlayerKey", title='Acceleration change with time')

fig.show()
Injured_speed=Trackdf_Copy['s'].mean()

Total_speed=players_track['s'].mean()

Injured_dis=Trackdf_Copy['dis'].mean() 

Total_dis=players_track['dis'].mean()

Injured_ac=last_row3['acceleration'].mean() 

Total_ac=players_track['acceleration'].mean()

percentage=round(((Injured_speed-Total_speed)/Total_speed)*100)

percentage1=round(((Injured_dis-Total_dis)/Total_dis)*100)

percentage2=round(((Injured_ac-Total_ac)/Total_ac)*100)

print("The difference in speed, distance and acceleration changes throughout the game vs injury time:")

print(f"The players are moved  %{percentage} yards/meter faster then usual")

print(f"The players move %{percentage1} meter more then usual ")

print(f"The players moving acceleration %{percentage2} less then usual ")