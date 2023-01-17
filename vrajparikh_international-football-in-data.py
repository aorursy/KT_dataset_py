# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")
df.info()
df.head()
df.isnull().sum()
df["winner"]=df["home_score"]-df["away_score"]

df["loser"]=df["winner"]

df["draw"]=df["winner"]
for x in range(0,df.shape[0]):

    if df.iloc[x,9]>0:

        df.iloc[x,9]=df.iloc[x,1]

        df.iloc[x,10]=df.iloc[x,2]

        df.iloc[x,11]=False

    elif df.iloc[x,9]<0:

        df.iloc[x,9]=df.iloc[x,2]

        df.iloc[x,10]=df.iloc[x,1]

        df.iloc[x,11]=False

    else :

        df.iloc[x,9]=float("Nan")

        df.iloc[x,10]=float("Nan")

        df.iloc[x,11]=True
df.shape
df["winner"].value_counts().loc["Brazil"]
away=df["away_team"].value_counts()

home=df["home_team"].value_counts()

for x in away.index:

    if x not in home.index:

        home[x]=0

for x in home.index:

    if x not in away.index:

        away[x]=0        

total=home+away

total.sort_values(ascending=False).to_frame(name="Number of Matches").style.background_gradient(cmap="icefire")
wins=df["winner"].value_counts().iloc[0:]

for x in total.index:

    if x not in wins.index:

        wins[x]=0

total_wins=pd.concat([total,wins],axis=1).rename(columns={0:"Matches","winner":"Wins"})

total_wins.sort_values("Wins",ascending=False).style.bar(color="orange",subset="Matches")
total_wins["losses"]=df["loser"].value_counts()

total_wins["draws"]=total_wins["Matches"]-total_wins["losses"]-total_wins["Wins"]
ax=total_wins.loc["Germany"][1:].plot.pie(explode=[0.1,0.1,0.1],shadow=True,cmap="winter",autopct="%.1f%%")

ax.set_ylabel("Results for Germany")

plt.figure(figsize=(12,6))

ax=total_wins.sort_values("Wins",ascending=False).iloc[:20,1].plot.bar(color="red")

ax.legend("Number of Wins")
temp=df[((df["home_team"]=="Germany") | (df["away_team"]=="Germany")) & ((df["home_team"]=="England") | (df["away_team"]=="England"))]["winner"].value_counts()

temp["Draws"]=df[((df["home_team"]=="Germany") | (df["away_team"]=="Germany")) & ((df["home_team"]=="England") | (df["away_team"]=="England"))]["draw"].value_counts().loc[True]

ax=temp.plot.pie(shadow=True,autopct="%.1f%%",cmap="viridis")

ax.set_ylabel("Results")
df["date"]=pd.to_datetime(df["date"],infer_datetime_format=True)

df["year"]=df["date"].apply(lambda x: x.year)

df["month"]=df["date"].apply(lambda x: x.month)

df["day"]=df["date"].apply(lambda x: x.day)
df["tournament"].value_counts()
df.head()
import plotly as pt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)
data=dict(type="choropleth",

         locations=total_wins.index,

         locationmode="country names",

          colorscale="electric",

          z=total_wins["Wins"],

          colorbar={"title":"Number of Wins"},

          text="wins"

         )

lay=dict(title="Country-wise wins",geo=dict(scope="world"))

ma=go.Figure(data=[data],layout=lay)

iplot(ma)
total_wins=total_wins.sort_values("Wins",ascending=False)
sns.set_style("whitegrid")

sns.set(rc={'figure.figsize':(17,5)})

sns.pointplot(x=total_wins.index[:30],y=total_wins["Wins"][:30],center=True,color="red")

sns.pointplot(x=total_wins.index[:30],y=total_wins["Matches"][:30],center=True,color="green")

plt.xticks(rotation=70)

plt.legend(labels=["Wins","Matches Played"])

plt.title("Country-wise Wins")

plt.xlabel("Team")
total_wins.head()
plt.figure(figsize=(20,5))

sns.set_style("whitegrid")

temp=df.groupby("year")["winner"].value_counts()

temp.loc[:,"Germany"].plot(color="red",alpha=0.6)

temp.loc[:,"England"].plot(color="green",alpha=0.6)

plt.xlabel("Year")

plt.ylabel("Wins")

plt.title("England vs Germany")

plt.legend(["Germany","England"])
plt.figure(figsize=(15,5))

sns.set_style("whitegrid")

temp=df.groupby("month")["winner"].value_counts()

temp.loc[:,"Germany"].plot(color="blue",alpha=0.6,marker="x",markersize=9,markeredgecolor="red")

temp.loc[:,"England"].plot(color="orange",alpha=0.6,marker="v",markersize=9,markerfacecolor="green",markeredgecolor="green")

plt.xlabel("Year")

plt.ylabel("Wins")

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],["January","February","March","April","May","June","July","August","September","October","November","December"])

plt.title("England vs Germany")

plt.legend(["Germany","England"])
plt.figure(figsize=(10,6))

df.groupby("year")["home_team"].count()[:-1].plot.area(alpha=0.6,color="purple")

plt.xlabel("Year")

plt.ylabel("Number of matches played")

plt.title("Growth of Football")
plt.plot(df["city"].value_counts()[:15],"g-x",markersize=9,markeredgecolor="blue")

plt.xticks(rotation=60)
temp=df[(df["home_team"]=="Germany") | (df["home_team"]=="England") | (df["home_team"]=="Brazil") |

       (df["home_team"]=="France") | (df["home_team"]=="Sweden") | (df["home_team"]=="Argentina") |

       (df["home_team"]=="Portugal") | (df["home_team"]=="India") | (df["home_team"]=="Pakistan") |(df["home_team"]=="Belgium")]

plt.figure(figsize=(17,6))

sns.boxenplot(x=temp["home_team"],y=temp["home_score"],scale="linear",palette="rainbow")

plt.xlabel("Home Teams")

plt.ylabel("Goals scored")

plt.title("Average goals scored in home games")

#df[df["home_team"]=="Germany"]["home_score"].plot.box(showfliers=True,cmap="winter",patch_artist=True)
ser=df.groupby(["tournament","year"])["country"].value_counts()["FIFA World Cup"]

temp=pd.concat([ser.index.get_level_values(1).value_counts(),df[df["tournament"]=="FIFA World Cup"].groupby("country")["year"].unique()],axis=1)

data=dict(type="choropleth",

          locations=temp.index,

          locationmode="country names",

          z=temp["country"],

          colorbar={"title":"Number of World Cups"},

          text=temp["year"]

         )

layout=dict(title="Number of World Cups hosted",geo={"scope":"world"})

ma=go.Figure(data=[data],layout=layout)

iplot(ma)
df[(df["country"]=="United States") & (df["tournament"]=="FIFA World Cup")]
df["goal_diff"]=df["home_score"]-df["away_score"]
temp=df[(df["month"]==8) & (df["day"]==31)]

temp[temp["goal_diff"] == temp["goal_diff"].max()]
temp=df.groupby("tournament")["home_team"].count().sort_values(ascending=False)[:20]

data=dict(type="pie",

          labels=temp.index,

          values=temp,

          hole=0.3,

          name="Tournaments",

          hoverinfo="label+percent+value"

         )

layout=dict(title="Most Played Tournaments")

iplot(go.Figure(data=[data],layout=layout))
df[df["home_team"]=="Germany"][["away_team","home_score","away_score","goal_diff"]].sort_values(["goal_diff","home_score","away_score"],ascending=(True,True,False))[:10]
df[df["home_team"]=="Germany"][["away_team","home_score","away_score","goal_diff"]].sort_values(["goal_diff","home_score","away_score"],ascending=(False,False,True))[:10]
df.corr()
sns.heatmap(df.corr(),cmap="winter",annot=True)
temp2=pd.DataFrame(columns=["away_team","home_score","away_score","goal_diff"])

for team in df["home_team"].unique()[:308]:

    temp2.loc[team]=df[df["home_team"] == team][["away_team","home_score","away_score","goal_diff"]].sort_values(["goal_diff","home_score","away_score"],ascending=(True,True,False)).iloc[0]

temp2=temp2.rename(columns={"away_team":"opp","home_score":"team_score","away_score":"opp_score"})    
temp3=pd.DataFrame(columns=["home_team","away_score","home_score","goal_diff"])

for team in df["away_team"].unique():

    temp3.loc[team]=df[df["away_team"] == team][["home_team","away_score","home_score","goal_diff"]].sort_values(["goal_diff","home_score","away_score"],ascending=(False,False,True)).iloc[0]

temp3["goal_diff"]=-temp3["goal_diff"] 

temp3=temp3.rename(columns={"home_team":"opp","away_score":"team_score","home_score":"opp_score"}) 
loss=pd.DataFrame(columns=["opp","team_score","opp_score","goal_diff","text"])

for team in temp3.index:

    if team in temp2.index:

        if temp2.loc[team]["goal_diff"] > temp3.loc[team]["goal_diff"]:

            loss.loc[team]=temp3.loc[team]

        else:   

            loss.loc[team]=temp2.loc[team] 

    else:

        loss.loc[team]=temp3.loc[team] 

for team in temp2.index:

    if team in temp3.index:

        if temp2.loc[team]["goal_diff"] > temp3.loc[team]["goal_diff"]:

            loss.loc[team]=temp3.loc[team]

        else:   

            loss.loc[team]=temp2.loc[team]    

    else:

        loss.loc[team]=temp2.loc[team]         

            

            

for x in range(0,loss.shape[0]):

    loss.iloc[x,4]=str(loss.iloc[x,1])+"-"+str(loss.iloc[x,2])+" against "+loss.iloc[x,0]

data=dict(type="choropleth",

          locations=loss.index,

          locationmode="country names",

          z=loss["goal_diff"],

          text=loss["text"],

          hoverinfo="text+location",

          colorbar={"title":"Goal Difference"},

          colorscale="viridis"

         )

lay=dict(title="Greatest Losses",geo={"scope":"world"})

iplot(go.Figure(data=[data],layout=lay))
temp2=pd.DataFrame(columns=["away_team","home_score","away_score","goal_diff"])

for team in df["home_team"].unique()[:308]:

    temp2.loc[team]=df[df["home_team"] == team][["away_team","home_score","away_score","goal_diff"]].sort_values(["goal_diff","home_score","away_score"],ascending=((False,False,True))).iloc[0]

temp2=temp2.rename(columns={"away_team":"opp","home_score":"team_score","away_score":"opp_score"})



temp3=pd.DataFrame(columns=["home_team","away_score","home_score","goal_diff"])

for team in df["away_team"].unique():

    temp3.loc[team]=df[df["away_team"] == team][["home_team","away_score","home_score","goal_diff"]].sort_values(["goal_diff","home_score","away_score"],ascending=(True,True,False)).iloc[0]

temp3["goal_diff"]=-temp3["goal_diff"] 

temp3=temp3.rename(columns={"home_team":"opp","away_score":"team_score","home_score":"opp_score"}) 

win=pd.DataFrame(columns=["opp","team_score","opp_score","goal_diff","text"])

for team in temp3.index:

    if team in temp2.index:

        if temp2.loc[team]["goal_diff"] < temp3.loc[team]["goal_diff"]:

            win.loc[team]=temp3.loc[team]

        else:   

            win.loc[team]=temp2.loc[team] 

    else:

        win.loc[team]=temp3.loc[team] 

for team in temp2.index:

    if team in temp3.index:

        if temp2.loc[team]["goal_diff"] < temp3.loc[team]["goal_diff"]:

            win.loc[team]=temp3.loc[team]

        else:   

            win.loc[team]=temp2.loc[team]    

    else:

        win.loc[team]=temp2.loc[team]         

            

            

for x in range(0,loss.shape[0]):

    win.iloc[x,4]=str(win.iloc[x,1])+"-"+str(win.iloc[x,2])+" against "+win.iloc[x,0]
data=dict(type="choropleth",

          locations=win.index,

          locationmode="country names",

          z=win["goal_diff"],

          text=win["text"],

          hoverinfo="text+location",

          colorbar={"title":"Goal Difference"},

          colorscale="viridis_r"

         )

lay=dict(title="Greatest Wins",geo={"scope":"world"})

iplot(go.Figure(data=[data],layout=lay))