import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
FootballData = pd.read_csv("../input/top250-00-19.csv")
FootballData.head()
FootballData.drop("Market_value",axis=1,inplace=True)
FootballData = FootballData.head(20)
FootballData.Transfer_fee = [float(i)/sum(FootballData.Transfer_fee) for i in FootballData.Transfer_fee]

FootballData.Age = [float(i)/sum(FootballData.Age) for i in FootballData.Age]
f,ax = plt.subplots(figsize=(16,8))

plt.xticks(rotation = 60)

sns.pointplot(x = "Name",y="Age",data=FootballData,color="black")

sns.pointplot(x = "Name",y="Transfer_fee",data=FootballData,color="red")

plt.text(15,0.1,'Age',color='black',fontsize = 10,style = 'italic')

plt.text(15,0.103,'Transfer Fee',color='red',fontsize = 10,style = 'italic')

plt.xlabel("Football Player Names")

plt.ylabel("Transfer Fee vs. Age")

plt.show()
sns.countplot(FootballData[:20].League_to)

plt.xlabel("League Names",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.show()
sns.countplot(x='Team_from',data=FootballData)

plt.xticks(rotation=90)

plt.xlabel("Team From")

plt.ylabel("Count")

plt.show()
labels = FootballData[:15].League_to.value_counts().index

colors = ["blue","red","green","yellow"]

explode = [0,0,0,0]

sizes = FootballData[:15].League_to.value_counts().values

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Transfers Leagues to',color = 'blue',fontsize = 15)

plt.show()
sns.swarmplot(x="Position",y="Transfer_fee",hue="League_from",data=FootballData)

plt.xticks(rotation=15)

plt.title("Swarm Plot for Transfer Fee and Position",fontsize=12)

plt.xlabel("Position",fontsize=10)

plt.ylabel("Transfer Fee",fontsize=10)

plt.show()
# trace1 is line plot

# go: graph object

trace1 = go.Scatter(

    x=FootballData.index,

    y=FootballData.Age,

    mode = "markers",

    xaxis='x2',

    yaxis='y2',

    name = "Footballer Age",

    marker = dict(color = 'rgba(0, 112, 20, 0.8)'),

)



# trace2 is histogram

trace2 = go.Histogram(

    x=FootballData.Age,

    opacity=0.75,

    name = "Footballer Age",

    marker=dict(color='rgba(10, 200, 250, 0.6)'))



# add trace1 and trace2

data = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.7, 1],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = ' Footballer Age Histogram and Scatter Plot'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
plt.rcdefaults()

fig, ax = plt.subplots()



y_pos = np.arange(len(FootballData.Team_from))

performance = 3 + 10 * np.random.rand(len(FootballData.Team_from))

error = np.random.rand(len(FootballData.Team_from))



ax.barh(y_pos, performance, xerr=error, align='center',

        color = "blue", ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(FootballData.Team_from)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Count')

plt.show()