# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/tcl.csv",sep=";")
df.head()
df.dtypes
team_list = list(df.team.unique())

point_ratio = []



for i in team_list:

    x = df[df.team==i]

    point_rate = sum(x.points)/len(x)

    point_ratio.append(point_rate)

data = pd.DataFrame({'team_list': team_list,'point_ratio':point_ratio})

new_index = (data['point_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['team_list'], y=sorted_data['point_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Teams')

plt.ylabel('Points Rate')

plt.title('Points Rate of Teams')

plt.show()
won_ratio = []

drawn_ratio = []

lost_ratio = []



for i in team_list:

    x = df[df.team==i]

    won_rate = sum(x.won)/len(x)

    won_ratio.append(won_rate)

    drawn_rate = sum(x.drawn)/len(x)

    drawn_ratio.append(drawn_rate)

    lost_rate = sum(x.lost)/len(x)

    lost_ratio.append(lost_rate)

    

    

f,ax = plt.subplots(figsize = (9,10))

sns.barplot(x=won_ratio,y=team_list,color='green',alpha = 0.7,label='Won' )

sns.barplot(x=drawn_ratio,y=team_list,color='blue',alpha = 0.6,label='Drawn')

sns.barplot(x=lost_ratio,y=team_list,color='red',alpha = 0.5,label='Lost')





ax.legend(loc='lower right',frameon = True) 

ax.set(xlabel='Percentage of Won,Drawn and Lost', ylabel='Teams',title = "Percentage of Team's Won,Drawn and Lost Rates ")

won_ratio = []

drawn_ratio = []

lost_ratio = []



for i in team_list:

    x = df[df.team==i]

    won_rate = sum(x.won)/len(x)

    won_ratio.append(won_rate)

    drawn_rate = sum(x.drawn)/len(x)

    drawn_ratio.append(drawn_rate)

    lost_rate = sum(x.lost)/len(x)

    lost_ratio.append(lost_rate)



f,ax = plt.subplots(figsize = (8,8))

sns.pointplot(y=won_ratio,x=team_list,color='orange',alpha = 0.7,label='Won' )

sns.pointplot(y=drawn_ratio,x=team_list,color='red',alpha = 0.6,label='Drawn')

sns.pointplot(y=lost_ratio,x=team_list,color='blue',alpha = 0.5,label='Lost')

plt.text(0.1,15.6,'Won Ratio',color='orange',fontsize = 17,style = 'italic')

plt.text(0.1,13.6,'Drawn Ratio',color='red',fontsize = 18,style = 'italic')

plt.text(0.1,11.6,'Lost Ratio',color='blue',fontsize = 19,style = 'italic')

plt.xlabel('Teams',fontsize = 15,color='green')

plt.ylabel('Values',fontsize = 15,color='green')

plt.title('Won , Drawn and Lost Rate',fontsize = 20,color='blue')

plt.grid()  
df.team.dropna(inplace = True)

labels = df.team.value_counts().index

colors = ['Grey','Blue','Green','Red','Yellow']

explode = [0,0,0,0,0]

sizes = df.team.value_counts().values



#visual

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=10)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()


goals_for_ratio = []

goals_against_ratio = []

for i in team_list:

    x = df[df.team==i]

    goal_rate = sum(x.goals_for)/len(x)

    goals_for_ratio.append(goal_rate)

    against_rate = sum(x.goals_against)/len(x)

    goals_against_ratio.append(against_rate)

    



data2 = pd.DataFrame({'goals_for_ratio': goals_for_ratio,'goals_against_ratio':goals_against_ratio})

data2.head()
g= sns.jointplot(data2.goals_for_ratio, data2.goals_against_ratio,kind="kde", size=7)

plt.savefig('graph.png')

plt.legend()

plt.show()
sns.lmplot(x="goals_for_ratio", y="goals_against_ratio", data=data2) #F

plt.show()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data2.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.1f',ax=ax)

plt.show()