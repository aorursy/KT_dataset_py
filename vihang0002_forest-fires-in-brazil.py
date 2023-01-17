# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv",encoding= 'ISO-8859-1')

df1
df1.describe()
x=np.array(df1["state"].unique())

x
y=np.array(df1.year.unique())

y
z=[]

for i in range(len(y)):

    z.append(df1[df1.year==y[i]].number.sum())

z   

    
plt.plot(y,z)

plt.xlabel('Year') 



plt.ylabel('Number of fires') 

plt.title("Forest fires betweent 1998-2007")

 

plt.show() 
v=[]

for i in range(len(x)):

    v.append(df1[df1.state==x[i]].number.sum())

v
plt.plot(x,v)

plt.xlabel('State') 



plt.ylabel('Number of fires') 

plt.title("State wise distribution of Forest fires")

plt.xticks(rotation=90)

plt.show() 
import plotly.graph_objects as go
labels=x

sizes=v

fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])

fig.update_layout(title_text="Total fires shared between states")

fig.show()
u=df1.month.unique()

u
q=[]

for i in range(len(u)):

    df2=df1[df1.month==u[i]]

    q.append(df2.number.sum())

q
plt.plot(u,q)

plt.xlabel('Month') 



plt.ylabel('Number of fires') 

plt.title("Month wise distribution of Forest fires")

plt.xticks(rotation=90)

plt.show() 
labels=u

sizes=q

fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])

fig.update_layout(title_text="Month wise distribution of fires")

fig.show()