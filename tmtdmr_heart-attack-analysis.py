# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objs as go

from plotly import tools



import plotly.plotly as py

from plotly.plotly import iplot





# plotly

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





# word cloud library

from wordcloud import WordCloud







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv",sep=",") 

data=data.sort_values(by=['age']) # we sort values according to age

data["genderText"]  = ["male" if 1 == each else "female" for each in data.sex] # adding new column which has text variables. 
data.info()
data.head()
# corelation map

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt='.2f', ax=ax)
# histogram (frequency of Happiness Score)

plt.hist(data.age, bins=50)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("histogram")

plt.show()
trace1 = go.Scatter(

                    x = data.age,

                    y = data.trestbps,

                    mode = "lines",

                    name = "trestbps",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= data.genderText)

# Creating trace2

trace2 = go.Scatter(

                    x = data.age,

                    y = data.chol,

                    mode = "lines+markers",

                    name = "chol",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= data.genderText)



data2 = [trace1, trace2]

layout = dict(title = 'trestbps and chol accoding to age',

              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False)

             )

fig = dict(data = data2, layout = layout)

iplot(fig)
# %% filtering and joint plot

dataFilter1 =data[data.target==1]



dataFilter0 =data[data.target==0]



g = sns.jointplot(dataFilter1.age, dataFilter1.trestbps, kind="kde", size=7)

#plt.savefig('graph.png')

plt.show()

# %% Violin plot - comparing male and female thalach values on Target = 1  

dataFilterMale = data[data.sex == 1] # both target 1 and male

dataFilterFemale = data[data.sex == 0] # both targer 1 and female



MaleThalach = pd.DataFrame(dataFilterMale.thalach)

FemaleThalach = pd.DataFrame(dataFilterFemale.thalach)

FemaleThalach.index = range(1,97,1) # index stars from 1

MaleThalach.index = range(1,208,1) # index stars from 1



dfMaleThalach = pd.DataFrame(MaleThalach).iloc[0:96,:] # we take only 96 row

dfFemaleThalach = pd.DataFrame(FemaleThalach)

unifiedThalach = pd.concat([dfMaleThalach, dfFemaleThalach], axis=1)

unifiedThalach.columns = ['Male thalach','Female thalach']  # we renama columns name



pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=unifiedThalach, palette=pal, inner="points")

plt.show()
trace0 = go.Box(

    y=data.trestbps,

    name = 'trestbps',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=data.chol,

    name = 'chol',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

fig = [trace0, trace1]

iplot(fig)