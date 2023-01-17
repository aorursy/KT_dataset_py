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
#Loading the dataset



df = pd.read_csv("/kaggle/input/indian-food-101/indian_food.csv")

df
df.info()
#checking null values



df.isna().sum()
# getting column names



print(list(df.columns))
# vegeterian and nonveg dishes count



count = list(df['diet'].value_counts())

count
import matplotlib.pyplot as plt

x=['vegeterian','Non vegeterian']

y = count

fig = plt.figure(figsize = (15, 5)) 

  

plt.bar(x, y, color ='pink',width = 0.3) 

  

plt.xlabel("Vegeterian or Not vegeterian") 

plt.ylabel("Number of Dishes") 



plt.title("Classification") 

plt.show() 
import re

import spacy

nlp = spacy.load('en')



def normalize(msg):

    

    msg = re.sub('[^A-Za-z]+', ' ', msg) #remove special character and intergers

    doc = nlp(msg)

    res=[]

    for token in doc:

        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #word filteration

            pass

        else:

            res.append(token.lemma_.lower())

    return res



df["ingredients"] = df["ingredients"].apply(normalize)

df.head()
# words count

from collections import Counter

words_collection = Counter([item for subtext in df['ingredients'] for item in subtext])

most_common = pd.DataFrame(words_collection.most_common(20))

most_common.columns = ['most_common_word','count']

most_common
import plotly.express as px 

  

fig = px.sunburst(most_common, path=['most_common_word'],values='count',color ='count')

fig.show()
# 3D Scatter Plot



import plotly.express as px





fig = px.scatter_3d(df, x='name', y='prep_time', z='cook_time', color='name')

fig.show()
flavour_count = list(df['flavor_profile'].value_counts())

flavour_count
flavours = list(df.flavor_profile.unique())

flavours
import numpy as np 

import plotly 

import plotly.graph_objects as go 

import plotly.offline as pyo 

from plotly.offline import init_notebook_mode 

  

init_notebook_mode(connected=True) 

  

# generating 150 random integers 

# from 1 to 50 

x = flavours

  



y = flavour_count

  

# plotting scatter plot 

fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 

        color=np.random.randn(10), 

        colorscale='Viridis',  

        showscale=True

    ) )) 

  

fig.show() 
course_count = list(df['flavor_profile'].value_counts())

course_count

  

course = list(df.course.unique())

course
# Donut chart



import plotly.graph_objects as go



labels = course

values = course_count



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
def ingredients_count(msg):

    

   

    

    msg_len = len(msg)

    

    return msg_len
df['ing_count'] = df['ingredients'].apply(ingredients_count)

df.head()
fig = go.Figure(data=go.Scatter(x=df['name'],

                                y=df['ing_count'],

                                mode='markers',

                                marker_color=df['ing_count'],

                                text=df['name'])) # hover text goes here



fig.update_layout(title='dish vs ingredients')

fig.show()
# Heat Map



import seaborn as sns

import matplotlib.pyplot as plt

correlation = df.corr()

plt.figure(figsize = (12 , 12))

sns.heatmap(correlation)
state_count = list(df['state'].value_counts())

state_count
state = list(df.state.unique())

state
import plotly.graph_objects as go

import numpy as np



fig = go.Figure(data=go.Scatter(

    x = state,

    y = state_count,

    mode='markers',

    marker=dict(

        size=16,

        color=np.random.randn(500), #set color equal to a variable

        colorscale='Viridis', # one of plotly colorscales

        showscale=True

    )

))



fig.show()