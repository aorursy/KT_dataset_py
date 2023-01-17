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
# Loading the dataset 

data = pd.read_csv("/kaggle/input/stackindex/MLTollsStackOverflow.csv")
print(data)
data = data.drop(columns=["Tableau"])
print(data)
data.info()
# Checking null values

print(data.isnull().sum())
# Getting column names

print(list(data.columns))
# Converting column names into lower form

columns = data.columns
columns = columns.str.lower()
print(list(columns))
# Splitting the month column

data[['year','months']] = data.month.str.split("-",expand=True)
print(data['year'], data['months'])
# Print new dataset

print(data.head())
# Operations for visualization

import matplotlib.pyplot as plt
included =  data.drop(columns=["month","year","months"])
print(included.head())
# Operations for visualization

question_count  = included.sum(axis = 0)
top_keys = question_count.sort_values(ascending = False)[0:11]
print(list(top_keys))
# Pie Chart

import matplotlib.pyplot as plt
  
# Creating plot 
#fig = plt.figure(figsize =(200, 7)) 
plt.pie(top_keys, labels = top_keys.index, radius=3) 
  
# show plot 
plt.show() 
# Grouping data by year

count = data.groupby(['year']).count()
print(count)
# Multiple Bar Plots
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(20, 10)) 

# Make the plot 
plt.bar(data["year"] , data["python"], color ='r', width = barWidth, edgecolor ='grey', label ='Python')
plt.bar(data["year"], data["r"], color = 'g', width = barWidth, edgecolor ='grey', label ='R') 
plt.bar(data["year"], data["machine-learning"], color ='b', width = barWidth, edgecolor ='grey', label ='ML') 
   
# Adding Xticks  
plt.xlabel('Year', fontweight ='bold') 
plt.ylabel('Tech-Keys', fontweight ='bold') 

plt.show()
# 3D Scatter Plot

import plotly.express as px

a = ['r', 'python']
fig = px.scatter_3d(data, x='year', y='months', z='numpy', color='year')
fig.show()
# Heat Map

import seaborn as sns
correlation = data.corr()
plt.figure(figsize = (12 , 12))
sns.heatmap(correlation)
# Donut chart

import plotly.graph_objects as go

labels = list(top_keys.index)
values = list(top_keys)

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()
import numpy as np 
import plotly 
import plotly.graph_objects as go 
import plotly.offline as pyo 
from plotly.offline import init_notebook_mode 
  
init_notebook_mode(connected=True) 
  
# generating 150 random integers 
# from 1 to 50 
x = list(top_keys.index)
  

y = list(top_keys)
  
# plotting scatter plot 
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 
        color=np.random.randn(20), 
        colorscale='Viridis',  
        showscale=True
    ) )) 
  
fig.show() 
data.head()

import plotly.express as px 
  
fig = px.sunburst(data, path=['year'],values='python',color ='python')
fig.show()