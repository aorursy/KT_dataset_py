# Importing libraries

import numpy as np

import pandas as pd



# plotly packages

import plotly

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)

cf.go_offline()
# Let's create a random data with numpy

df = pd.DataFrame(np.random.randn(200, 4), columns='A B C D'.split())



# check the data once

df.head()
# Let's create another data frame

df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 50, 42]})
# Plotting randomly with matplotlib

df.plot()
df.iplot()
# Scatter plot

df.iplot(kind='scatter', x='A', y='B')
df.iplot(kind = 'scatter', x='A', y='B', mode='markers')
# Bar plots

# for a bar plots the data should be bivariant(categorical, numeerical)

df2.iplot(kind='bar', x='Category', y='Values')
# we can also do some interesting things with plotlt

# let's see

print(df.iplot(kind='bar'))
# count each column in df dataframe and plot bar graphs

print(df.count().iplot(kind='bar'))
# Now take sum of each column in df and plot bar graph

print(df.sum().iplot(kind='bar'))
# Box plots

df.iplot(kind='box')
# 3D surface plot

# Let's create a another random datframe

df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[500,400,300,200,100]})

df4 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[1,2,3,4,5]})
# This is for 3 Dimensional values

df3
df3.iplot(kind='surface')
df4.iplot(kind='surface', colorscale='rdylbu') #rd- red, yl-yellow, bu-blue
# Histogram plot

df['A'].iplot(kind='hist', bins=25)
# We can plot all the columns in df in hist

# On the right side labels, we can on and off any partiucal column

df.iplot(kind='hist')
# Spread type visualization

# stocks type

df[['A', 'B']].iplot(kind='spread')
# Bubble plots

# which is similar to scatter plot

df.iplot(kind='bubble', x='A', y='B', size='C')
# Scatter matrix plot

# Which is similar to pairplots

df.scatter_matrix()