import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import plotly

import plotly.graph_objs as go

# Tells plotly to activate the offline, notebook mode

plotly.offline.init_notebook_mode(connected=True)
train_df = pd.read_csv('../input/train.csv')
# Try other columns to experiment

X_COL = "var_81"

Y_COL = "var_68"

Z_COL = "var_108"

HUE_COL = "target"

N_SAMPLES = 10000 # A very large value isn't recommended.

df = train_df.sample(N_SAMPLES)
# The 3D scatter plot 

trace = go.Scatter3d(

    x=df[X_COL],

    y=df[Y_COL],

    z=df[Z_COL],

    mode='markers',

    marker=dict(

        size=12,

        color=df[HUE_COL],            

        opacity=0.5,

        showscale=True,

        colorscale=[[0.0, 'red'], [1.0, 'blue']]

        

    ),

)



# How it should look

layout = go.Layout(

    width=600, # Maybe someone could find better width and height parameters?

    height=600,

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    scene = dict(    

        xaxis = dict(

            title=X_COL),

        yaxis = dict(

            title=Y_COL),

        zaxis = dict(

            title=Z_COL),

    ),

)

# The object to plot

fig = go.Figure(data=[trace], layout=layout)
# The result

plotly.offline.iplot(fig)