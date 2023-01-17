import pandas as pd
import numpy as np
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))
# load data

df = pd.read_csv('../input/diet_data.csv')

df.head()
df.std()
# drop all rows contain empty value
df = df.dropna()
# display calories base on date
data = [go.Bar(x=df.Date, y=df.calories)]
iplot(data, filename='jupyter-basic_bar')
# display line chart: activity effect on weight changes?
change_on_date = go.Scatter(y=df.change, x=df.Date, mode='lines', name="change")
walk_on_date = go.Scatter(x=df.Date, y=df.walk, mode='lines+markers', name='walk')
run_on_date= go.Scatter(x=df.Date, y=df.run, mode='markers', name = 'run')
data = [change_on_date, walk_on_date, run_on_date]

iplot(data, filename='line-mode')
# weight base exercise with weight change? _subplot_
change_on_date = go.Scatter(y=df.change, x=df.Date, mode='lines', name="weight change")
weight_on_date = go.Scatter(x=df.Date, y=df.weight, mode='lines', name='weight exercise')

# two plot
fig = tools.make_subplots(rows=3, cols=1)
fig.append_trace(change_on_date, 1, 1)
fig.append_trace(weight_on_date, 3, 1)

iplot(fig, filename='stacked-subplots')
# daily calurious
date_with_calories = go.Scatter(x=df.Date, y=df.calories, mode='lines', name="weight change vs calories")
data1 = [date_with_calories]
iplot(data1, filename='line-mode')
# try box plots
# summary for calories throught time

trace = go.Box(y=df.calories, name ='calories', marker =  dict(color='rgb(214,12,140)'))
trace2 =  go.Box(y=df.change, name= 'weight change', marker = dict(color='rgb(121,23,110)'))

fig2 = tools.make_subplots(rows=1, cols=2)
fig2.append_trace(trace, 1, 1)
fig2.append_trace(trace2, 1, 2)

iplot(fig2, filename='stacked-subplots')

print('Average calories throught time is 2575. The hight one is 9150 and lowers one is1400.')
print('Overall, the weight does not change a lot thought time. only lose 3 lb. ')
# histogoram 
# using cals_per_oz column's value.
print('basic histogram base on cals_per_oz')
data1 = [go.Histogram(x=df.cals_per_oz)]
iplot(data1, filename='basic histogram')
# histogoram 
# using cals_per_oz column's value.
print('normalized histogram base on cols_per_oz')
data2 = [go.Histogram(x=df.cals_per_oz, histnorm='probability')]
iplot(data2, filename='normalized histogram')

# pie chart
# coluns: five_donuts, walk, run, wine, prot, weight
print('All % of activites through the time')
labels = ['five_donuts', 'walk', 'run', 'wine', 'prot', 'weight']
values= [df.five_donuts.sum(), df.walk.sum(), df.run.sum(), df.wine.sum(), df.prot.sum(), df.weight.sum()]
fig3 = go.Pie(labels=labels, values=values)
iplot([fig3], filename='basic_pie_chart')

