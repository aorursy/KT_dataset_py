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
import plotly.express as px
df = px.data.election()
fig = px.scatter_ternary(df, a="Joly", b="Coderre", c="Bergeron")
fig.show()
import plotly.graph_objects as go
rawData = [
    {'$':0.48502,'Whcycle':1,'Wh':0.111,'label':'LiTO3'}, #' $11788/224644kWh $1000kWh 7000cycles 80Wh/g 0.011Wh/cycle 560000
    {'$':1,'Whcycle':0.232,'Wh':0.3616,'label':'LiNiCoAlO'}, #$24304/224644kWh $350kWh 500 260 0.7428 0.52 130000
    {'$':0.54707,'Whcycle':0.7857,'Wh':0.305,'label':'LiMgO'},#$13296/224644kWh $420 2000 220 0.523 0.11 440000
    
    {'$':0.8229,'Whcycle':0.533,'Wh':1,'label':'Gasoline'},# $20,000/224644kWh, 416 cycles 719W/g 1.72Wh/cycle 
    ];

def makeAxis(title, tickangle):
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 2,
      'showline': True,
      'showgrid': True
    }

fig = go.Figure(go.Scatterternary({
    'mode': 'markers',
    'a': [i for i in map(lambda x: x['Whcycle'], rawData)],
    'b': [i for i in map(lambda x: x['$'], rawData)],
    'c': [i for i in map(lambda x: x['Wh'], rawData)],
    'text': [i for i in map(lambda x: x['label'], rawData)],
    'marker': {
        'symbol': 100,
        'color': 'blue',
        'size': 14,
        'line': { 'width': 1 }
    }
}))

fig.update_layout({
    'ternary': {
        'sum': 10,
        'aaxis': makeAxis('How Long it lasts', 0),
        'baxis': makeAxis('<br>lifecycle cost', 0),
            'caxis': makeAxis('<br>Milelage', 0)
    },
    'annotations': [{
      'showarrow': False,
      'text': '',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
})

fig.show()
import plotly.figure_factory as ff
import numpy as np
Al = np.array([0. , 0. , 0., 0., 1./3, 1./3, 1./3, 2./3, 2./3, 1.])
Cu = np.array([0., 1./3, 2./3, 1., 0., 1./3, 2./3, 0., 1./3, 0.])
Y = 1 - Al - Cu
# synthetic data for mixing enthalpy
# See https://pycalphad.org/docs/latest/examples/TernaryExamples.html
enthalpy = 2.e6 * (Al - 0.01) * Cu * (Al - 0.52) * (Cu - 0.48) * (Y - 1)**2 - 5000
fig = ff.create_ternary_contour(np.array([Al, Y, Cu]), enthalpy,
                                pole_labels=['Al', 'Y', 'Cu'],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,
                                title='Mixing enthalpy of ternary alloy')
fig.show()
import json
import pandas as pd

contour_raw_data = pd.read_json('https://raw.githubusercontent.com/plotly/datasets/master/contour_data.json')
scatter_raw_data = pd.read_json('https://raw.githubusercontent.com/plotly/datasets/master/scatter_data.json')

scatter_data =  scatter_raw_data['Data']

def clean_data(data_in):
    """
    Cleans data in a format which can be conveniently
    used for drawing traces. Takes a dictionary as the
    input, and returns a list in the following format:

    input = {'key': ['a b c']}
    output = [key, [a, b, c]]
    """
    key = list(data_in.keys())[0]
    data_out = [key]
    for i in data_in[key]:
        data_out.append(list(map(float, i.split(' '))))

    return data_out


#Example:
print(clean_data({'L1': ['.03 0.5 0.47','0.4 0.5 0.1']}))

import plotly.graph_objects as go

a_list = []
b_list = []
c_list = []
text = []

for raw_data in scatter_data:
    data = clean_data(raw_data)
    text.append(data[0])
    c_list.append(data[1][0])
    a_list.append(data[1][1])
    b_list.append(data[1][2])

fig = go.Figure(go.Scatterternary(
  text=text,
  a=a_list,
  b=b_list,
  c=c_list,
  mode='markers',
  marker={'symbol': 100,
          'color': 'green',
          'size': 10},
))

fig.update_layout({
    'title': 'Ternary Scatter Plot',
    'ternary':
        {
        'sum':1,
        'aaxis':{'title': 'X', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'baxis':{'title': 'W', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
        'caxis':{'title': 'S', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }
    },
    'showlegend': False
})

fig.show()

import plotly.graph_objects as go


contour_dict = contour_raw_data['Data']

# Defining a colormap:
colors = ['#8dd3c7','#ffffb3','#bebada',
          '#fb8072','#80b1d3','#fdb462',
          '#b3de69','#fccde5','#d9d9d9',
          '#bc80bd']
colors_iterator = iter(colors)

fig = go.Figure()

for raw_data in contour_dict:
    data = clean_data(raw_data)

    a = [inner_data[0] for inner_data in data[1:]]
    a.append(data[1][0]) # Closing the loop

    b = [inner_data[1] for inner_data in data[1:]]
    b.append(data[1][1]) # Closing the loop

    c = [inner_data[2] for inner_data in data[1:]]
    c.append(data[1][2]) # Closing the loop

    fig.add_trace(go.Scatterternary(
        text = data[0],
        a=a, b=b, c=c, mode='lines',
        line=dict(color='#444', shape='spline'),
        fill='toself',
        fillcolor = colors_iterator.__next__()
    ))

fig.update_layout(title = 'Ternary Contour Plot')
fig.show()
