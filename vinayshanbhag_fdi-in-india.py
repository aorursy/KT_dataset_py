# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.offline as offline

offline.init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/FDI_in_India.csv')
years = ['2000', '2001', '2002', '2003', '2004', '2005',

       '2006', '2007', '2008', '2009', '2010', '2011',

       '2012', '2013', '2014', '2015', '2016']

df.columns= ['Sector']+years
annual_fdi = df[years].sum(axis=0)
annual_fdi[-4:]
trace = go.Scatter(

    x= annual_fdi.index[-4:],

    y=annual_fdi.values[-4:],

    mode = 'lines',

    name = 'FDI'

)



layout = go.Layout(

    title='FDI by year',

    xaxis=dict(

        title='Year',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='Total FDI',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    )

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#df['Sector'].unique()

df[df['Sector']=='METALLURGICAL INDUSTRIES'][years].values[0][-4:]
df.columns=['Sector']+years

data = []

sectors = df[['Sector','2016']].sort_values(by='2016',ascending=False)['Sector'][:5].values #df['Sector'].unique()

for sector in sectors:

    s = df[df['Sector']==sector]

    y_val = s[years].values[0][-4:]

    trace = go.Scatter(

        x= years[-4:],

        y=y_val,

        fill='tozeroy',

        name=sector,

        mode='lines'

    )

    data.append(trace)



layout = go.Layout(

    title='Annual FDI by sector',

    xaxis=dict(

        title='Year',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='FDI',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
def f(x, y, n):

    if x:return ((y/x)**(1/n))-1

    else: return np.nan

y_start = 2012

y_end = 2016

df['cagr'] = df[['Sector',str(y_start),str(y_end)]].apply(lambda x: f(x[str(y_start)],x[str(y_end)], y_end-y_start), axis=1)
df[['Sector','cagr']].sort_values(by='cagr',ascending=False)
t = df[['Sector','2016']]['2016'].sum()



trace = go.Pie(labels=df['Sector'].values, values=((df['2016']/t)*100).values)

layout = go.Layout(showlegend=False)

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df[['Sector','2016']].sort_values(by='2016',ascending=False)['Sector'][:5].values