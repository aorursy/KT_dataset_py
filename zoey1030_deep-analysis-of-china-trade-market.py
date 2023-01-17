# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/commodity_trade_statistics_data.csv', low_memory=False)
cn_df = df[df.country_or_area == "China"]
df.head()
cn_i = cn_df[(cn_df.flow == 'Import') & (cn_df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')
cn_e = cn_df[(cn_df.flow == 'Export') & (cn_df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

trace1 = go.Bar(
                x = cn_i.year,
                y = cn_i.trade_usd,
                name = "China Import",
                marker = dict(color = 'rgba(102, 216, 137, 0.8)'),
)

trace2 = go.Bar(
               x = cn_e.year,
                y = cn_e.trade_usd,
                name = "China Export",
                marker = dict(color = 'rgba(224, 148, 215, 0.8)'),
)
data = [trace1, trace2]
layout = {
    'xaxis': {'title': 'Year 1992-2016'},
    'yaxis': {'title': 'Trade of Import & Export in China (USD)'},
    'barmode': 'group',
    'title': 'Import and Export in China'
}
fig = go.Figure(data = data, layout = layout)
iplot(fig)
temp = cn_df[(cn_df.year==1992) & (cn_df.flow=='Import')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]
cn_1992i = temp.sort_values(by="trade_usd",  ascending=True)
trace1 = go.Bar(
                x = cn_1992i.trade_usd,
                y = cn_1992i.commodity,
                marker = dict(color = 'rgba(152, 213, 245, 0.8)'),
                orientation = 'h'
)


data = [trace1]
layout = {
#     'xaxis': {'title': 'Trade in USD'},
    'yaxis': {'automargin':True,},
    'title': "Top 10 Commodities in China Import Trade (USD), 1992"
}
fig = go.Figure(data = data, layout = layout)
iplot(fig)

temp1 = cn_df[(cn_df.year==2016) & (cn_df.flow=='Import')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]
cn_2016i = temp1.sort_values(by="trade_usd",  ascending=True)
trace1 = go.Bar(
                x = cn_2016i.trade_usd,
                y = cn_2016i.commodity.tolist(),
                marker = dict(color = 'rgba(249, 205, 190, 0.8)'),
                orientation = 'h'
)


data = [trace1]
layout = {
#     'xaxis': {'title': 'Trade in USD'},
    'yaxis': {'automargin':True,},
    'title': "Top 10 Commodities in China Import Trade (USD), 2016"
}
fig = go.Figure(data = data, layout = layout)
iplot(fig)
petro = cn_df[(cn_df.comm_code == '270900') & (cn_df.flow == 'Import')]
urea = cn_df[(cn_df.comm_code == '310210') & (cn_df.flow == 'Import')]
iron = cn_df[(cn_df.comm_code == '260111') & (cn_df.flow == 'Import')]

trace1 = go.Scatter(
    x = petro.year,
    y = petro.trade_usd,
    mode = "lines+markers",
    name = "Petroleum oils",
    marker = dict(color = 'rgba(255, 196, 100, 0.8)')
)

trace2 = go.Scatter(
    x = urea.year,
    y = urea.trade_usd,
    mode = "lines+markers",
    name = "Urea",
    marker = dict(color = 'rgba(241, 130, 133, 0.8)')
)

trace3 = go.Scatter(
    x = iron.year,
    y = iron.trade_usd,
    mode = "lines+markers",
    name = "Iron ore",
    marker = dict(color = 'rgba(130, 241, 140, 0.8)')
)


data = [trace1, trace2, trace3]
layout = dict(title = "Some Commodities' value in China Import Trade (USD)",
              xaxis= dict(title= 'Year 1992-2016',ticklen= 5,zeroline= False),
              yaxis = {'title': 'Import trade value(USD)'}
 )
fig = dict(data = data, layout = layout)
iplot(fig)
temp = cn_df[(cn_df.year==1992) & (cn_df.flow=='Export')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]
cn_1992e = temp.sort_values(by="trade_usd",  ascending=True)
trace1 = go.Bar(
                x = cn_1992e.trade_usd,
                y = cn_1992e.commodity,
                marker = dict(color = 'rgba(173, 164, 239, 0.8)'),
                orientation = 'h'
)


data = [trace1]
layout = {
#     'xaxis': {'title': 'Trade in USD'},
    'yaxis': {'automargin':True,},
    'title': "Top 10 Commodities in China Export Trade (USD), 1992"
}
fig = go.Figure(data = data, layout = layout)
iplot(fig)

temp1 = cn_df[(cn_df.year==2016) & (cn_df.flow=='Export')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]
cn_2016e = temp1.sort_values(by="trade_usd",  ascending=True)
trace1 = go.Bar(
                x = cn_2016e.trade_usd,
                y = cn_2016e.commodity,
                marker = dict(color = 'rgba(255, 241, 117, 0.8)'),
                orientation = 'h'
)


data = [trace1]
layout = {
#     'xaxis': {'title': 'Trade in USD'},
    'yaxis': {'automargin':True,},
    'title': "Top 10 Commodities in China Export Trade (USD), 2016"
}
fig = go.Figure(data = data, layout = layout)
iplot(fig)
toys = cn_df[(cn_df.comm_code == '950390') & (cn_df.flow == 'Export')]
maize = cn_df[(cn_df.comm_code == '100590') & (cn_df.flow == 'Export')]
light = cn_df[(cn_df.comm_code == '940540') & (cn_df.flow == 'Export')]

trace1 = go.Scatter(
    x = toys.year,
    y = toys.trade_usd,
    mode = "lines+markers",
    name = "toys",
    marker = dict(color = 'rgba(255, 196, 100, 0.8)')
)

trace2 = go.Scatter(
    x = maize.year,
    y = maize.trade_usd,
    mode = "lines+markers",
    name = "maize",
    marker = dict(color = 'rgba(241, 130, 133, 0.8)')
)

trace3 = go.Scatter(
    x = light.year,
    y = light.trade_usd,
    mode = "lines+markers",
    name = "light",
    marker = dict(color = 'rgba(130, 241, 140, 0.8)')
)


data = [trace1, trace2, trace3]
layout = dict(title = "Some Commodities' value in China Export Trade (USD)",
              xaxis= dict(title= 'Year 1992-2016',ticklen= 5,zeroline= False),
              yaxis = {'title': 'Export trade value(USD)'}
 )
fig = dict(data = data, layout = layout)
iplot(fig)
cn_trade = cn_df[cn_df.comm_code!= 'TOTAL'].groupby(['year'],as_index=False)['trade_usd'].agg('sum')
wd_trade = df[(df.year >1991) & (df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')
# cn_trade.shape

trace0 = {
    'x': cn_trade.year,
    'y': cn_trade.trade_usd,
    'name': "China",
    'type': 'bar',
    'marker': {'color':'rgba(129, 239, 208, 0.8)'}
}
trace1 = {
    'x': wd_trade.year,
    'y': wd_trade.trade_usd,
    'name': "World",
    'type': 'bar',
    'marker': {'color':'rgba(255, 171, 202, 0.8)'}
}

data = [trace0, trace1]
layout = {
    'xaxis': {'title': 'Year 1992-2016'},
    'yaxis': {'title': 'Value of Trade in USD'},
    'barmode': 'relative',
    'title': 'World vs China: Value of Trade'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)

# ratio
trace3 = go.Scatter(
                    x = cn_trade.year,
                    y = cn_trade.trade_usd/wd_trade.trade_usd*100,
                    mode = "lines+markers",
                    name = "Ratio of China/World",
                    marker = dict(color = 'rgba(245, 150, 104, 0.8)')
)
data2 = [trace3]
layout2 = dict(title = 'Percentage of China Trade in World Trade (%)',
              xaxis= dict(title= 'Year 1992-2016',ticklen= 5,zeroline= False),
              yaxis = {'title': 'Percentage (%)'}
 )
fig2 = dict(data = data2, layout = layout2)
iplot(fig2)
cn_ie = cn_df[cn_df.comm_code!= 'TOTAL'].groupby(['year'],as_index=False)['weight_kg'].agg('sum')
wd_ie = df[(df.year >1991) & (df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['weight_kg'].agg('sum')

trace1 = go.Bar(
                x = wd_ie.year,
                y = wd_ie.weight_kg,
                name = "World",
                marker = dict(color = 'rgba(104, 206, 245, 0.8)'),
)

trace2 = go.Bar(
                x = cn_ie.year,
                y = cn_ie.weight_kg,
                name = "China",
                marker = dict(color = 'rgba(255, 248, 12, 0.8)'),
)
data = [trace1, trace2]
layout = {
    'xaxis': {'title': 'Year 1992-2016'},
    'yaxis': {'title': 'Import & Export in Weight (kg)'},
    'barmode': 'group',
    'title': 'World vs China: Import & Export in Weight'
}
fig = go.Figure(data = data, layout = layout)
iplot(fig)

# ratio
trace3 = go.Scatter(
                    x = cn_ie.year,
                    y = cn_ie.weight_kg/wd_ie.weight_kg*100,
                    mode = "lines+markers",
                    name = "Ratio of China/World",
                    marker = dict(color = 'rgba(84, 222, 90, 0.8)')
)
data2 = [trace3]
layout2 = dict(title = 'Percentage of China\'s Import & Export in World (%)',
              xaxis= dict(title= 'Year 1992-2016',ticklen= 5,zeroline= False),
              yaxis = {'title': 'Percentage (%)'}
 )
fig2 = dict(data = data2, layout = layout2)
iplot(fig2)
us_trade = df[(df.country_or_area == "USA") & (df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')
jp_trade = df[(df.country_or_area == "Japan") & (df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

cn_1992 = int(cn_trade[cn_trade.year==1992].iloc[0][1])
us_1992 = int(us_trade[us_trade.year==1992].iloc[0][1])
jp_1992 = int(jp_trade[jp_trade.year==1992].iloc[0][1])
ot_1992 = int(wd_trade[wd_trade.year==1992].iloc[0][1]) - cn_1992 - us_1992 - jp_1992

cn_2016 = int(cn_trade[cn_trade.year==2016].iloc[0][1])
us_2016 = int(us_trade[us_trade.year==2016].iloc[0][1])
jp_2016 = int(jp_trade[jp_trade.year==2016].iloc[0][1])
ot_2016 = int(wd_trade[wd_trade.year==2016].iloc[0][1]) - cn_2016 - us_2016 - jp_2016

labels = ['China','USA','Japan','Others']
colors = ['#f18285', '#86e48f', '#e8a2d8', '#fff76e']

#####
trace = go.Pie(labels=labels, values=[cn_1992, us_1992, jp_1992, ot_1992],
               marker=dict(colors=colors,  line=dict(color='#000', width=2)) )
layout = go.Layout(
    title='1992 Import & Export Trade in USD',
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='basic_pie_chart')

######
trace1 = go.Pie(labels=labels, values=[cn_2016, us_2016, jp_2016, ot_2016],
               marker=dict(colors=colors,  line=dict(color='#000', width=2)) )
layout1 = go.Layout(
    title='2016 Import & Export Trade in USD',
)

fig1 = go.Figure(data=[trace1], layout=layout1)
iplot(fig1, filename='basic_pie_chart1')
