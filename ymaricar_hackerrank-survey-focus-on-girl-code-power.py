import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import plotly
import  plotly.offline as py
py.init_notebook_mode(connected=True)
#import plotly.plotly as py
import plotly.graph_objs as go
df=pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv', parse_dates=['StartDate','EndDate'])
df_n = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric.csv', parse_dates=['StartDate','EndDate'])
df_women = df[df.q3Gender == 'Female']
df_men = df[df.q3Gender != 'Female']
df.shape
df = df.dropna(axis=0, how='all')
df.shape
#c = 0 
#for i in df.columns : 
#    print(i + " "+ str(c))
#    c+= 1

df.head(1)
prog = df[df.columns[139:163]]
prog['Gender'] = df['q3Gender']
prog = prog.dropna(axis=0, how='all')
prog.columns
prog[0:5]
for i in prog.columns[:-1] :
    print(i + ": "+str(prog[i].isnull().sum()))

colors = ["blue", "orange", "greyish", "faded green", "dusty purple"]
fig, ax = plt.subplots(figsize=(20,20), ncols=5, nrows=5)
count = 0
times = 0
for i in prog.columns[:-1]:
    #sns.regplot(x='value', y='wage', data=df_melt, ax=axs[count])
    sns.countplot(x=str(i), hue="Gender", data=prog, palette = sns.xkcd_palette(colors), ax=ax[times][count])
    count += 1
    if count == 5 :
        times += 1
        count = 0

    
trace1 = go.Bar(
    x=df_men['q2Age'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_men['q2Age'].value_counts().tolist(),np.sum(df_men['q2Age'].value_counts().tolist())).tolist(),100).tolist(),
    name='Men Respondents'
)
trace2 = go.Bar(
    x=df_women['q2Age'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_women['q2Age'].value_counts().tolist(),np.sum(df_women['q2Age'].value_counts().tolist())).tolist(),100).tolist(),
    name='Female Respondents'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
trace1 = go.Bar(
    x=df_men['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_men['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_men['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Men Respondents'
)
trace2 = go.Bar(
    x=df_women['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_women['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_women['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Female Respondents'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
#df['time']=(df['EndDate']-df['StartDate']).astype('timedelta64[m]')

focus_country = df['CountryNumeric2'].value_counts().to_frame()
print("our TOP 10 country respondents is :") 
print(focus_country.head(10).index)
data = [ dict(
        type = 'choropleth',
        locations = focus_country.index,
        locationmode = 'country names',
        z = focus_country['CountryNumeric2'],
        text = focus_country['CountryNumeric2'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 1
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Respondents'),
      ) ]

layout = dict(
    title = 'Number of respondents by country',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
df_men_c = [0,0,0]
df_women_c = [0,0,0]
count = 0
for i in focus_country.head(3).index : 
    df_men_c[count] = df_men[df_men['CountryNumeric2'] == i]
    df_women_c[count] = df_women[df_women['CountryNumeric2'] == i]
    print('N° of Male respondents for '+ i + ' is : '+ str(df_men_c[count].shape[0]))
    print('N° of Female respondents for '+ i + ' is : '+ str(df_women_c[count].shape[0]))
    
    trace1 = go.Bar( 
    x=df_men_c[count]['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_men_c[count]['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_men_c[count]['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Men Respondents in '+i
    )
    trace2 = go.Bar(
    x=df_women_c[count]['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_women_c[count]['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_women_c[count]['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Female Respondents in '+i
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='grouped-bar')
    count = count + 1

