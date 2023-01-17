import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, date

from matplotlib import pyplot as plt

!pip install chart_studio

import chart_studio.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook! - thanks, Rachael

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
import os, time

# print(os.listdir("../input/"))

file_mod_time = os.path.getmtime("../input/daily-inmates-in-custody.csv")

print("The data file was last modified", time.ctime(file_mod_time), "UTC.")
df = pd.read_csv("../input/daily-inmates-in-custody.csv")

print(df.head(10))

list(df) # list the column names; uncomment for reference as needed

print(len(df))

print(df.nunique()) # check that INMATEID is unique for every entry -- it is
df.sample(10)
# explore male-female numbers by race

color_and_sex = df.groupby(['RACE','GENDER']).INMATEID.count().reset_index()

c_and_s_pivot = color_and_sex.pivot(columns="RACE", index="GENDER", values='INMATEID').reset_index()

print(c_and_s_pivot)
women_race = c_and_s_pivot.iloc[0,1:]

# print(women_race)

men_race = c_and_s_pivot.iloc[1,1:]

# print(men_race)

list_of_races = ['Asian','Black','Inidian','Other','Unspecified','White']



trace1 = go.Bar(

    x= list_of_races,

    y= men_race,

    name='Men'

)

trace2 = go.Bar(

    x= list_of_races,

    y= women_race,

    name='Women'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='NYC Incarceration by Race and Sex'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
fig = {

  "data": [

    {

      "values": men_race,

      "labels": list_of_races,

      "domain": {"x": [0, .48]},

      "name": "Men",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },

    {

      "values": women_race,

      "labels": list_of_races,

      "text":["Women"],

      "textposition":"inside",

      "domain": {"x": [.52, 1]},

      "name": "Women",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"NYC Incarceration by Race and Sex",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Men",

                "x": 0.2,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Women",

                "x": 0.82,

                "y": 0.5

            }

        ]

    }

}

iplot(fig, filename='donut')
# challenge: populate with a for loop

trace1 = go.Bar(

    x=['Men', 'Women'],

    y=[men_race[1], women_race[1]],

    name='Black'

)

trace2 = go.Bar(

    x=['Men', 'Women'],

    y=[men_race[3], women_race[3]],

    name='Other'

)

trace3 = go.Bar(

    x=['Men', 'Women'],

    y=[men_race[5], women_race[5]],

    name='White'

)

trace4 = go.Bar(

    x=['Men', 'Women'],

    y=[men_race[0], women_race[0]],

    name='Asian'

)

trace5 = go.Bar(

    x=['Men', 'Women'],

    y=[men_race[2], women_race[2]],

    name='Indian'

)

trace6 = go.Bar(

    x=['Men', 'Women'],

    y=[men_race[4], women_race[4]],

    name='Unspecified'

)



data = [trace1, trace2, trace3, trace4, trace5, trace6]

layout = go.Layout(

    barmode='group',

    title="NYC Incarceration by Race and Sex"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped-bar')



# here's the for loop

# data = []

# for i in range(length(list_of_races)):

#     trace = go.Bar(

#    x=['Men', 'Women'],

#    y=[men_race[i], women_race[i]],

#    name=list_of_races[i])

#    data.append(trace)

# modify the loop so it displays in the same order as the non-for loop chart -- aids readability 
# explore numbers by race and gang affiliation

race_gang = df.groupby(['RACE','SRG_FLG']).INMATEID.count().reset_index()

race_gang_pivot = race_gang.pivot(columns="RACE", index="SRG_FLG", values='INMATEID').reset_index()

print(race_gang_pivot)

race_no_gang = race_gang_pivot.iloc[0,1:]

# print(women_race)

race_gang = race_gang_pivot.iloc[1,1:]

gang_list = ['No Gang Affiliation', 'Gang Affilition']

trace1 = go.Bar(

    x=gang_list,

    y=[race_no_gang[1], race_gang[1]],

    name='Black'

)

trace2 = go.Bar(

    x=gang_list,

    y=[race_no_gang[3], race_gang[3]],

    name='Other'

)

trace3 = go.Bar(

    x=gang_list,

    y=[race_no_gang[5], race_gang[5]],

    name='White'

)

trace4 = go.Bar(

    x=gang_list,

    y=[race_no_gang[0], race_gang[0]],

    name='Asian'

)

trace5 = go.Bar(

    x=gang_list,

    y=[race_no_gang[2], race_gang[2]],

    name='Indian'

)

trace6 = go.Bar(

    x=gang_list,

    y=[race_no_gang[4], race_gang[4]],

    name='Unspecified'

)



data = [trace1, trace2, trace3, trace4, trace5, trace6]

layout = go.Layout(

    barmode='group',

    title="NYC Incarceration by Race and Gang Affiliation"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped-bar_gang')
# explore numbers by age

age_sort = df.groupby('AGE').INMATEID.count().reset_index()

# print(age_sort.head())

age_x = age_sort.iloc[:,0]

age_y = age_sort.iloc[:,1]

# print(age_x)

# print(age_y)

data = [go.Scatter(x=age_x, y=age_y)]



# specify the layout of our figure

layout = dict(title = "NYC Incarceration by Age",

              xaxis= dict(title= 'Age',ticklen=5,zeroline=False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
# compare numbers by age and race

age_race_sort = df.groupby(['AGE','RACE']).INMATEID.count().reset_index()

age_race_pivot = age_race_sort.pivot(columns="RACE", index="AGE", values='INMATEID').reset_index()

# print(age_race_sort.head())

print("age_race_pivot \n", age_race_pivot.head())
title = 'NYC Incarceration by Age and Race'

labels = list_of_races

colors = ['red', 'orange', 'brown', 'green', 'blue', 'indigo']

mode_size = [8] * 6

line_size = [2] * 6



x_data = [

    age_race_pivot.iloc[:,0],

    age_race_pivot.iloc[:,0],

    age_race_pivot.iloc[:,0],

    age_race_pivot.iloc[:,0],

    age_race_pivot.iloc[:,0],

    age_race_pivot.iloc[:,0],

]



y_data = [

    np.array(age_race_pivot.iloc[:,1]),

    np.array(age_race_pivot.iloc[:,2]),

    np.array(age_race_pivot.iloc[:,3]),

    np.array(age_race_pivot.iloc[:,4]),

    np.array(age_race_pivot.iloc[:,5]),

    np.array(age_race_pivot.iloc[:,6]),

]



traces = []



for i in range(0,6):

    traces.append(go.Scatter(

        x=x_data[i],

        y=y_data[i],

        mode='lines',

        name=labels[i],

        line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,

    ))



layout = go.Layout(

    xaxis=dict(

        showline=True,

        showgrid=True,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickcolor='rgb(204, 204, 204)',

        tickwidth=2,

        ticklen=5,

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=True,

        zeroline=False,

        showline=False,

        showticklabels=True,

    ),

    autosize=True,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

    ),

    showlegend=True

)



annotations = []



# Title

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text=title,

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

# Source

annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,

                              xanchor='center', yanchor='top',

                              text='Age',

                              font=dict(family='Arial',

                                        size=12,

                                        color='black'),

                              showarrow=False))



layout['annotations'] = annotations



fig = go.Figure(data=traces, layout=layout)

iplot(fig)
q75, q25 = np.percentile(age_x, [75 ,25])

mean_age = np.mean(age_x)

median_age = np.median(age_x)

print(f"The 25th percentile age is {int(q25)}, and the 75th percentile age is {int(q75)}.")

print(f"The mean age is {int(mean_age)}, and the median age is {int(median_age)}.")

print(f"The youngest is {min(age_x)}, and the oldest is {max(age_x)}.")
# explore custody level numbers by race

color_and_custody = df.groupby(['RACE','CUSTODY_LEVEL']).INMATEID.count().reset_index()

c_and_c_pivot = color_and_custody.pivot(columns="RACE", index="CUSTODY_LEVEL", values='INMATEID').reset_index()

print(c_and_c_pivot)
max_race = np.array(c_and_c_pivot.iloc[0,1:])

med_race = np.array(c_and_c_pivot.iloc[1,1:])

min_race = np.array(c_and_c_pivot.iloc[2,1:])

# print(max_race)  # for reference 

trace1 = go.Bar(

    x= list_of_races,

    y= max_race,

    name='Maximum'

)

trace2 = go.Bar(

    x= list_of_races,

    y= med_race,

    name='Medium'

)

trace3 = go.Bar(

    x= list_of_races,

    y= min_race,

    name='Minimum'

)

data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='stack',

    title='NYC Incarceration by Race and Custody Level'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
custody_legend = ['Maximum', 'Medium', 'Minimum']

color_max = 'rgb(230, 76, 0)'

color_med = 'rgb(230, 191, 0)'

color_min = 'rgb(128, 128, 128)'



x_list = [[0, 0.31], [ 0.33, 0.64], [0.66, 1]] # x-coordinates for pie plots (three columns)

y_list = [[0, 0.49], [ 0.51, 1]]               # y-coordinates for pie plots (two rows)

data_pie = []

for i in range(len(max_race)):

    data_pie.append({

        'labels': custody_legend,

        'values': [max_race[i], med_race[i], min_race[i]],

        'type': 'pie',

        'name': list_of_races[i],

        'marker': {'colors': [color_max,

                              color_med,

                              color_min]},

        'domain': {'x': x_list[i % 3],

                   'y': y_list[i % 2]},

        'hoverinfo':'label+percent+name',

        'textinfo':'none',

        'title': list_of_races[i]

    })

    

fig = {

    'data': data_pie,

    'layout': {'title': 'NYC Custody Level by Race - Shown Proportionally',

               'showlegend': True}

}



iplot(fig, filename='pie_chart_subplots')
# compare numbers by age and sex using a bar chart

age_sex_sort = df.groupby(['AGE','GENDER']).INMATEID.count().reset_index()

age_sex_pivot = age_sex_sort.pivot(columns="GENDER", index="AGE", values='INMATEID').reset_index()

# print(age_sex_sort.head())

print(age_sex_pivot.head())
age_array = np.array(age_sex_pivot.iloc[:,0]) # take all rows of of column 0 (AGE)

age_women = np.array(age_sex_pivot.iloc[:,1]) 

# age_women[~(age_women > 0)] = 0 # replace null values with 0 to keep the array the same size as the age_array

age_men = np.array(age_sex_pivot.iloc[:,2])

# print(age_women, age_men)

# a_s_labels = []

a_s_legend = ['Men', 'Women']

plt.figure(figsize=(10,8))

plt.bar(age_array, age_men)

plt.bar(age_array, age_women)

plt.legend(a_s_legend, loc=1)

plt.xlabel('Age')

plt.ylabel('Incarcerated')

plt.title('NYC Incarceration by Sex and Age')

plt.show()
# compare numbers by age and race

age_race_sort = df.groupby(['AGE','RACE']).INMATEID.count().reset_index()

age_race_pivot = age_race_sort.pivot(columns="RACE", index="AGE", values='INMATEID').reset_index()

# print(age_race_sort.head())

print(age_race_pivot.head())
df["duration"] = df.ADMITTED_DT.apply(lambda x: (datetime.now() - datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000")).days)

# print("full DataFrame length is", len(df))

days_offset = min(df.duration)



print("The minimum stay is", min(df.duration) - days_offset, "day(s).")

print("The maximum stay (so far) is", max(df.duration) - days_offset, "days.")

print("The average stay is", round(np.mean(df.duration),0) - days_offset, "days.")

print("The median length of stay is", round(np.median(df.duration) - days_offset,0), "days.")

print("The standard deviation is", round(np.std(df.duration),0) - days_offset, "days.")
data = [go.Histogram(x=df.duration)]

layout = go.Layout(

    title='Duration of Stay',

    xaxis=dict(

        title='Days'

    ),

    yaxis=dict(

        title='Number of Inmates'

    ),

) # end layout

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic histogram')
# df_short = df[df.duration < 1500] 

df_short = df[df.duration < np.percentile(df.duration, 90)]

# print(len(df_short))

print("The minimum stay is", min(df_short.duration) - days_offset, "day(s).")

print("The maximum stay (so far) is", max(df_short.duration) - days_offset, "days.")

print("The average stay is", round(np.mean(df_short.duration) - days_offset,0), "days.")

print("The median length of stay is", round(np.median(df_short.duration) - days_offset,0), "days.")

print("The standard deviation is", round(np.std(df_short.duration),0) - days_offset, "days.")



data = [go.Histogram(x=df_short.duration)]

layout = go.Layout(

    title='Duration of Stay',

    xaxis=dict(

        title='Days'

    ),

    yaxis=dict(

        title='Number of Inmates'

    ),

) # end layout

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic_histogram')
# compare by race and length of stay (so far)

# compare by sex and length of stay (so far)

# compare by age and length of stay (so far)