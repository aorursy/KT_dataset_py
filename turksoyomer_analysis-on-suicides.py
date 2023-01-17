# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
masterdata = pd.read_csv("../input/master.csv")
masterdata.rename(columns={"suicides/100k pop":"suicides_100kpop",

                    "country-year":"country_year",

                    "HDI for year":"hdi_for_year",

                    " gdp_for_year ($) ":"gdp_for_year",

                    "gdp_per_capita ($)":"gdp_per_capita",

                    }, inplace=True)

masterdata.gdp_for_year = [int(i.replace(",","")) for i in masterdata.gdp_for_year]
trace1 = go.Histogram(

    x=masterdata.gdp_per_capita,

    opacity=1,

    marker=dict(color='rgba(211, 94, 96, 1)'))



data = [trace1]

layout = go.Layout(barmode='overlay',

                   title='Suicide Frequency For Different GDP Per Capita Values',

                   xaxis=dict(title='GDP Per Capita'))



fig = go.Figure(data=data, layout=layout)

iplot(fig)
total_suicides_per_gdp = []

for i in masterdata.gdp_per_capita.unique():

    total_suicides_per_gdp.append(sum(masterdata[masterdata.gdp_per_capita == i].suicides_no))

trace0 = go.Box(

    y=total_suicides_per_gdp,

    name = 'Total Suicides For GDP Per Capita',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=masterdata.gdp_per_capita.unique(),

    name = 'GDP Per Capita',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

iplot(data)
suicides_gdp = pd.DataFrame({"total_suicides_per_gdp":total_suicides_per_gdp, "gdp_per_capita":masterdata.gdp_per_capita.unique()})

suicides_gdp = suicides_gdp[(suicides_gdp['total_suicides_per_gdp']<=5713) & (suicides_gdp['gdp_per_capita']<=57790)]

suicides_gdp = suicides_gdp.sort_values(by=["gdp_per_capita"])

trace1 = go.Scatter(

                    x = suicides_gdp.gdp_per_capita,

                    y = suicides_gdp.total_suicides_per_gdp,

                    mode = "markers",

                    name = "Suicides",

                    marker = dict(color = 'rgba(62, 150, 81, 0.8)'))



data = [trace1]

layout = dict(title = 'Total Suicides for per "GDP Per Capita" Values',

              xaxis= dict(title= 'GDP Per Suicides',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Total Suicides',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
turkey_suicide_gdp = masterdata[masterdata.country == "Turkey"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

usa_suicide_gdp = masterdata[masterdata.country == "United States"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

russia_suicide_gdp = masterdata[masterdata.country == "Russian Federation"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

brazil_suicide_gdp = masterdata[masterdata.country == "Brazil"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

france_suicide_gdp = masterdata[masterdata.country == "France"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

qatar_suicide_gdp = masterdata[masterdata.country == "Qatar"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

japan_suicide_gdp = masterdata[masterdata.country == "Japan"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

southAfrica_suicide_gdp = masterdata[masterdata.country == "South Africa"][["country","suicides_100kpop","gdp_for_year","gdp_per_capita"]]

turkey_mean_suicides_per_gdp = []

turkey_gdp_per_capita = turkey_suicide_gdp.gdp_per_capita.unique()

for i in turkey_gdp_per_capita:

    turkey_mean_suicides_per_gdp.append(round(sum(turkey_suicide_gdp[turkey_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(turkey_suicide_gdp[turkey_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))

usa_mean_suicides_per_gdp = []

usa_gdp_per_capita = usa_suicide_gdp.gdp_per_capita.unique()

for i in usa_gdp_per_capita:

    usa_mean_suicides_per_gdp.append(round(sum(usa_suicide_gdp[usa_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                     len(usa_suicide_gdp[usa_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))

russia_mean_suicides_per_gdp = []

russia_gdp_per_capita = russia_suicide_gdp.gdp_per_capita.unique()

for i in russia_gdp_per_capita:

    russia_mean_suicides_per_gdp.append(round(sum(russia_suicide_gdp[russia_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(russia_suicide_gdp[russia_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))

brazil_mean_suicides_per_gdp = []

brazil_gdp_per_capita = brazil_suicide_gdp.gdp_per_capita.unique()

for i in brazil_gdp_per_capita:

    brazil_mean_suicides_per_gdp.append(round(sum(brazil_suicide_gdp[brazil_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(brazil_suicide_gdp[brazil_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))

france_mean_suicides_per_gdp = []

france_gdp_per_capita = france_suicide_gdp.gdp_per_capita.unique()

for i in france_gdp_per_capita:

    france_mean_suicides_per_gdp.append(round(sum(france_suicide_gdp[france_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(france_suicide_gdp[france_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))  

qatar_mean_suicides_per_gdp = []

qatar_gdp_per_capita = qatar_suicide_gdp.gdp_per_capita.unique()

for i in qatar_gdp_per_capita:

    qatar_mean_suicides_per_gdp.append(round(sum(qatar_suicide_gdp[qatar_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(qatar_suicide_gdp[qatar_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))  

japan_mean_suicides_per_gdp = []

japan_gdp_per_capita = japan_suicide_gdp.gdp_per_capita.unique()

for i in japan_gdp_per_capita:

    japan_mean_suicides_per_gdp.append(round(sum(japan_suicide_gdp[japan_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(japan_suicide_gdp[japan_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))       

southAfrica_mean_suicides_per_gdp = []

southAfrica_gdp_per_capita = southAfrica_suicide_gdp.gdp_per_capita.unique()

for i in southAfrica_gdp_per_capita:

    southAfrica_mean_suicides_per_gdp.append(round(sum(southAfrica_suicide_gdp[southAfrica_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"])/

                                        len(southAfrica_suicide_gdp[southAfrica_suicide_gdp.gdp_per_capita == i]["suicides_100kpop"]),2))

trace1 =go.Scatter(

                    x = turkey_gdp_per_capita,

                    y = turkey_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "Turkey",

                    marker = dict(color = 'rgba(204, 37, 41, 0.9)'))

trace2 =go.Scatter(

                    x = usa_gdp_per_capita,

                    y = usa_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "USA",

                    marker = dict(color = 'rgba(57, 106, 177, 0.9)'))

trace3 =go.Scatter(

                    x = russia_gdp_per_capita,

                    y = russia_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "Russia",

                    marker = dict(color = 'rgba(107, 76, 154, 0.9)'))

trace4 =go.Scatter(

                    x = brazil_gdp_per_capita,

                    y = brazil_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "Brazil",

                    marker = dict(color = 'rgba(148, 139, 61, 0.9)'))

trace5 =go.Scatter(

                    x = france_gdp_per_capita,

                    y = france_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "France",

                    marker = dict(color = 'rgba(83, 81, 84, 0.9)'))

trace6 =go.Scatter(

                    x = qatar_gdp_per_capita,

                    y = qatar_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "Qatar",

                    marker = dict(color = 'rgba(62, 150, 81, 0.9)'))

trace7 =go.Scatter(

                    x = japan_gdp_per_capita,

                    y = japan_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "Japan",

                    marker = dict(color = 'rgba(146, 36, 40, 0.9)'))

trace8 =go.Scatter(

                    x = southAfrica_gdp_per_capita,

                    y = southAfrica_mean_suicides_per_gdp,

                    mode = "markers",

                    name = "South Africa",

                    marker = dict(color = 'rgba(218, 124, 48, 0.7)'))

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]

layout = dict(title = 'Relationship Between GDP and Suicides in Different Countries',

              xaxis= dict(title= 'GDP Per Capita',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Mean Suicides/100k Population Per GDP',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig)
countries = []

sexes = []

total_suicides = []

for c in ["Turkey", "United States", "Russian Federation", "Brazil", "France", "Qatar", "Japan", "South Africa"]:

    for s in masterdata[masterdata.country == c].sex.unique():

        countries.append(c)

        sexes.append(s)

        total_suicides.append(sum(masterdata[(masterdata.country == c) & (masterdata.sex == s)].suicides_no))

country_sex = pd.DataFrame({"country":countries, "sex":sexes, "total_suicides":total_suicides})

country_sex.sort_values(by=["sex"], inplace=True)

trace1 = {

  'x': country_sex.country,

  'y': country_sex[country_sex.sex == "male"].total_suicides,

  'name': 'Male',

  'type': 'bar'

};

trace2 = {

  'x': country_sex.country,

  'y': country_sex[country_sex.sex == "female"].total_suicides,

  'name': 'Female',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Countries'},

  'barmode': 'relative',

  'title': 'Suicide Rates by Sex in Some Countries'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
suicides_sex = pd.DataFrame(columns=["sex","suicides_no","period"])

for s in masterdata.sex.unique():

    new_data1985_1992 = pd.DataFrame(data={"sex":[s],

                                           "suicides_no":[sum(masterdata[(masterdata.sex == s) & (masterdata.year >= 1985) & (masterdata.year <= 1992)].suicides_no)],

                                           "period":["1985-1992"]})

    new_data1993_2000 = pd.DataFrame(data={"sex":[s],

                                           "suicides_no":[sum(masterdata[(masterdata.sex == s) & (masterdata.year >= 1993) & (masterdata.year <= 2000)].suicides_no)],

                                           "period":["1993-2000"]})

    new_data2001_2008 = pd.DataFrame(data={"sex":[s],

                                           "suicides_no":[sum(masterdata[(masterdata.sex == s) & (masterdata.year >= 2001) & (masterdata.year <= 2008)].suicides_no)], 

                                           "period":["2001-2008"]})

    new_data2009_2016 = pd.DataFrame(data={"sex":[s],

                                           "suicides_no":[sum(masterdata[(masterdata.sex == s) & (masterdata.year >= 2009) & (masterdata.year <= 2016)].suicides_no)],

                                           "period":["2009-2016"]})   

    suicides_sex = pd.concat([suicides_sex,new_data1985_1992])

    suicides_sex = pd.concat([suicides_sex,new_data1993_2000])

    suicides_sex = pd.concat([suicides_sex,new_data2001_2008])

    suicides_sex = pd.concat([suicides_sex,new_data2009_2016])

    

fig = {

    'data': [

        {

            'labels': suicides_sex[suicides_sex.period == "2001-2008"].sex,

            'values': suicides_sex[suicides_sex.period == "2001-2008"].suicides_no,

            'type': 'pie',

            'name': '2001 - 2008',

            'marker': {'colors': ['rgb(0, 92, 153)',

                                  'rgb(204, 153, 255)']},

            'domain': {'x': [0, .48],

                       'y': [0, .49]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'

        },

        {

            'labels': suicides_sex[suicides_sex.period == "2009-2016"].sex,

            'values': suicides_sex[suicides_sex.period == "2009-2016"].suicides_no,

            'marker': {'colors': ['rgb(0, 92, 153)',

                                  'rgb(204, 153, 255)']},

            'type': 'pie',

            'name': '2009 - 2016',

            'domain': {'x': [.52, 1],

                       'y': [0, .49]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'



        },

        {

            'labels': suicides_sex[suicides_sex.period == "1985-1992"].sex,

            'values': suicides_sex[suicides_sex.period == "1985-1992"].suicides_no,

            'marker': {'colors': ['rgb(0, 92, 153)',

                                  'rgb(204, 153, 255)']},

            'type': 'pie',

            'name': '1985 - 1992',

            'domain': {'x': [0, .48],

                       'y': [.51, 1]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'

        },

        {

            'labels': suicides_sex[suicides_sex.period == "1993-2000"].sex,

            'values': suicides_sex[suicides_sex.period == "1993-2000"].suicides_no,

            'marker': {'colors': ['rgb(0, 92, 153)',

                                  'rgb(204, 153, 255)']},

            'type': 'pie',

            'name':'1993 - 2000',

            'domain': {'x': [.52, 1],

                       'y': [.51, 1]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'

        }

    ],

    'layout': {'title': 'Suicide Rates by Sex For Periods (1985-1992, 1993-2000, 2001-2008, 2009-2016)',

               'showlegend': True}

}



iplot(fig)
most_suicides_country = masterdata[["country", "suicides_no"]]

countries = []

suicides= []

for i in most_suicides_country.country.unique():

    suicides.append(sum(most_suicides_country[most_suicides_country["country"]==i]["suicides_no"]))

    countries.append(i)

most_suicides_country = pd.DataFrame({"countries":countries, "suicides": suicides})

most_suicides_country.sort_values(by="suicides", inplace=True, ascending=False)

mean_gdp_per_capita = []

for c in most_suicides_country.countries[:10]:

    mean_gdp_per_capita.append(int(sum(masterdata[masterdata.country == c].gdp_per_capita)/len(masterdata[masterdata.country == c].gdp_per_capita)))

mean_gdp_per_capita = ["Mean GDP Per Capita in This Country: "+str(i) for i in mean_gdp_per_capita]

trace1 = go.Bar(

                x = most_suicides_country.countries[:10],

                y = most_suicides_country.suicides[:10],

                name = "citations",

                marker = dict(color=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],colorscale='Viridis',

                    line=dict(color='rgb(0,0,0)',width=1.5)),

                text=mean_gdp_per_capita)

data = [trace1]

layout = go.Layout(title='Top Ten Most Suicidal Countries Between 1985 and 2016',xaxis=dict(title="Country"), yaxis=dict(title="Total Suicides"))

fig = go.Figure(data = data, layout = layout)

plt.savefig("graph.png")

iplot(fig)
md1985_1992 = masterdata[(masterdata.year >= 1985) & (masterdata.year <= 1992)]

countries1985_1992 = md1985_1992.country.unique()

total_suicides1985_1992 = []

for c in countries1985_1992:

    total_suicides1985_1992.append(sum(md1985_1992[md1985_1992.country == c].suicides_no))



md1993_2000 = masterdata[(masterdata.year >= 1993) & (masterdata.year <= 2000)]

countries1993_2000 = md1993_2000.country.unique()

total_suicides1993_2000 = []

for c in countries1993_2000:

    total_suicides1993_2000.append(sum(md1993_2000[md1993_2000.country == c].suicides_no))

    

md2001_2008 = masterdata[(masterdata.year >= 2001) & (masterdata.year <= 2008)]

countries2001_2008 = md2001_2008.country.unique()

total_suicides2001_2008 = []

for c in countries2001_2008:

    total_suicides2001_2008.append(sum(md2001_2008[md2001_2008.country == c].suicides_no))

    

md2009_2016 = masterdata[(masterdata.year >= 2009) & (masterdata.year <= 2016)]

countries2009_2016 = md2009_2016.country.unique()

total_suicides2009_2016 = []

for c in countries2009_2016:

    total_suicides2009_2016.append(sum(md2009_2016[md2009_2016.country == c].suicides_no))

    

md1985_1992 = pd.DataFrame({"countries":countries1985_1992, "total_suicides": total_suicides1985_1992})

md1993_2000 = pd.DataFrame({"countries":countries1993_2000, "total_suicides": total_suicides1993_2000})

md2001_2008 = pd.DataFrame({"countries":countries2001_2008, "total_suicides": total_suicides2001_2008})

md2009_2016 = pd.DataFrame({"countries":countries2009_2016, "total_suicides": total_suicides2009_2016})



md1985_1992.sort_values(by=["total_suicides"], inplace=True, ascending=False)

md1993_2000.sort_values(by=["total_suicides"], inplace=True, ascending=False)

md2001_2008.sort_values(by=["total_suicides"], inplace=True, ascending=False)

md2009_2016.sort_values(by=["total_suicides"], inplace=True, ascending=False)



trace1 = go.Bar(

    x=md1985_1992.countries[:10],

    y=md1985_1992.total_suicides[:10],

    xaxis='x3',

    yaxis='y3',

    name = "1985-1992"

)

trace2 = go.Bar(

    x=md1993_2000.countries[:10],

    y=md1993_2000.total_suicides[:10],

    xaxis='x4',

    yaxis='y4',

    name = "1993-2000"

)

trace3 = go.Bar(

    x=md2001_2008.countries[:10],

    y=md2001_2008.total_suicides[:10],

    name = "2001-2008"

)

trace4 = go.Bar(

    x=md2009_2016.countries[:10],

    y=md2009_2016.total_suicides[:10],

    xaxis='x2',

    yaxis='y2',

    name = "2009-2016"

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = 'Most Suicidal Countries For Some Periods Between 1985-2016'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
generation_list = masterdata.generation.unique()

total_suicides_for_generations = []

for g in generation_list:

    total_suicides_for_generations.append(sum(masterdata[masterdata.generation == g].suicides_no))

    

labels = generation_list

values = total_suicides_for_generations

colors = ['#FEBFB3', '#C39BD3', '#96D38C', '#F8C471', "#5499C7", "#CD6155"]



trace = go.Pie(labels=labels, values=values,title="Total Suicides for Different Generations",titlefont=dict(size=18),

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

data = [trace]



fig = go.Figure(data=data)

iplot(fig)