import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings 

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
whr = pd.read_csv("../input/2017.csv")

whr2015 = pd.read_csv("../input/2015.csv")

whr2016 = pd.read_csv("../input/2016.csv")

whr2017 = pd.read_csv("../input/2017.csv")
whr.head()
whr.info()
country = list(whr['Country'].unique())

whr.drop(["Freedom","Generosity","Trust..Government.Corruption."],axis=1)

family_ratio = []

for i in country:

    x = whr[whr['Country']==i]

    Family = sum(x.Family)/len(x)

    family_ratio.append(Family)

data = pd.DataFrame({'country': country,'family_report':family_ratio})

new_index = (data['family_report'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

    

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["country"][0:10],y=sorted_data["family_report"][0:10])

plt.xlabel("Countries")

plt.ylabel("Family Report")

plt.title("Sorting countris depend on family ratio")

plt.show()
country = list(whr['Country'].unique())

family = []

freedom = []

generosity = []

for i in country:

    x = whr[whr['Country']==i]

    family.append(sum(x.Family)/len(x))

    freedom.append(sum(x.Freedom) / len(x))

    generosity.append(sum(x.Generosity) / len(x))

    

f,ax = plt.subplots(figsize = (9,7))

sns.barplot(x=family[0:15],y=country[0:15],color='green',alpha = 0.5,label='Family' )

sns.barplot(x=freedom[0:15],y=country[0:15],color='blue',alpha = 0.7,label='Freedom')

sns.barplot(x=generosity[0:15],y=country[0:15],color='cyan',alpha = 0.6,label='Generosity')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel="Ratios", ylabel='Countries',title = "Family,Freedom and Generosity Ratios for each country")

plt.show()
whsker = whr["Whisker.low"]/max(whr["Whisker.low"])

fmly = whr["Family"]/max(whr["Family"])

g = sns.jointplot(fmly,whsker,kind="kde",height=7)

plt.savefig("graph.png")

plt.show()
whsker = whr["Whisker.low"]/max(whr["Whisker.low"])

fmly = whr["Family"]/max(whr["Family"])

g = sns.jointplot(fmly,whsker,kind="hex",height=7)

plt.savefig("graph.png")

plt.show()
sns.lmplot(x="Economy..GDP.per.Capita.",y="Happiness.Score",data=whr)

plt.xlabel("Economy Ratio")

plt.ylabel("Happiness Score")

plt.show()
sns.kdeplot(whr["Economy..GDP.per.Capita."],whr["Happiness.Score"],shade= True,cut=3)

plt.xlabel("Economy Ratio")

plt.ylabel("Happiness Score")

plt.show()
family = whr.Family

freedom = whr.Freedom

new_data = pd.DataFrame({'family': family,'freedom':freedom})

new_data.head()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=new_data, palette=pal, inner="points")

plt.title("VIOLIN PLOT")

plt.show()
whr.corr()
f,ax = plt.subplots(figsize=(6, 6))

sns.heatmap(whr.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title("HEATMAP")

plt.show()
sns.pairplot(new_data)

plt.show()
dictionary = {"List1":[1,np.nan,3,np.nan,5],

             "List2":[1,np.nan,np.nan,3,np.nan],

             "List3":[1,np.nan,3,4,5]}

data_msno = pd.DataFrame(dictionary)



import missingno as msno

msno.matrix(data_msno)

plt.show()
from pandas.tools.plotting import parallel_coordinates



dropped_whr = whr.drop(["Generosity","Family","Freedom","Happiness.Rank","Whisker.high","Whisker.low","Trust..Government.Corruption."],axis=1)

plt.figure(figsize=(15,10))

parallel_coordinates(dropped_whr[0:3], 'Country', colormap=plt.get_cmap("Set1"))

plt.savefig('graph.png')

plt.show()
family = whr.Family

freedom = whr.Freedom

generosity = whr.Generosity



from matplotlib_venn import venn2

family = data.iloc[:,0]

freedom = data.iloc[:,1]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(family)-15, len(freedom)-15, 15), set_labels = ('family', 'freedom'))

plt.show()
family = whr.Family

freedom = whr.Freedom

generosity = whr.Generosity

#

feature_names = "family","freedom","generosity"

feature_size = [len(family),len(freedom),len(generosity)]

#

circle = plt.Circle((0,0),0.2,color = "white")

plt.pie(feature_size, labels = feature_names, colors = ["red","green","blue","cyan"] )

p = plt.gcf()

p.gca().add_artist(circle)

plt.title("Number of Each Features")

plt.show()
df = whr.loc[:,["Happiness.Score","Generosity","Family","Freedom"]]

df1 = whr.Generosity

x = dict(zip(df1.unique(),"rgb"))

row_colors = df1.map(x)

cg = sns.clustermap(df,row_colors=row_colors,figsize=(12, 12),metric="correlation")

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),rotation = 0,size =8)

plt.show()
data = pd.read_csv('../input/2017.csv')



Norway = data[data.Country == "Norway"]



Iceland = data[data.Country == "Iceland"]



trace1 = go.Scatter3d(

    x=Norway.Family,

    y=Norway.Freedom,

    z=Norway.Generosity,

    mode='markers',

    name = "Norway",

    marker=dict(

        color='rgb(217, 100, 100)',

        size=12,

        line=dict(

            color='rgb(255, 255, 255)',

            width=0.1

        )

    )

)

trace2 = go.Scatter3d(

    x=Iceland.Family,

    y=Iceland.Freedom,

    z=Iceland.Generosity,

    mode='markers',

    name = "Iceland",

    marker=dict(

        color='rgb(54, 170, 127)',

        size=12,

        line=dict(

            color='rgb(204, 204, 204)',

            width=0.1

        )

    )

    

)

data = [trace1, trace2]

layout = go.Layout(

    title = ' 3D iris_setosa and iris_virginica',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df = whr.iloc[:100,:]



import plotly.graph_objs as go



trace1 = go.Scatter(

                    x = df["Happiness.Rank"],

                    y = df["Happiness.Score"],

                    mode = "lines",

                    name = "Happiness Score",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.Country)

trace2 = go.Scatter(

                    x = df["Happiness.Rank"],

                    y = df["Economy..GDP.per.Capita."],

                    mode = "lines+markers",

                    name = "Economy",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.Country)

data = [trace1, trace2]

layout = dict(title = "Top 100 Happiest Country's Happiness and Economy Scores",

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
whr2015 = whr2015[:20]

whr2016 = whr2016[:20]

whr2017 = whr2017[:20]



import plotly.graph_objs as go

trace1 =go.Scatter(

                    x = whr2015["Happiness Rank"],

                    y = whr2015["Happiness Score"],

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 255, 1)'),

                    text= whr2015.Country)

trace2 =go.Scatter(

                    x = whr2016["Happiness Rank"],

                    y = whr2016["Happiness Score"],

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(255, 128, 2, 1)'),

                    text= whr2016.Country)

trace3 =go.Scatter(

                    x = whr2017["Happiness.Rank"],

                    y = whr2017["Happiness.Score"],

                    mode = "markers",

                    name = "2017",

                    marker = dict(color = 'rgba(0, 255, 200, 1)'),

                    text= whr2017.Country)

data = [trace1, trace2, trace3]

layout = dict(title = 'Happiness Score vs Happiness Rank of top 100 Countries with 2014, 2015 and 2016 years',

              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Happiness Score',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
import plotly.graph_objs as go



trace1 = go.Bar(

                x = whr2015[:3].Country,

                y = whr2015[:3]["Happiness Score"],

                name = "Happiness Score",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = whr2015[:3].Country)

trace2 = go.Bar(

                x = whr2015[:3].Country,

                y = whr2015[:3]["Family"],

                name = "Family",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = whr2015[:3].Country)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
import plotly.graph_objs as go



x = whr2015[:3].Country



trace1 = {

  'x': x,

  'y': whr2015[:3]["Happiness Score"],

  'name': 'Happiness Score',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': whr2015[:3]["Family"],

  'name': 'Family',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 Countries'},

  'barmode': 'relative',

  'title': 'Happiness and Family of top 3 countries in 2015'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt



y_saving = [each for each in whr2017["Generosity"][:5]]

y_net_worth  = [float(each) for each in whr2017["Happiness.Score"][:5]]

x_saving = [each for each in whr2017.Country[:5]]

x_net_worth  = [each for each in whr2017.Country[:5]]

trace0 = go.Bar(

                x=y_saving,

                y=x_saving,

                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

                name='Generosity',

                orientation='h',

)

trace1 = go.Scatter(

                x=y_net_worth,

                y=x_net_worth,

                mode='lines+markers',

                line=dict(color='rgb(63, 72, 204)'),

                name='Happiness',

)

layout = dict(

                title='Generosity and Happiness',

                yaxis=dict(showticklabels=True,domain=[0, 0.85]),

                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),

                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),

                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),

                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),

                margin=dict(l=200, r=20,t=70,b=70),

                paper_bgcolor='rgb(248, 248, 255)',

                plot_bgcolor='rgb(248, 248, 255)',

)

annotations = []

y_s = np.round(y_saving, decimals=2)

y_nw = np.rint(y_net_worth)

# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x_saving):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))



layout['annotations'] = annotations



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                          shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)
data = [

    {

        'y': whr2017["Generosity"][:10],

        'x': whr2017["Happiness.Rank"][:10],

        'mode': 'markers',

        'marker': {

            'color': whr2017["Happiness.Score"],

            'size': whr2017["Happiness.Score"],

            'showscale': True

        },

        "text" :  whr2017["Country"][:10]    

    }

]



iplot(data)