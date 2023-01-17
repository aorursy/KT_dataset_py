#########################################################

#                  Content                              #

#########################################################

# 1.) A matplotLib plot                                 #

# 2.) Plotly pie chart                                  #

# 3.) Plotly Bar Graphs                                 #

# 4.) Plotly 3d Scatter                                 #

# 5.) dist_plot ploty.figure_factory                    #

# 6.)                                                   #

#########################################################



##########################################################

# Resource Used : https://youtu.be/D35m2CdMhVs           #

##########################################################



###########################################################

#             Do UPVOTE if tutorial_is_useful()           #

###########################################################
import numpy as np

import pandas as pd

DataFrame = pd.read_csv('/kaggle/input/csgo-round-winner-classification/csgo_round_snapshots.csv')

#DataFrame.info()
DataFrame.loc[0 : 10]
## let's see which map prefer which team

uMap = set(DataFrame['map'])

CTwins = []

Twins = []

MAP = []



for mem in uMap:

    MAP.append(mem)

    CTwins.append( len(DataFrame[ (DataFrame['map'] == mem) & (DataFrame['round_winner'] == 'CT')]) )

    Twins.append( len(DataFrame[ (DataFrame['map'] == mem) & (DataFrame['round_winner'] == 'T')]) )
totalCTwins = len(DataFrame[DataFrame['round_winner'] == 'CT'])

totalTwins = len(DataFrame[DataFrame['round_winner'] == 'T'])

Total = len(DataFrame)



import matplotlib.pyplot as plt





plt.figure(figsize = (16 ,7))

plt.subplot(1,2,1)

plt.rcParams.update({'font.size': 22})

plt.bar(MAP, CTwins, label = ' CT wins')

plt.xticks(rotation = 'vertical')

plt.title('CT wins Mapwise Distribution')

plt.grid(True)

plt.legend()



plt.subplot(1,2,2)

plt.rcParams.update({'font.size': 22})

plt.bar(MAP, Twins, label = ' T wins', color = 'orange')

plt.xticks(rotation = 'vertical')

plt.title('T wins Mapwise Distribution')

plt.grid(True)

plt.legend()



########################################################

#    From Here onwards we will be using Plotly         #

#    for our Visualizations                            #

########################################################



import plotly

from plotly.offline import plot, iplot, download_plotlyjs, init_notebook_mode



## note that this tells  plotly to plot on the Jupyter notebook

init_notebook_mode(connected=True)



## Lets find who wins how many matches

labels = ['T wins', 'CT wins']

values = [len(DataFrame[DataFrame['round_winner'] == 'T']) , len(DataFrame[DataFrame['round_winner'] == 'CT'])]



## importing Plotly graph objects

import plotly.graph_objs as go





## adjusting fonts

fig = go.Figure()

fig.update_layout(

font=dict(

        family="Courier New, monospace",

        size=50,

        color="RebeccaPurple")

)



## Plotting data

trace = go.Pie(

labels = labels,

values = values,

title = 'PieChart: T Vs CT wins',

)



##displaying Data

## using iplot instead of plot tells plotly to display data in jupyter notebooks itself

## using plot will open a spereate window in browser

iplot([trace])
## It seems T wins most of the time Bummer!!!
#####################################################

#       plotting groupped bar graphs                #

#####################################################





x_val = ["T wins", "CT wins"]

y_label1 = [

    len(DataFrame[(DataFrame['round_winner'] == 'T') & (DataFrame['bomb_planted']== True)]),

    len(DataFrame[(DataFrame['round_winner'] == 'CT') & (DataFrame['bomb_planted']== True)])

           ]



y_label2 = [

    len(DataFrame[(DataFrame['round_winner'] == 'T') & (DataFrame['bomb_planted']== False)]),

    len(DataFrame[(DataFrame['round_winner'] == 'CT') & (DataFrame['bomb_planted']== False)])

]



## We here are plotting two different series of bars

## 1.) T Vs CT wins when bomb was Planted

## 2.) T Vs CT wins when bomb was not Planted



trace1 = go.Bar(

x = x_val,

y = y_label1,

name='T Vs CT Wins when bomb was planted'

)



trace2 =  go.Bar(

x = x_val,

y = y_label2,

name='T Vs CT Wins when bomb was Not  planted'

)



data  = [trace1, trace2]

layout = go.Layout(barmode='group')



fig = go.Figure(data = data, layout= layout)

iplot(fig, filename='was bomb_planted')



###############################################################

#           This provides much easiear way to plot            #

#           groupped data like marks of student in            #

#           different subjects. much simpler to               #

#           than matplot lib approach                         #

###############################################################





################################################################

#          Also Please dont let  T to Plant bomb               #

################################################################
##########################################################

#        Plotting 3d plot in Plotly                      #

#        x = health                                      #

#        y = armor                                       #

#        z =  money                                      #

##########################################################



trace1 = go.Scatter3d(

    x = DataFrame[DataFrame['round_winner'] == 'CT'].head(500).ct_health,

    y = DataFrame[DataFrame['round_winner'] == 'CT'].head(500).ct_armor,

    z = DataFrame[DataFrame['round_winner'] == 'CT'].head(500).ct_money,

    mode = 'markers',

    name= 'T wins'

)



trace2 = go.Scatter3d(

    x = DataFrame[DataFrame['round_winner'] == 'T'].head(500).t_health,

    y = DataFrame[DataFrame['round_winner'] == 'T'].head(500).t_armor,

    z = DataFrame[DataFrame['round_winner'] == 'T'].head(500).t_money,

    mode = 'markers',

    name = 'CT wins'

)



iplot([trace1, trace2])



###################################################

#   again much easier than matplot lib            #

###################################################
#import plotly.express as px

#fig = px.choropleth_mapbox(DataFrame)

#iplot(fig)
##########################################################

#      Plotting gaussian distributions                   #

#      for this example lets take health                 #

#      variable of data frame                            #

##########################################################





import plotly.figure_factory as ff

fig = ff.create_distplot([DataFrame[DataFrame['round_winner'] == 'T'].t_health,

                          DataFrame[DataFrame['round_winner'] == 'CT'].ct_health],

                         ['health distribution Plot when T wins', 'health distribution when CT wins'])

fig.show()