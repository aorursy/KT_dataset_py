# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library
import plotly.plotly as py # visualization library
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
import warnings            
warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.
plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.
# Any results you write to the current directory are saved as output.

aerial = pd.read_csv("../input/operations.csv")
# drop countries that are NaN
aerial = aerial[pd.isna(aerial.Country)==False]
# drop if target longitude is NaN
aerial = aerial[pd.isna(aerial['Target Longitude'])==False]
# Drop if takeoff longitude is NaN
aerial = aerial[pd.isna(aerial['Takeoff Longitude'])==False]
# drop unused features
drop_list = ['Mission ID','Unit ID','Target ID','Altitude (Hundreds of Feet)','Airborne Aircraft',
             'Attacking Aircraft', 'Bombing Aircraft', 'Aircraft Returned',
             'Aircraft Failed', 'Aircraft Damaged', 'Aircraft Lost',
             'High Explosives', 'High Explosives Type','Mission Type',
             'High Explosives Weight (Pounds)', 'High Explosives Weight (Tons)',
             'Incendiary Devices', 'Incendiary Devices Type',
             'Incendiary Devices Weight (Pounds)',
             'Incendiary Devices Weight (Tons)', 'Fragmentation Devices',
             'Fragmentation Devices Type', 'Fragmentation Devices Weight (Pounds)',
             'Fragmentation Devices Weight (Tons)', 'Total Weight (Pounds)',
             'Total Weight (Tons)', 'Time Over Target', 'Bomb Damage Assessment','Source ID']
aerial.drop(drop_list, axis=1,inplace = True)
aerial = aerial[ aerial.iloc[:,8]!="4248"] # drop this takeoff latitude 
aerial = aerial[ aerial.iloc[:,9]!=1355]   # drop this takeoff longitude
aerial.info()
print(aerial['Country'].value_counts())
plt.figure(figsize=(22,10))
sns.countplot(aerial['Country'])
plt.show()
print(aerial['Target Country'].value_counts()[:10])
plt.figure(figsize=(22,10))
sns.countplot(aerial['Target Country'])
plt.xticks(rotation=90)
plt.show()
data = aerial['Aircraft Series'].value_counts()
print(data[:10])
data = [go.Bar(
            x=data[:10].index,
            y=data[:10].values,
            hoverinfo = 'text',
            marker = dict(color = 'rgba(177, 14, 22, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    )]

layout = dict(
    title = 'Aircraft Series',
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
aerial.head()
#aerial["color"] = ""
#aerial.color[aerial.Country == "USA"] = "rgb(0,116,217)"
#aerial.color[aerial.Country == "GREAT BRITAIN"] = "rgb(255,65,54)"
#aerial.color[aerial.Country == "NEW ZEALAND"] = "rgb(133,20,75)"
#erial.color[aerial.Country == "SOUTH AFRICA"] = "rgb(255,133,27)"

#data = [dict(
#    type='scattergeo',
 #   lon = aerial['Takeoff Longitude'],
  #  lat = aerial['Takeoff Latitude'],
   # hoverinfo = 'text',
#    text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],
 #   mode = 'markers',
  #  marker=dict(
   #     sizemode = 'area',
    #    sizeref = 1,
     #   size= 10 ,
      #  line = dict(width=1,color = "white"),
       # color = aerial["color"],
        #opacity = 0.7),
#)]
#layout = dict(
 #   title = 'Countries Take Off Bases ',
  #  hovermode='closest',
   # geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
    #           countrywidth=1, projection=dict(type='Mercator'),
     #         landcolor = 'rgb(217, 217, 217)',
      #        subunitwidth=1,
       #       showlakes = True,
        #      lakecolor = 'rgb(255, 255, 255)',
         #     countrycolor="rgb(5, 5, 5)")
#)
#fig = go.Figure(data=data, layout=layout)
#iplot(fig)
