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
import pandas as pd

dataset=pd.read_csv('/kaggle/input/covid19-dataset/COVID-19 Dataset.csv')

dataset.head(10)
latestDate = dataset[(dataset['Date'] == '2020-06-28')]

latestDate.head()
import plotly.graph_objects as go



totalCases = go.Figure(data=go.Choropleth(

    locations = latestDate['iso_alpha'],

    z = latestDate['Total Cases'],

    text = latestDate['Country'],

    colorscale = 'Reds',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Total Cases',

))



totalCases.update_layout(

    title="Country by Total Cases - 28/06/2020")

    

totalCases.show()
totalRecoveries = go.Figure(data=go.Choropleth(

    locations = latestDate['iso_alpha'],

    z = latestDate['Recovered'],

    text = latestDate['Country'],

    colorscale = 'Blues',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Recoveries',

))



totalRecoveries.update_layout(

    title_text='Country by Recoveries - 28/06/2020')

    

totalRecoveries.show()
totalDeaths = go.Figure(data=go.Choropleth(

    locations = latestDate['iso_alpha'],

    z = latestDate['Deaths'],

    text = latestDate['Country'],

    colorscale = 'blackbody',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Deaths from COVID-19',

))



totalDeaths.update_layout(title="Country by Deaths - 28/06/2020")

    

totalDeaths.show()
animation_dataset = dataset.sort_values(by=['Date'])

#Active Cases Animation

import plotly.express as px

ac_animation = px.choropleth(animation_dataset, locations="iso_alpha", animation_frame="Date", animation_group="Country", hover_name="Country", color="Active Cases")



ac_animation.update_layout(title="COVID-19 Active Cases from first occurrence to 7/01 per country")



ac_animation.show()
#Recovery Rate Animation



animation_dataset = animation_dataset[animation_dataset['Recovery Rate'] != "#DIV/0!"]

animation_dataset['Recovery Rate'] = animation_dataset['Recovery Rate'].str.rstrip('%').astype('float') / 100.0

animation_dataset = animation_dataset[animation_dataset['Death Rate'] != "#DIV/0!"]

animation_dataset['Death Rate'] = animation_dataset['Death Rate'].str.rstrip('%').astype('float') / 100.0





animation_dataset.head(10)
import plotly.express as px

recovery_animation = px.choropleth(animation_dataset, locations="iso_alpha", animation_frame="Date", animation_group="Country", hover_name="Country", color="Recovery Rate")



recovery_animation.update_layout(title="COVID-19 Recovery Rate from first occurrence to 7/01 per country", coloraxis_colorbar=dict(

    title="Recovery Rate (%)"))





recovery_animation.show()
import plotly.express as px

death_animation = px.choropleth(animation_dataset, locations="iso_alpha", animation_frame="Date", animation_group="Country", hover_name="Country", color="Death Rate")



death_animation.update_layout(title="COVID-19 Death Rate from first occurrence to 7/01 per country",coloraxis_colorbar=dict(

    title="Death Rate (%)"))





death_animation.show()