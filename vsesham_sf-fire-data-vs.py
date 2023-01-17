# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading csv file using pandas read_csv method

data_input = pd.read_csv('../input/fire-department-calls-for-service.csv')



# Creating a dataframe with call count by call type

all_call_types = data_input['Call Type'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)



# Creating a dataframe with top 5 call types

top_5_call_types = all_call_types.head(5)



# Plotting a bar chart for top 5 call types

# converting the top 5 medical incidents series to dataframe and assign columns names 

top5_df = pd.DataFrame({'CallType':top_5_call_types.index, 'IncidentCount':top_5_call_types.values})



data_top5 = [go.Bar(x=top5_df.CallType, y=top5_df.IncidentCount)]



# specify the layout of figure

layout_top5 = dict(title = "Top 5 Call Types by Incidents",

              xaxis= dict(title= 'Call Type',ticklen= 1,zeroline= False))



# create and show figure

fig_top5 = dict(data = data_top5, layout = layout_top5)

iplot(fig_top5)



# create a dataset for Call Type = 'Medical Incident'

medical_incidents = data_input.loc[data_input['Call Type'] == 'Medical Incident']



# Convert 'Received DtTm' to datetime type

medical_incidents['Received DtTm'] = pd.to_datetime(medical_incidents['Received DtTm'])



# Extract Year from 'Received DtTm' and assign it to 'Received Year' column

medical_incidents['Received Year'] = medical_incidents['Received DtTm'].dt.year



#Calculate no of medical incidents occured in each year

mi_count_by_year = medical_incidents['Received Year'].value_counts(sort=False)



# converting the medical incidents by year series to dataframe and assign columns names 

mi_df = pd.DataFrame({'ReceivedYear':mi_count_by_year.index, 'IncidentCount':mi_count_by_year.values})



data = [go.Scatter(x=mi_df.ReceivedYear, y=mi_df.IncidentCount)]



# specify the layout of figure

layout = dict(title = "Number of Medical Incidents per Year",

              xaxis= dict(title= 'Year',ticklen= 2,zeroline= False))



# create and show figure

fig = dict(data = data, layout = layout)

iplot(fig)


