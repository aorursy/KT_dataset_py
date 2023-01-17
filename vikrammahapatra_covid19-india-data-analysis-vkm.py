# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
df_complete = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')

df_patient_wise = pd.read_csv('../input/covid19-corona-virus-india-dataset/patients_data.csv')





#date and state wise total

df = pd.DataFrame(df_complete.groupby(['Date','Name of State / UT'])

                  ['Total Confirmed cases (Indian National)'].sum()).reset_index()

df[df['Name of State / UT']=='Maharashtra']



#State wise Total till 29th March

df_stateWiseTot =  pd.DataFrame(df.groupby(['Name of State / UT'])

                                ['Total Confirmed cases (Indian National)'].sum()).reset_index()

df_stateWiseTot.sort_values('Total Confirmed cases (Indian National)', 

                            axis = 0, ascending = False, inplace = True, na_position ='last') 

df_stateWiseTot.nlargest(5,'Total Confirmed cases (Indian National)')
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

import folium

import seaborn as sns





#add Graph

fig1=go.Figure()

fig1.add_trace(go.Scatter(x=df[(df['Name of State / UT']=='Maharashtra') & 

                               (df['Date'] < '2020-03-29')

                              ]['Date'],

                          y=df[df['Name of State / UT']=='Maharashtra']['Total Confirmed cases (Indian National)'],

                          name='Maharashtra'

                               ))

fig1.add_trace(go.Scatter(x=df[(df['Name of State / UT']=='Kerala') & 

                               (df['Date'] < '2020-03-29')

                              ]['Date'],

                          y=df[df['Name of State / UT']=='Kerala']['Total Confirmed cases (Indian National)'],

                          name='Kerala'

                               ))

fig1.add_trace(go.Scatter(x=df[(df['Name of State / UT']=='Uttar Pradesh') & 

                               (df['Date'] < '2020-03-29')

                              ]['Date'],

                          y=df[df['Name of State / UT']=='Uttar Pradesh']['Total Confirmed cases (Indian National)'],

                          name='Uttar Pradesh'

                               ))

fig1.add_trace(go.Scatter(x=df[(df['Name of State / UT']=='Karnataka') & (df['Date'] < '2020-03-29') ]['Date'],

                                 y=df[df['Name of State / UT']=='Karnataka']['Total Confirmed cases (Indian National)'],

                          name='Karnataka'

                               ))

fig1.add_trace(go.Scatter(x=df[(df['Name of State / UT']=='Delhi') & (df['Date'] < '2020-03-29') ]['Date'],

                                 y=df[df['Name of State / UT']=='Delhi']['Total Confirmed cases (Indian National)'],

                          name='Delhi'

                               ))



fig1.layout.update(title_text='COVID-19 Top 4 State Wise Data in India',xaxis_showgrid=False, yaxis_showgrid=False, width=900,

        height=500,font=dict(

#         family="Courier New, monospace",

        size=12,

        color="white"

    ))

fig1.layout.plot_bgcolor = 'Black'

fig1.layout.paper_bgcolor = 'Black'

fig1.show()