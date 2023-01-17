# Importing Libraries and dataset
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
#print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings(action='once')

df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
#df = pd.read_csv("bus-breakdown-and-delays.csv")

dtype_structure = {
    "category":["School_Year", "Busbreakdown_ID", "Run_Type", "Bus_No", "Route_Number", "Reason", 
                "Schools_Serviced", "Boro", "Bus_Company_Name", "Incident_Number", 
                "Breakdown_or_Running_Late", "School_Age_or_PreK"],
    #"float":   ["How_Long_Delayed"], # This will need to be cleaned later
    "int":     ["Number_Of_Students_On_The_Bus"],
    "datetime64":["Occurred_On", "Created_On", "Informed_On", "Last_Updated_On"],
    "bool":    ["Has_Contractor_Notified_Schools", "Has_Contractor_Notified_Parents", "Have_You_Alerted_OPT"],
    "object":  []    
}

# Dropping data for school year 2019-2020
df.drop(df[df.School_Year == '2019-2020'].index, inplace=True)

# Converting to datetime & creating a new column
df['Occurred_On'] = pd.to_datetime(df['Occurred_On'])
df['YearMonthDay'] = df['Occurred_On'].apply(lambda x:x.strftime('%Y%m%d'))

# Plotly - Import
import plotly.plotly as py
import plotly.graph_objs as go
import pylab
# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Average incidents per School Day
TotalIncidentsPerSchoolYear = list(df.groupby(['School_Year'])['School_Year'].count())
TotalServiceDaysPerSchoolYear = list(df.groupby(['School_Year'])['YearMonthDay'].nunique())

AverageIncidentSchoolYear =[]
for i in range(len(TotalIncidentsPerSchoolYear)):
    AverageIncidentSchoolYear.append(round(TotalIncidentsPerSchoolYear[i]/TotalServiceDaysPerSchoolYear[i],0))

col_names =  ['School_Year', 'AverageIncidentPerSchoolDay']
df_Graph  = pd.DataFrame(columns = col_names)
df_Graph['School_Year'] = list(df.groupby(['School_Year'])['YearMonthDay'].nunique().index)
df_Graph['AverageIncidentPerSchoolDay'] = AverageIncidentSchoolYear

# Plotting - Thanks to Jonathan Leon Seattle, WA, USA. - Week 2
data_1 = go.Bar(
            x=df_Graph['School_Year'],
            y=df_Graph['AverageIncidentPerSchoolDay'],
            marker=dict(
                color=['rgba(255,255,109,1)', 'rgba(219,109,0,1)',
               'rgba(182,219,255,1)', 'rgba(255,182,219,1)',
               ]),
    )

data_2 = go.Bar(
            x = df['Reason'].value_counts()[:5].index,
            y = round(df['Reason'].value_counts()[:5]/df['YearMonthDay'].nunique(),0),
            marker=dict(
                color=['rgba(0,109,219,1)', 'rgba(0,109,219,1)',
               'rgba(0,109,219,1)', 'rgba(0,109,219,1)',
               'rgba(0,109,219,1)',     
               ]),
        
    )

data = [data_1, data_2]

layout = dict(
    title='NY School Bus Incidents',
    titlefont=dict(
        size=24,
        family="Arial, Raleway, Roman"
    ),
    autosize=True,
    hovermode='closest',
    )

annotations =  [
    {
      "x": 0.2, 
      "y": 1.0, 
      "font": {"size": 16, "family":"Arial, Raleway, Roman"}, 
      "showarrow": False, 
      "text": "<br>Average Number of Incidents <br> per School Day", 
      "xanchor": "center", 
      "xref": "paper", 
      "yanchor": "bottom", 
      "yref": "paper"
    }, 
    {
      "x": 0.75, 
      "y": 1.0, 
      "font": {"size": 16, "family":"Arial, Raleway, Roman"}, 
      "showarrow": False, 
      "text": "<br>Top 5 Cause of Incidents <br> per School Day", 
      "xanchor": "center", 
      "xref": "paper", 
      "yanchor": "bottom", 
      "yref": "paper",
    }
]



layout['annotations'] = annotations

fig = dict(data=data, layout=layout)
fig['layout'].update(showlegend=False, plot_bgcolor='#C7A575', paper_bgcolor='#C7A575', font=dict(color= 'black'))

iplot(fig)
