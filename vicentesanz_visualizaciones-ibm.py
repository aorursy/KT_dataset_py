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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
project= pd.read_csv(r"/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")


project.drop(["Over18", "EmployeeCount","EmployeeNumber","StandardHours"],axis="columns",inplace=True)


data = [go.Bar(
            x=project["Attrition"].value_counts().index.values,
            y= project["Attrition"].value_counts().values)]

py.iplot(data, filename='basic-bar')


attrition=['Yes', 'No']

fig = go.Figure([go.Bar(x=attrition, y=[237,1233])])
                                       
fig.show()

attrition = project[(project['Attrition'] != "No")]
no_attrition = project[(project['Attrition'] =="No")]

trace = go.Bar(x = (len(attrition), len(no_attrition)), y = ['Yes_attrition', 'No_attrition'], orientation = 'h', opacity = 0.8, marker=dict(
        color=['lime', 'mediumblue'],
        line=dict(color='#000000',width=1.5)))

layout = dict(title =  'Count of attrition variable')
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
trace1 = go.Pie(values  = attrition["Gender"].value_counts().values.tolist(),
                    labels  = attrition["Gender"].value_counts().keys().tolist(),
                    textfont=dict(size=15), opacity = 0.8,
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "attrition employes",
                    marker  = dict(colors = ("olive","hotpink"), line = dict(width = 1.5)))

trace2 = go.Pie(values  = no_attrition["Gender"].value_counts().values.tolist(),
                    labels  = no_attrition["Gender"].value_counts().keys().tolist(),
                    textfont=dict(size=15), opacity = 0.8,
                    hoverinfo = "label+percent+name",
                    marker  = dict(colors = ("olive","hotpink"), line = dict(width = 1.5)),
                    domain  = dict(x = [.52,1]),
                    name    = "Non attrition employes" )
    
layout = go.Layout(dict(title = "gender" + " distribution in employes attrition ",
                            annotations = [dict(text = "Yes_attrition",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .22, y = -0.1),
                                            dict(text = "No_attrition",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .8,y = -.1)]))
                                          

fig  = go.Figure(data = [trace1,trace2],layout = layout)
py.iplot(fig)

attrition

trace1 = go.Pie(values  = attrition["Department"].value_counts().values.tolist(),
                    labels  = attrition["Department"].value_counts().keys().tolist(),
                    textfont=dict(size=15), opacity = 0.8,
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "attrition employes",
                    marker  = dict(colors = ("magenta","seashell","bisque"), line = dict(width = 1.5)))

trace2 = go.Pie(values  = no_attrition["Department"].value_counts().values.tolist(),
                    labels  = no_attrition["Department"].value_counts().keys().tolist(),
                    textfont=dict(size=15), opacity = 0.8,
                    hoverinfo = "label+percent+name",
                    marker  = dict(colors = ("magenta","seashell"), line = dict(width = 1.5)),
                    domain  = dict(x = [.52,1]),
                    name    = "Non attrition employes" )
    
layout = go.Layout(dict(title = "Department" + " distribution in employes attrition ",
                            annotations = [dict(text = "Yes_attrition",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .22, y = -0.1),
                                            dict(text = "No_attrition",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .8,y = -.1)]))
                                          

fig  = go.Figure(data = [trace1,trace2],layout = layout)

py.iplot(fig)


project.hist(figsize=(20,20))
plt.show()
plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(project.loc[project['Attrition'] == 'No', 'Age'], label = 'Active Employee')
sns.kdeplot(project.loc[project['Attrition'] == 'Yes', 'Age'], label = 'Ex-Employees')
plt.xlim(left=18, right=60)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution in Percent by Attrition Status')
df_EducationField = pd.DataFrame(columns=["Field", "% of Leavers"])
i=0
for field in list(project['EducationField'].unique()):
    ratio = project[(project['EducationField']==field)&(project['Attrition']=="Yes")].shape[0] / project[project['EducationField']==field].shape[0]
    df_EducationField.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_EF = df_EducationField.groupby(by="Field").sum()
df_EF.plot.bar(title="% Attrition by Education",legend=False)
df_EF
df_Marital = pd.DataFrame(columns=["Marital Status", "% of Leavers"])
i=0
for field in list(project['MaritalStatus'].unique()):
    ratio = project[(project['MaritalStatus']==field)&(project['Attrition']=="Yes")].shape[0] / project[project['MaritalStatus']==field].shape[0]
    df_Marital.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_MF = df_Marital.groupby(by="Marital Status").sum()
df_MF.plot.bar(title="% Attrition by Marital Status",legend=False,colormap="summer")
df_BusinessTravel = pd.DataFrame(columns=["Business Travel", "% of Leavers"])
i=0
for field in list(project['BusinessTravel'].unique()):
    ratio = project[(project['BusinessTravel']==field)&(project['Attrition']=="Yes")].shape[0] / project[project['BusinessTravel']==field].shape[0]
    df_BusinessTravel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_BT = df_BusinessTravel.groupby(by="Business Travel").sum()
df_BT.plot.bar(title="% Attrition by Bussines Travel",legend=False,colormap="flag")