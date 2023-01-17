# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
asean_countries_survey = ['Indonesia', 'Singapore', 'Thailand', 'Viet Nam', 'Malaysia', 'Philippines']

# Survey 2017

df_17 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding='ISO-8859-1')

df17_asean = df_17[df_17['Country'].isin(asean_countries_survey)]

df17_rest = df_17[~df_17['Country'].isin(asean_countries_survey)]

df_17['Area']=["Asean" if x in asean_countries_survey else "Others" for x in df_17['Country']]



# Survey 2018

df_18 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")

df18_asean = df_18[df_18['Q3'].isin(asean_countries_survey)]

df18_rest = df_18[~df_18['Q3'].isin(asean_countries_survey)]

df_18['Area']=["Asean" if x in asean_countries_survey else "Others" for x in df_18['Q3']]



# Survey 2019

df_19 = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

df_19 = df_19.drop(0, axis=0)

df19_asean = df_19[df_19['Q3'].isin(asean_countries_survey)]

df19_rest = df_19[~df_19['Q3'].isin(asean_countries_survey)]

df_19['Area']=["Asean" if x in asean_countries_survey else "Others" for x in df_19['Q3']]
tmp = df_19.Area.value_counts()

labels = (np.array(tmp.index))

sizes = (np.array((tmp / tmp.sum())*100))



trace = go.Pie(labels = labels, values = sizes)

layout = go.Layout(

    title = 'Asean Respondent VS The Rest of The World'

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = "Compare_Respondent")
tmp = df19_asean.Q3.value_counts()

labels = (np.array(tmp.index))

sizes = (np.array((tmp / tmp.sum())*100))



trace = go.Pie(labels = labels, values = sizes)

layout = go.Layout(

    title = 'Asean Respondent'

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = "Asean_Respondent")
import plotly.graph_objects as go



respond_2019_asean = df_19.Area.value_counts()['Asean']

respond_2018_asean = df_18.Area.value_counts()['Asean']

respond_2017_asean = df_17.Area.value_counts()['Asean']

data = [[2017, respond_2017_asean], [2018, respond_2018_asean], [2019, respond_2019_asean]]

custom = pd.DataFrame(data, columns = ['Year', 'Total_Asean_Respondent']) 



trace = go.Scatter(

    x = custom.Year,

    y = custom.Total_Asean_Respondent,

    mode = 'lines',

    name = 'Asean Respondent by Year'

)



layout = go.Layout(title = 'Asean Respondent by Year')

figure = go.Figure(data = trace, layout = layout)

figure.show()
respond_2019 = df_19['Q1'].count()

respond_2018 = df_18['Q1'].count()

respond_2017 = df_17['Country'].count()

data = [[2017, respond_2017], [2018, respond_2018], [2019, respond_2019]]

custom = pd.DataFrame(data, columns = ['Year', 'Total_Respondent']) 



trace = go.Scatter(

    x = custom.Year,

    y = custom.Total_Respondent,

    mode = 'lines',

    name = 'Total Respondent by Year'

)



layout = go.Layout(title = 'Total Respondent by Year')

figure = go.Figure(data = trace, layout = layout)

figure.show()
import plotly.graph_objs as go



asean_age_percentage = ((df_19.groupby('Area').get_group('Asean')['Q1'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Asean')['Q1'].sort_index() .count())*100)

others_age_percentage = ((df_19.groupby('Area').get_group('Others')['Q1'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Others')['Q1'].sort_index() .count())*100)

    

    

x = df_19.groupby('Area').get_group('Asean')['Q1'].value_counts().sort_index().index

y1 = asean_age_percentage

y2 = others_age_percentage





trace1 = go.Bar(

    x = x,

    y = y1,

    name = 'Asean',

    marker = dict(

        color='rgb(49,130,189)'

    )

)

trace2 = go.Bar(

    x = x,

    y = y2,

    name = 'Others',

    marker = dict(

        color='rgb(55, 83, 109)'

    )

)



layout = go.Layout(

    title = 'Age',

    xaxis=dict(tickangle=-45, title='Age by Total Respondent Each Area'),

    barmode='group',

    yaxis=dict(

        title='percentage',

    )

)

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

fig.show()
tmp = df19_asean.Q2.value_counts()

labels = (np.array(tmp.index))

sizes = (np.array((tmp / tmp.sum())*100))



trace = go.Pie(labels = labels, values = sizes)

layout = go.Layout(

    title = 'Gender of Asean Respondent'

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = "Gender_Asean_Respondent")
asean_education_percentage = ((df_19.groupby('Area').get_group('Asean')['Q4'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Asean')['Q4'].sort_index() .count())*100)

others_education_percentage = ((df_19.groupby('Area').get_group('Others')['Q4'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Others')['Q4'].sort_index() .count())*100)

    

    

x = df_19.groupby('Area').get_group('Asean')['Q4'].value_counts().sort_index().index

y1 = asean_education_percentage

y2 = others_education_percentage





trace1 = go.Bar(

    x = x,

    y = y1,

    name = 'Asean',

    marker = dict(

        color='rgb(49,130,189)'

    )

)

trace2 = go.Bar(

    x = x,

    y = y2,

    name = 'Others',

    marker = dict(

        color='rgb(55, 83, 109)'

    )

)



layout = go.Layout(

    title = 'Highest Education Level',

    xaxis=dict(tickangle=-20),

    barmode='group',

    yaxis=dict(

        title='percentage',

    )

)

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

fig.show()
Q13_1 = df19_asean.Q13_Part_1.value_counts()

Q13_2 = df19_asean.Q13_Part_2.value_counts()

Q13_3 = df19_asean.Q13_Part_3.value_counts()

Q13_4 = df19_asean.Q13_Part_4.value_counts()

Q13_5 = df19_asean.Q13_Part_5.value_counts()

Q13_6 = df19_asean.Q13_Part_6.value_counts()

Q13_7 = df19_asean.Q13_Part_7.value_counts()

Q13_8 = df19_asean.Q13_Part_8.value_counts()

Q13_9 = df19_asean.Q13_Part_9.value_counts()

Q13_10 = df19_asean.Q13_Part_10.value_counts()

Q13_11 = df19_asean.Q13_Part_11.value_counts()

Q13_12 = df19_asean.Q13_Part_12.value_counts()



Q13_index = [Q13_1.index[0], Q13_2.index[0], Q13_3.index[0], Q13_4.index[0],

            Q13_5.index[0], Q13_6.index[0], Q13_7.index[0], Q13_8.index[0],

            Q13_9.index[0], Q13_10.index[0], Q13_11.index[0], Q13_12.index[0]]

Q13_value = [Q13_1[0], Q13_2[0], Q13_3[0], Q13_4[0],

            Q13_5[0], Q13_6[0], Q13_7[0], Q13_8[0],

            Q13_9[0], Q13_10[0], Q13_11[0], Q13_12[0]]



Q13 = pd.Series(Q13_value, index = Q13_index)



tmp = Q13

labels = (np.array(tmp.index))

sizes = (np.array((tmp / tmp.sum())*100))



trace = go.Pie(labels = labels, values = sizes)

layout = go.Layout(

    title = 'Popular Platform To Study Data Science in Asean'

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = "Platform_Asean_Respondent")
Q13_1 = df19_rest.Q13_Part_1.value_counts()

Q13_2 = df19_rest.Q13_Part_2.value_counts()

Q13_3 = df19_rest.Q13_Part_3.value_counts()

Q13_4 = df19_rest.Q13_Part_4.value_counts()

Q13_5 = df19_rest.Q13_Part_5.value_counts()

Q13_6 = df19_rest.Q13_Part_6.value_counts()

Q13_7 = df19_rest.Q13_Part_7.value_counts()

Q13_8 = df19_rest.Q13_Part_8.value_counts()

Q13_9 = df19_rest.Q13_Part_9.value_counts()

Q13_10 = df19_rest.Q13_Part_10.value_counts()

Q13_11 = df19_rest.Q13_Part_11.value_counts()

Q13_12 = df19_rest.Q13_Part_12.value_counts()



Q13_index = [Q13_1.index[0], Q13_2.index[0], Q13_3.index[0], Q13_4.index[0],

            Q13_5.index[0], Q13_6.index[0], Q13_7.index[0], Q13_8.index[0],

            Q13_9.index[0], Q13_10.index[0], Q13_11.index[0], Q13_12.index[0]]

Q13_value = [Q13_1[0], Q13_2[0], Q13_3[0], Q13_4[0],

            Q13_5[0], Q13_6[0], Q13_7[0], Q13_8[0],

            Q13_9[0], Q13_10[0], Q13_11[0], Q13_12[0]]



Q13 = pd.Series(Q13_value, index = Q13_index)



tmp = Q13

labels = (np.array(tmp.index))

sizes = (np.array((tmp / tmp.sum())*100))



trace = go.Pie(labels = labels, values = sizes)

layout = go.Layout(

    title = 'Popular Platform To Study Data Science in Non-Asean'

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = "Platform_Non_Asean_Respondent")
asean_exp_percentage = ((df_19.groupby('Area').get_group('Asean')['Q15'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Asean')['Q15'].sort_index() .count())*100)

others_exp_percentage = ((df_19.groupby('Area').get_group('Others')['Q15'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Others')['Q15'].sort_index() .count())*100)

    

    

x = df_19.groupby('Area').get_group('Asean')['Q15'].value_counts().sort_index().index

y1 = asean_exp_percentage

y2 = others_exp_percentage





trace1 = go.Bar(

    x = x,

    y = y1,

    name = 'Asean',

    marker = dict(

        color='rgb(49,130,189)'

    )

)

trace2 = go.Bar(

    x = x,

    y = y2,

    name = 'Others',

    marker = dict(

        color='rgb(55, 83, 109)'

    )

)



layout = go.Layout(

    title = 'Experience Writing Code to Analyse Data',

    xaxis=dict(tickangle=-20),

    barmode='group',

    yaxis=dict(

        title='percentage',

    )

)

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

fig.show()
asean_job_percentage = df_19['Q5'].value_counts().sort_index()

    

    

x = df_19['Q5'].value_counts().sort_index().index

y = asean_job_percentage





trace = go.Bar(

    x = x,

    y = y,

    name = 'Asean',

    marker = dict(

        color='rgb(49,130,189)'

    )

)



layout = go.Layout(

    title = 'Job Oportunity',

    xaxis=dict(tickangle=-20),

    barmode='group',

    yaxis=dict(

        title='percentage',

    )

)

fig = go.Figure(data=trace, layout=layout)

fig.show()
asean_salary_percentage = ((df_19.groupby('Area').get_group('Asean')['Q10'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Asean')['Q10'].sort_index() .count())*100)

others_salary_percentage = ((df_19.groupby('Area').get_group('Others')['Q10'].value_counts().sort_index()  / 

                         df_19.groupby('Area').get_group('Others')['Q10'].sort_index() .count())*100)

    

    

x = df_19.groupby('Area').get_group('Asean')['Q10'].value_counts().sort_index().index

y1 = asean_salary_percentage

y2 = others_salary_percentage





trace1 = go.Bar(

    x = x,

    y = y1,

    name = 'Asean',

    marker = dict(

        color='rgb(49,130,189)'

    )

)

trace2 = go.Bar(

    x = x,

    y = y2,

    name = 'Others',

    marker = dict(

        color='rgb(55, 83, 109)'

    )

)



layout = go.Layout(

    title = 'Salary Overview',

    xaxis=dict(tickangle=-20),

    barmode='group',

    yaxis=dict(

        title='percentage',

    )

)

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

fig.show()