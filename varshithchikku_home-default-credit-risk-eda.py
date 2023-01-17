import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif


import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
color = sns.color_palette("husl", 8)

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import squarify
from PIL import Image
app_test =  pd.read_csv('../input/application_test.csv')
app_train = pd.read_csv('../input/application_train.csv')
app_train.info()
app_train.columns
app_train.head()
#Univariate Analysis
#Total Number of Target Values

plt.figure(figsize = (6,4))
sns.countplot(app_train['TARGET'])
plt.ylabel('TARGET')
plt.xlabel('Count')
plt.title('Total Count of Target Values')
app_train.head()
temp = app_train["NAME_CONTRACT_TYPE"].value_counts()
labels = temp.index
values = temp.values
trace = go.Pie(labels=labels, values=values)
iplot([trace], filename='basic_pie_chart')
temp = app_train["CODE_GENDER"].value_counts()
labels = temp.index
values = temp.values
trace = go.Pie(labels=labels, values=values)
iplot([trace], filename='basic_pie_chart')
temp1  = app_train["FLAG_OWN_CAR"].value_counts()
temp2 = app_train["FLAG_OWN_REALTY"].value_counts()
fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [0, .48]},
      "name": "FLAG OWN CAR",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": temp2.values,
      "labels": temp2.index,
      "text":"FLAG OWN REALITY",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "FLAG OWN REALTY",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"OWNS A CAR OR REALITY",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "CAR",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "REALITY",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')
temp1  = app_train["NAME_TYPE_SUITE"].value_counts()
temp2 = app_train["NAME_INCOME_TYPE"].value_counts()
fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [0, .48]},
      "name": "TYPE SUITE",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": temp2.values,
      "labels": temp2.index,
      "text":"INCOME TYPE",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "FLAG OWN REALTY",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"INCOME AND TYPE SUITE",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "TYPE",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "INCOME",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')
temp1  = app_train["NAME_FAMILY_STATUS"].value_counts()
temp2 = app_train["NAME_HOUSING_TYPE"].value_counts()
fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [0, .48]},
      "name": "FAMILY_STATUS",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": temp2.values,
      "labels": temp2.index,
      "text":"INCOME TYPE",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "HOUSING TYPE",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"FAMILY TYPE AND HOUSING STATUS",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "FAMILY",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "HOUSE",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')
#EDUCATION STATUS

temp1  = app_train["NAME_EDUCATION_TYPE"].value_counts()

fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [.30, .78]},
      "name": "FLAG OWN CAR",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"EDUCATION STATUS",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "EDUCATION",
                "x": 0.54,
                "y": 0.5
            },
           
        ]
    }
}
py.iplot(fig, filename='donut')
temp  = app_train["OCCUPATION_TYPE"].value_counts()

trace0 = go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='OCCUPATION STATUS',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='text-hover-bar')
acc2 = app_train.groupby('NAME_CONTRACT_TYPE').TARGET.sum()
print(acc2)
#CONTRACT TYPE vs PAID RATIO

grade = pd.DataFrame(app_train['NAME_CONTRACT_TYPE'].value_counts()).reset_index()
grade.columns = ['project_grade', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('NAME_CONTRACT_TYPE').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['project_grade'],
    y=acc2['TARGET'],
    name='NOT PAID'
)
trace2 = go.Bar(
    x=grade['project_grade'],
    y=acc2['accepted'],
    name='PAID'
)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title = 'Contract type to Paid Ratio'
)

fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='stacked-bar')

#CONTRACT TYPE vs PAID RATIO

grade = pd.DataFrame(app_train['CODE_GENDER'].value_counts()).reset_index()
grade.columns = ['gender', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('CODE_GENDER').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['gender'],
    y=acc2['TARGET'],
    name='NOT PAID'
)
trace2 = go.Bar(
    x=grade['gender'],
    y=acc2['accepted'],
    name='PAID'
)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title = 'Gender to Paid Ratio'
)

fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='stacked-bar')

#CONTRACT TYPE vs PAID RATIO

grade = pd.DataFrame(app_train['FLAG_OWN_CAR'].value_counts()).reset_index()
grade.columns = ['contract_type', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('FLAG_OWN_CAR').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['contract_type'],
    y=acc2['TARGET'],
    name='NOT PAID'
)
trace2 = go.Bar(
    x=grade['contract_type'],
    y=acc2['accepted'],
    name='PAID'
)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title = 'Owning a CAR vs PAID'
)

fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='stacked-bar')

#Owing A Reality vs PAID RATIO

grade = pd.DataFrame(app_train['FLAG_OWN_REALTY'].value_counts()).reset_index()
grade.columns = ['reality', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('FLAG_OWN_REALTY').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['reality'],
    y=acc2['TARGET'],
    name='NOT PAID'
)
trace2 = go.Bar(
    x=grade['reality'],
    y=acc2['accepted'],
    name='PAID'
)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title = 'Owning a REALTY vs PAID'
)

fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='stacked-bar')

#TYPE OF SUITE vs PAID RATIO

grade = pd.DataFrame(app_train['NAME_TYPE_SUITE'].value_counts()).reset_index()
grade.columns = ['gender', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('NAME_TYPE_SUITE').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['gender'],
    y=acc2['TARGET'],
    name='NOT PAID'
)
trace2 = go.Bar(
    x=grade['gender'],
    y=acc2['accepted'],
    name='PAID'
)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title = 'Type Of Suite vs PAID'
)

fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='stacked-bar')

grade = pd.DataFrame(app_train['NAME_INCOME_TYPE'].value_counts()).reset_index()
grade.columns = ['income', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('NAME_INCOME_TYPE').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['income'],
    y=acc2['accepted'],
    name='PAID',
    text='PAID',
    textposition = 'auto',
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace2 = go.Bar(
     x=grade['income'],
    y=acc2['TARGET'],
    text= 'NOT PAID',
    name = 'NOT PAID',
    textposition = 'auto',
    marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

data = [trace1,trace2]

layout = go.Layout(
        title = 'Income type to Repaying Rate'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar-direct-labels')
grade = pd.DataFrame(app_train['NAME_EDUCATION_TYPE'].value_counts()).reset_index()
grade.columns = ['income', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('NAME_EDUCATION_TYPE').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['income'],
    y=acc2['accepted'],
    name='PAID',
    text='PAID',
    textposition = 'auto',
    marker=dict(
        color='rgba(50, 171, 96, 1.0)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace2 = go.Bar(
     x=grade['income'],
    y=acc2['TARGET'],
    text= 'NOT PAID',
    name = 'NOT PAID',
    textposition = 'auto',
    marker=dict(
        color='rgba(245, 246, 249)',
        line=dict(
            color='rgba(245, 246, 249)',
            width=1.5),
        ),
    opacity=0.6
)

data = [trace1,trace2]

layout = go.Layout(
        title = 'Education status to Repaying Rate'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar-direct-labels')
grade = pd.DataFrame(app_train['NAME_FAMILY_STATUS'].value_counts()).reset_index()
grade.columns = ['income', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('NAME_FAMILY_STATUS').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['income'],
    y=acc2['accepted'],
    name='PAID',
    text='PAID',
    textposition = 'auto',
    marker=dict(
        color='rgba(50, 171, 96, 1.0)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

trace2 = go.Bar(
     x=grade['income'],
    y=acc2['TARGET'],
    text= 'NOT PAID',
    name = 'NOT PAID',
    textposition = 'auto',
    marker=dict(
        color='rgba(245, 246, 249)',
        line=dict(
            color='rgba(245, 246, 249)',
            width=1.5),
        ),
    opacity=0.6
)

data = [trace1,trace2]

layout = go.Layout(
        title = 'Family status to Repaying Rate'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar-direct-labels')
#TYPE OF INCOME vs PAID RATIO

grade = pd.DataFrame(app_train['NAME_HOUSING_TYPE'].value_counts()).reset_index()
grade.columns = ['gender', 'count']
grade = grade.reset_index().drop('index', axis=1)

acc2 = app_train.groupby('NAME_HOUSING_TYPE').TARGET.sum()
acc2.columns = ['approved', 'count']
acc2 = acc2.reset_index()

acc2.sort_values('TARGET', ascending =False, inplace = True)
acc2.reset_index(inplace = True)
acc2.drop('index', axis = 1)

acc2['accepted'] = grade['count'] - acc2['TARGET']
acc2.drop('index', axis = 1)

trace1 = go.Bar(
    x=grade['gender'],
    y=acc2['TARGET'],
    name='NOT PAID'
)
trace2 = go.Bar(
    x=grade['gender'],
    y=acc2['accepted'],
    name='PAID'
)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title = 'Type Of Income vs Paying Ratio'
)

fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='stacked-bar')

app_train.iloc[:, 10:20].head()
bur = pd.read_csv('../input/bureau.csv')
bur.shape
bur['DAYS_CREDIT'].plot('hist')
temp = bur["CREDIT_ACTIVE"].value_counts()
labels = temp.index
values = temp.values
colors = [ '#3498DB','#B3B6B7', '#E1396C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
layout = go.Layout(
    title = 'Status Of The Credit'
)
fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='basic_pie_chart')
#Credit Currency Status in Bureau

temp  = bur["CREDIT_CURRENCY"].value_counts()

fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [.30, .78]},
      "name": "Credit Currency",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Credit Currency",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "CURRENCY",
                "x": 0.54,
                "y": 0.5
            },
           
        ]
    }
}
py.iplot(fig, filename='donut')
temp  = bur["CREDIT_TYPE"].value_counts()

colors = ['#B3B6B7', '#3498DB', '#E1396C', '#D0F9B1']

trace0 = go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(
        color=colors,
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='type of credit',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='text-hover-bar')
bur.head()
bur.shape
bur.AMT_ANNUITY.mean()
trace1 = {"x": bur['AMT_ANNUITY'].head(100000),
          
          "marker": {"color": '#5DADE2', "size": 12},
          "mode": "markers",
          "name": "ANNUITY",
          "type": "scatter"
}


layout = {"title": "ANNITY AMOUNT",
          "xaxis": {"title": "ANNITY AMOUNT", },
          "yaxis": {"title": "Frequency"}}

fig = go.Figure(data=[trace1], layout=layout)
py.iplot(fig, filename='basic_dot-plot')
# #Removing Outliers using Strandard Deviation and Normal Distribution
# #The Outliers are eliminatin the points that were above (Mean - 2*Strandard Deviation)
# mean= bur['AMT_ANNUITY'].mean()
# sd= bur['AMT_ANNUITY'].std()

# p = (mean - 2*sd)

# #bur['AMT_ANNUITY'].apply(lambda x : p.replace(for x in ) in x < p, p, inplace = True)

# for i in bur.AMT_ANNUITY:
#     if i>p:
#         bur.at[i] = p
#bur.style.set_precision(2)
bur.head()
ccbalance = pd.read_csv('../input/credit_card_balance.csv')
ccbalance.shape
#EDUCATION STATUS

temp1  = ccbalance["NAME_CONTRACT_STATUS"].value_counts()

fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [.30, .78]},
      "name": "FLAG OWN CAR",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"CONTRACT STATUS",
        "annotations": [
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "CONTRACT",
                "x": 0.54,
                "y": 0.5
            },
           
        ]
    }
}
py.iplot(fig, filename='donut')
ccbalance['AMT_BALANCE'].plot('hist')
ccbalance.iloc[:, 0:10].head()
iap = pd.read_csv('../input/installments_payments.csv')
iap.shape
temp = iap["NUM_INSTALMENT_VERSION"].value_counts()
labels = temp.index
values = temp.values
colors = [ '#3498DB','#B3B6B7', '#E1396C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
layout = go.Layout(
    title = 'Status Of The Credit'
)
fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='basic_pie_chart')
#iap['NUM_INSTALMENT_NUMBER'].plot('hist')
#taking too much time
'''
x = iap['NUM_INSTALMENT_NUMBER']
data = [go.Histogram(x=x,
                     histnorm='probability')]

layout = go.Layout(
    title='Instalment Number',
    xaxis=dict(
        title='Value'
    ),
    yaxis=dict(
        title='Frequency'
    )
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='normalized histogram')
'''
iap.head()
pcb = pd.read_csv('../input/POS_CASH_balance.csv')
pcb.shape
#EDUCATION STATUS

temp1  = pcb["NAME_CONTRACT_STATUS"].value_counts()

colors = [ '#3498DB','#B3B6B7', '#E1396C', '#D0F9B1']

fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [.30, .78]},
      "name": "FLAG OWN CAR",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
      
    }],
  "layout": {
        "title":"CONTRACT STATUS",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "CONTRACT",
                "x": 0.54,
                "y": 0.5
            },
           
        ]
    }
}
py.iplot(fig, filename='donut')
pcb.isnull().sum().sum()
pcb.head()
pa = pd.read_csv('../input/previous_application.csv')
pa.shape
temp = pa["NAME_CONTRACT_TYPE"].value_counts()
labels = temp.index
values = temp.values
colors = [ '#CA6F1E','#239B56', '#E1396C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
layout = go.Layout(
    title = 'Name Of The Contract Type'
)
fig = go.Figure(data=[trace], layout=layout)


iplot(fig, filename='basic_pie_chart')
temp  = pa["NAME_CASH_LOAN_PURPOSE"].value_counts()

colors = ['#48C9B0', '#F8C471', '#E1396C', '#D0F9B1']

trace0 = go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(
        color=colors,
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='Purpose Of The Cash Loan',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='text-hover-bar')
temp = pa["NAME_CONTRACT_STATUS"].value_counts()
labels = temp.index
values = temp.values
colors = [ '#CA6F1E','#239B56', '#E1396C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
layout = go.Layout(
    title = 'Status Of The Contract'
)
fig = go.Figure(data=[trace], layout=layout)


iplot(fig, filename='basic_pie_chart')
pa.isnull().sum().sum()
img = Image.open('1.png')
img.show()

#Label Encoding for Application train&test
cols = []
for i in app_train.columns:
    if app_train[i].dtype == 'object':
        if len(list(app_train[i].unique())) >=2:
            #cols.append(i)
            le = preprocessing.LabelEncoder()
            le.fit(app_train[i].astype(str))
            app_train[i] = le.transform(app_train[i].astype(str))
            app_test[i] = le.transform(app_test[i].astype(str))
    
# for col in cols:
#     le = preprocessing.LabelEncoder()
#     le.fit(app_train[col].astype(str))
#     app_train = le.transform(app_train[col].astype(str))
#     app_test = le.transform(app_test[col].astype(str))
null_values = app_train.isnull().sum()
null_values = pd.DataFrame(null_values)
null_values.columns = ['values']

only_null_values = null_values[null_values.values != 0]
only_null_values['per_mis_val'] = (only_null_values/len(app_train))*100
only_null_values.sort_values('values', ascending = False).head()


#from here: https://www.kaggle.com/codename007/home-credit-complete-eda-feature-importance
data = [
    go.Heatmap(
        z= app_train.corr().values,
        x= app_train.columns.values,
        y= app_train.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Corelation features of app_train',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 900,
margin=dict(
    l=240,
),)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
cols = only_null_values.index
app_train[cols[0:10]].head()
#If the percentage of NULL values more than 30% of then adding some otherindividual variable
#replacing other values with mean or medium

cols =  only_null_values[only_null_values.per_mis_val> 30.0].index

for col in cols:
    if app_train[col].dtype == 'float64':
        app_train[col].fillna('-9999', inplace = True)
cols2  = []
for item in only_null_values.index:
    if item in cols:
        pass
    else:
       # do something
       cols2.append(item)
    
for  i in cols2:
    if app_train[i].dtype == 'float64':
        app_train[i].fillna(app_train[i].mean(), inplace = True)
# for i in cols2:
#     if app_train[i].dtype == 'float64':
#         app_train[i].fillna(app_train[i].mean(), inplace = True)
null_values_test = app_test.isnull().sum()
null_values_test = pd.DataFrame(null_values_test)
null_values_test.columns = ['values']

only_null_values_test = null_values_test[null_values_test.values != 0]
only_null_values_test['per_mis_val'] = (only_null_values_test/len(app_test))*100
only_null_values_test.sort_values('values', ascending = False).head(20)

cols_test =  only_null_values_test[only_null_values_test.per_mis_val> 30.0].index

for col in cols_test:
    if app_test[col].dtype == 'float64':
        app_test[col].fillna('-9999', inplace = True)
        
cols2_test  = []
for item in only_null_values_test.index:
    if item in cols_test:
        pass
    else:
       # do something
       cols2_test.append(item)
    
for  i in cols2_test:
    if app_test[i].dtype == 'float64':
        app_test[i].fillna(app_test[i].mean(), inplace = True)
# for i in cols2:
# for  i in only_null_values.index:
#     if app_train[i].dtype == 'float64':
#         app_train[i].fillna(app_train[i].mean(), inplace = True)
        
# for  i in only_null_values_test.index:
#     if app_test[i].dtype == 'float64':
#         app_test[i].fillna(app_test[i].mean(), inplace = True)
#Label Encoding for Bureau
for i in bur.columns:
    if bur[i].dtype == 'object':
        if len(list(bur[i].unique())) >=2:
            #cols.append(i)
            le = preprocessing.LabelEncoder()
            le.fit(bur[i].astype(str))
            bur[i] = le.transform(bur[i].astype(str))
    

#Null Values for Bereau
null_values_bur = bur.isnull().sum()
null_values_bur = pd.DataFrame(null_values_bur)
null_values_bur.columns = ['values']

only_null_values_bur = null_values_bur[null_values_bur.values != 0]
only_null_values_bur['per_mis_val'] = (only_null_values_bur/len(bur))*100
only_null_values_bur.sort_values('values', ascending = False).head(20)

cols_bur =  only_null_values_bur[only_null_values_bur.per_mis_val> 30.0].index

for col in cols_bur:
    if bur[col].dtype == 'float64':
        bur[col].fillna('-9999', inplace = True)
        
cols2_bur  = []
for item in only_null_values_bur.index:
    if item in cols_bur:
        pass
    else:
       # do something
       cols2_bur.append(item)
    
for  i in cols2_bur:
    if bur[i].dtype == 'float64':
        bur[i].fillna(bur[i].mean(), inplace = True)
# for i in cols2:
bur.isnull().sum().sum()
#Label Encoding for Bureau
for i in ccbalance.columns:
    if ccbalance[i].dtype == 'object':
        if len(list(ccbalance[i].unique())) >=2:
            #cols.append(i)
            le = preprocessing.LabelEncoder()
            le.fit(ccbalance[i].astype(str))
            ccbalance[i] = le.transform(ccbalance[i].astype(str))
    

null_values_bal = ccbalance.isnull().sum()
null_values_bal = pd.DataFrame(null_values_bal)
null_values_bal.columns = ['values']

only_null_values_bal = null_values_bal[null_values_bal.values != 0]
only_null_values_bal['per_mis_val'] = (only_null_values_bal/len(ccbalance))*100
only_null_values_bal.sort_values('values', ascending = False).head(20)

#Replacing all NULL values with the mean
for  i in only_null_values_bal.index:
    if ccbalance[i].dtype == 'float64':
        ccbalance[i].fillna(ccbalance[i].mean(), inplace = True)

ccbalance.dropna(inplace = True)
#bur.style.set_precision(2)
#Label Encoding for Installments and Payments
for i in iap.columns:
    if iap[i].dtype == 'object':
        if len(list(iap[i].unique())) >=2:
            #cols.append(i)
            le = preprocessing.LabelEncoder()
            le.fit(iap[i].astype(str))
            iap[i] = le.transform(iap[i].astype(str))
    

#Percentage of Missing Values in the Installments and Payments
null_values_iap = iap.isnull().sum()
null_values_iap = pd.DataFrame(null_values_iap)
null_values_iap.columns = ['values']

only_null_values_iap = null_values_iap[null_values_iap.values != 0]
only_null_values_iap['per_mis_val'] = (only_null_values_iap/len(iap))*100
only_null_values_iap.sort_values('values', ascending = False).head(20)

# #Replacing all NULL values with the mean
for  i in only_null_values_iap.index:
    if iap[i].dtype == 'float64':
        iap[i].fillna(iap[i].mean(), inplace = True)

iap.dropna(inplace = True)
#bur.style.set_precision(2)
iap.isnull().sum().sum()
#Label Encoding for Bureau
for i in pcb.columns:
    if pcb[i].dtype == 'object':
        if len(list(pcb[i].unique())) >=2:
            #cols.append(i)
            le = preprocessing.LabelEncoder()
            le.fit(pcb[i].astype(str))
            pcb[i] = le.transform(pcb[i].astype(str))
    

null_values_pcb = pcb.isnull().sum()
null_values_pcb = pd.DataFrame(null_values_pcb)
null_values_pcb.columns = ['values']

only_null_values_pcb = null_values_pcb[null_values_pcb.values != 0]
only_null_values_pcb['per_mis_val'] = (only_null_values_pcb/len(pcb))*100
only_null_values_pcb.sort_values('values', ascending = False).head(20)

#Replacing all NULL values with the mean
for  i in only_null_values_pcb.index:
    if pcb[i].dtype == 'float64':
        pcb[i].fillna(pcb[i].mean(), inplace = True)

pcb.dropna(inplace = True)
#bur.style.set_precision(2)
#Label Encoding for Bureau
for i in pa.columns:
    if pa[i].dtype == 'object':
        if len(list(pa[i].unique())) >=2:
            #cols.append(i)
            le = preprocessing.LabelEncoder()
            le.fit(pa[i].astype(str))
            pa[i] = le.transform(pa[i].astype(str))
    

#Null Values and Percentage
null_values_pa = pa.isnull().sum()
null_values_pa = pd.DataFrame(null_values_pa)
null_values_pa.columns = ['values']

only_null_values_pa = null_values_pa[null_values_pa.values != 0]
only_null_values_pa['per_mis_val'] = (only_null_values_pa/len(pa))*100
only_null_values_pa.sort_values('values', ascending = False).head(20)

cols_pa =  only_null_values_pa[only_null_values_pa.per_mis_val> 30.0].index

for col in cols_pa:
    if pa[col].dtype == 'float64':
        pa[col].fillna('-9999', inplace = True)
        
cols2_pa  = []
for item in only_null_values_pa.index:
    if item in cols_pa:
        pass
    else:
       # do something
       cols2_pa.append(item)
    
for  i in cols2_pa:
    if pa[i].dtype == 'float64':
        pa[i].fillna(pa[i].mean(), inplace = True)

pa = pa.drop(['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'NAME_TYPE_SUITE'], axis = 1)
pa.isnull().sum().sum()