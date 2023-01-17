import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy  as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedShuffleSplit

from sklearn.metrics import classification_report,confusion_matrix

from pandas_profiling import ProfileReport

from sklearn.cluster import KMeans

from datetime import datetime, timedelta,date
!pip install chart_studio
import chart_studio.plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go
df_data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df_data.head()
profile = ProfileReport(df_data,title='Profile Report')

profile.to_widgets()
def label_encoder(data):

    le = preprocessing.LabelEncoder()

    data = le.fit_transform(data)

    return data
pyoff.init_notebook_mode()

def ploting(data,col):

    df_plot = data.groupby(col).Churn.mean().reset_index()

    plot_data = [

        go.Bar(

            x=df_plot[col],

            y=df_plot['Churn'],

            width = [0.2, 0.2, 0.2, 0.2],

            marker=dict(

            color=['green','blue','red','yellow'])

        )

    ]



    plot_layout = go.Layout(

            xaxis={"type": "category"},

            yaxis={"title": "Churn Rate"},

            title=col,

            plot_bgcolor  = 'rgb(243,243,243)',

            paper_bgcolor  = 'rgb(243,243,243)',

        )

    fig = go.Figure(data=plot_data, layout=plot_layout)

    pyoff.iplot(fig)

    pass
category_cols = ['gender', 'Partner', 'Dependents',

        'PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod']

df_data['Churn'] = label_encoder(df_data['Churn'])

df_data.head()
for col in category_cols:

    ploting(df_data,col)
quantile_list = [0, .25, .5, .75, 1.]

quantiles = df_data['tenure'].quantile(quantile_list)

quantiles
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

df_data['tenure_quantiles'] = pd.qcut(df_data['tenure'],q=quantile_list, labels=quantile_labels)

df_data.head()
ploting(df_data,'tenure_quantiles')
df_plot = df_data.groupby('tenure').Churn.mean().reset_index()





plot_data = [

    go.Scatter(

        x=df_plot['tenure'],

        y=df_plot['Churn'],

        mode='markers',

        name='Low',

        marker= dict(size= 8,

            line= dict(width=2),

            color= 'green',

            opacity= 0.9

           ),

    )

]



plot_layout = go.Layout(

        yaxis= {'title': "Churn Rate"},

        xaxis= {'title': "Tenure"},

        title='Tenure based Churn rate',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
quantile_list = [0, .25, .5, .75, 1.]

quantiles = df_data['MonthlyCharges'].quantile(quantile_list)

quantiles
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

df_data['MonthlyCharges_quantiles'] = pd.qcut(df_data['MonthlyCharges'],q=quantile_list, labels=quantile_labels)

df_data.head()
ploting(df_data,'MonthlyCharges_quantiles')
df_plot = df_data.copy()

df_plot['MonthlyCharges'] = df_plot['MonthlyCharges'].astype(int)

df_plot = df_plot.groupby('MonthlyCharges').Churn.mean().reset_index()





plot_data = [

    go.Scatter(

        x=df_plot['MonthlyCharges'],

        y=df_plot['Churn'],

        mode='markers',

        name='Low',

        marker= dict(size= 8,

            line= dict(width=2),

            color= 'red',

            opacity= 0.8

           ),

    )

]



plot_layout = go.Layout(

        yaxis= {'title': "Churn Rate"},

        xaxis= {'title': "Monthly Charges"},

        title='Monthly Charge vs Churn rate',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
df_data.isna().sum()
df_data = df_data.dropna()
df_data['TotalCharges'] = df_data['TotalCharges'].astype('float64')
quantile_list = [0, .25, .5, .75, 1.]

quantiles = df_data['TotalCharges'].quantile(quantile_list)

quantiles
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

df_data['TotalCharges_quantiles'] = pd.qcut(df_data['TotalCharges'],q=quantile_list, labels=quantile_labels)

df_data.head()
ploting(df_data,'TotalCharges_quantiles')