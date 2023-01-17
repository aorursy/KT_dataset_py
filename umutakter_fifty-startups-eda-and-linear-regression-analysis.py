import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB
startups = pd.read_csv("../input/50_Startups.csv")

df=startups.copy()
df.head()
df.info()
df.shape
df.isna().sum()
df = df.rename(columns = {"R&D Spend":"RdSpend","Marketing Spend":"MarketingSpend"})
df.head()
corr = df.corr()

corr
sns.heatmap(corr,annot=True)
fig = px.scatter(df, x="RdSpend", y="Profit",color="RdSpend")



fig.update_layout(

    title={

        'text': "RdSpend-Profit Scatter Plot",

        'y':0.95,

        'x':0.5

})



fig.show()
fig = px.scatter(df, x="RdSpend", y="Profit", trendline="ols",color="RdSpend")



fig.update_layout(

    title={

        'text': "RdSpend-Profit Regression Plot",

        'y':0.95,

        'x':0.5

})



fig.show()
df[df.Profit<=40000]
trace1 = go.Histogram(

    x=df.Administration,

    opacity=0.75,

    name = "Administration",

    marker=dict(color='rgba(255, 51, 0, 0.8)'))



trace2 = go.Histogram(

    x=df.RdSpend,

    opacity=0.75,

    name = "RdSpend",

    marker=dict(color='rgba(77, 77, 255, 0.8)'))



trace3 = go.Histogram(

    x=df.MarketingSpend,

    opacity=0.75,

    name = "MarketingSpend",

    marker=dict(color='rgba(0, 204, 0,0.8)'))



trace4 = go.Histogram(

    x=df.Profit,

    opacity=0.75,

    name = "Profit",

    marker=dict(color='rgba(0, 0, 102 0.8)'))



data = [trace1,trace2,trace3,trace4]

layout = go.Layout(barmode='group',

                   title=' Sayısal Değişkenlerin Dağılımı',

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df.describe().T
df.State.unique()
df["State"] = pd.Categorical(df["State"])

dfDummies = pd.get_dummies(df["State"],prefix="state")

dfDummies.head()
df = pd.concat([df, dfDummies],axis=1)

df.head()
df =df.drop(columns ="state_New York")

df.head()
df =df.drop(columns ="State")

df = df.rename(columns = {"state_California":"StateC","state_Florida":"StateF"})

df.head()
y = df['Profit']

X = df.drop(['Profit'], axis=1)
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
X_train.info()
X_test.info()
y_train
y_test
from sklearn.linear_model import LinearRegression



linear_regresyon = LinearRegression()
linear_regresyon.fit(X, y)
y_pred=linear_regresyon.predict(X)
df["y_pred"]=y_pred

df.head()
df["predFark"]=df["Profit"]-df["y_pred"]

df.head()
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(df["Profit"], df["y_pred"])

MSE
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(df["Profit"], df["y_pred"])

MAE
import math



RMSE = math.sqrt(MSE)

RMSE
linear_regresyon.score(X,y)