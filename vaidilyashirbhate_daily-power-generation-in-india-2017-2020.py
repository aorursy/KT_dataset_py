import pandas as pd

from plotly.offline import iplot

import plotly.graph_objs as go

import plotly.io as pio

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
pio.templates.default="plotly_dark"
df=pd.read_csv("../input/daily-power-generation-in-india-20172020/file_02.csv")
df.head()
df['Date']=pd.to_datetime(df['Date'])
df.info()
df["Thermal Generation Actual (in MU)"]=df["Thermal Generation Actual (in MU)"].apply(lambda x:float(''.join(map(str,x.split(',')))))

df["Thermal Generation Estimated (in MU)"]=df["Thermal Generation Estimated (in MU)"].apply(lambda x:float(''.join(map(str,x.split(',')))))
def plot_graph(agg_dir):

    df_group=df.groupby(["Date"]).agg(agg_dir)

    columns=df_group.columns

    trace1=go.Scatter(x=df_group.index,y=df_group[columns[0]],mode="lines",name=columns[0],opacity=0.8)

    trace2=go.Scatter(x=df_group.index,y=df_group[columns[1]],mode="lines",name=columns[1],opacity=0.8)

    data = [trace1, trace2]

    layout = dict(

                  title = 'Total '+columns[0].split(" ")[0]+' Generated every day',

                  yaxis= dict(title=columns[0].split(" ")[0]+' Generated (in MU)'),

                  xaxis= dict(title='Date')

                 )

    fig = dict(data = data, layout = layout)

    iplot(fig)
dict_features = {

    "Thermal Generation Estimated (in MU)": "sum",

    "Thermal Generation Actual (in MU)": "sum",

   

}

plot_graph(dict_features)

dict_features = {

    "Nuclear Generation Estimated (in MU)": "sum",

    "Nuclear Generation Actual (in MU)": "sum",

}

plot_graph(dict_features)

dict_features = {

     "Hydro Generation Estimated (in MU)": "sum",

    "Hydro Generation Actual (in MU)": "sum"

}

plot_graph(dict_features)
df.columns
df=df.fillna(0)
df["Total Generation Actual (in MU)"]=df["Thermal Generation Actual (in MU)"]+df["Nuclear Generation Actual (in MU)"]+df["Hydro Generation Actual (in MU)"]
df_group=df.groupby(["Region"]).agg({"Total Generation Actual (in MU)":"sum"})

df_group
tracepie=go.Pie(values=df_group["Total Generation Actual (in MU)"].values,labels=df_group.index,hole=0.3)#,marker=dict(colors=["#7579e7","#ff414d","#a3d8f4","#b9fffc","#e8ffff"]))

layout=dict(title="Total Energy Generated distribution according to Region(in MU)")

fig=dict(data=[tracepie],layout=layout)

iplot(fig)
df_group=df.groupby(["Region"]).agg({"Thermal Generation Estimated (in MU)":"sum",

                                "Thermal Generation Actual (in MU)":"sum",

                                 "Nuclear Generation Estimated (in MU)":"sum",

                                "Nuclear Generation Actual (in MU)":"sum",

                               "Hydro Generation Estimated (in MU)":"sum",

                                "Hydro Generation Actual (in MU)":"sum"})



df_group
def plot_est_act_cmp(num):   

    t1=go.Bar(x=list(map(lambda x:' '.join(x.split(" ")[0:2]),df_group.columns[0::2]))

              ,y=df_group.iloc[num].values[0::2]

              ,name="Estimated")

    t2=go.Bar(x=list(map(lambda x:' '.join(x.split(" ")[0:2]),df_group.columns[0::2]))

              ,y=df_group.iloc[num].values[1::2]

               ,name="Actual")

    layout=dict(title="Comparing Estimated and Actuals of "+df_group.index[num])

               #,xlable=dict(title="Energy"))

    fig=dict(data=[t1,t2],layout=layout)

    iplot(fig)
for i in range(5):

    plot_est_act_cmp(i)
state_df=pd.read_csv("/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv")

state_df.head()
data = go.Bar(

       y= state_df['National Share (%)'],

       x= state_df['State / Union territory (UT)'],

       )

layout=dict(title="State wise representation of Nation wise Power Distribution")

           #,xlable=dict(title="State")

           #,ylable=dict(title="National Share (%)"))

fig=dict(data=[data],layout=layout)

iplot(fig)