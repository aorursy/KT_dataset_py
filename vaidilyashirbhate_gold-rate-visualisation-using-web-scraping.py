import bs4

import requests

import pandas as pd
import numpy as np

from plotly.offline import iplot

import plotly.graph_objects as go

import plotly.io as pio

from plotly.subplots import make_subplots

#pio.templates.default="plotly_dark"
req=requests.get("https://www.goodreturns.in/gold-rates/")
soup=bs4.BeautifulSoup(req.text,"html.parser")
soup.select("strong")[0].text
soup.select("strong",{"id":"el"})[1].text+" is : "+soup.select("strong",{"id":"el"})[0].text
strarr=soup.select("table")[2].text
arr=strarr.split("\n")
for i in range(arr.count("")):

    arr.remove("")
arr.append("")
arr_city=arr[3:-1:3]

arr_22_gold=arr[4:-1:3]

arr_24_gold=arr[5:-1:3]
df=pd.DataFrame({"City":arr_city

                 ,"22 Carat Gold Today":arr_22_gold

                 ,"24 Carat Gold Today":arr_24_gold})
df["22 Carat Gold Today"]=df["22 Carat Gold Today"].apply(lambda x:int(''.join(x[3:].split(','))))

df["24 Carat Gold Today"]=df["24 Carat Gold Today"].apply(lambda x:int(''.join(x[3:].split(','))))
df
Data22=go.Bar(y=df["22 Carat Gold Today"],x=df["City"],name='22 Carat Gold',opacity=0.8,marker = dict(color = '#ffa600'))

Data24=go.Bar(y=df["24 Carat Gold Today"],x=df["City"],name='24 Carat Gold',opacity=0.8,marker = dict(color = '#fdd576'))

layout=dict(title="Indian Major Cities Gold Rates Today",yaxis=dict(title="Price"),xaxis=dict(title="City"))

Fig=dict(data=[Data22,Data24],layout=layout)

iplot(Fig)
str_last_10_day=soup.select("table")[3].text
arr_date=str_last_10_day.split("\n")
df_gold_last_10days=pd.DataFrame({"Date":arr_date[8:-1:13]

                                  ,"22 Carat":arr_date[10:-1:13]

                                  ,"22 carat inc":arr_date[12:-1:13]

                                  ,"24 Carat":arr_date[15:-1:13]

                                   ,"24 Carat inc":arr_date[17:-1:13]})
df_gold_last_10days.index=df_gold_last_10days.Date
df_gold_last_10days.drop(["Date"],axis=1,inplace=True)
def conv(x):

    return x.split("(")[1][1:-2]

df_gold_last_10days["22 carat inc"]=df_gold_last_10days["22 carat inc"].apply(conv)

df_gold_last_10days["24 Carat inc"]=df_gold_last_10days["24 Carat inc"].apply(conv)
df_gold_last_10days["22 carat inc"]=df_gold_last_10days["22 carat inc"].astype('int64')

df_gold_last_10days["24 Carat inc"]=df_gold_last_10days["24 Carat inc"].astype('int64')

df_gold_last_10days["22 Carat"]=df_gold_last_10days["22 Carat"].apply(lambda x:int(''.join(x[2:].split(','))))

df_gold_last_10days["24 Carat"]=df_gold_last_10days["24 Carat"].apply(lambda x:int(''.join(x[2:].split(','))))
df_gold_last_10days
Data22=go.Scatter(y=df_gold_last_10days["22 carat inc"],x=df_gold_last_10days.index,name="Inc in 22 Carat gold rate")

Data24=go.Scatter(y=df_gold_last_10days["24 Carat inc"],x=df_gold_last_10days.index,name="Inc in 24 Carat gold rate")

Data22p=go.Scatter(y=df_gold_last_10days["22 Carat"],x=df_gold_last_10days.index,name="22 Carat gold price")

Data24p=go.Scatter(y=df_gold_last_10days["24 Carat"],x=df_gold_last_10days.index,name="24 Carat gold price")
fig=make_subplots(rows=2, cols=2,subplot_titles=("Inc in 22 Carat gold rate","22 Carat gold price","Inc in 24 Carat gold rate","24 Carat gold price"))

fig.add_trace(Data22,row=1, col=1)

fig.add_trace(Data22p,row=1, col=2)

fig.add_trace(Data24,row=2, col=1)

fig.add_trace(Data24p,row=2, col=2)

fig.update_layout(title_text="Gold Rate in India for Last 10 Days (10 g)", height=700)

fig.show()