# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")
df.columns
df.dtypes
df['banking_crisis'].value_counts()
df.shape
df2=df.groupby(['country']).banking_crisis.value_counts().reset_index(name='counts')
df2['percentage']=np.nan

for i in range(len(df2['counts'])):

    if i % 2==0:

        val=df2['counts'][i]+df2['counts'][i+1]

    df2['percentage'][i]=(df2['counts'][i]/val)*100
df2[df2['banking_crisis']=='no_crisis'].sort_values(by='percentage',ascending=False)
df2[df2['banking_crisis']=='crisis'].sort_values(by='percentage',ascending=False)
df3=df.groupby(['year']).banking_crisis.value_counts().reset_index(name='counts')
df3.sort_values(by='counts',ascending=False)
countries=df['country'].unique()




fig = go.Figure()





colors = ['rgb(70,150,200)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(30,30,30)','rgb(30,350,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)','rgb(67,67,67)', 'rgb(30,300,115)', 'rgb(49,130,189)', 'rgb(189,189,189)','rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']



c=0

while c < len(countries):

    df2=df[df['country']==countries[c]]

    fig.add_trace(go.Scatter(x=df2.year,y=df2.exch_usd,name="USD",mode='lines+markers',visible=False,line=dict(color=colors[c])))

    c=c+1







fig.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            active=0,

            buttons=list([

                dict(label="None",

                     method="update",

                     args=[{"visible": [False, False, False, False,False, False, False, False,False, False, False, False,False]},

                           {"title": 'NONE',

                            "annotations": []}]),

                dict(label="Algeria",

                     method="update",

                     args=[{"visible": [True, False, False, False,False, False, False, False,False, False, False, False,False]},

                           {"title": 'Algeria',

                            "annotations": []}]),

                dict(label="Angola",

                     method="update",

                     args=[{"visible": [False,True, False, False,False, False, False, False,False, False, False, False,False]},

                           {"title": 'Angola',

                            "annotations": []}]),

                dict(label="Central African Republic",

                     method="update",

                     args=[{"visible": [False, False, True, False,False, False, False, False,False, False, False, False,False]},

                           {"title": 'Central African Republic',

                            "annotations": [] }]),

                dict(label="Ivory Coast",

                     method="update",

                     args=[{"visible": [False,False, False, True,False, False, False,False, False, False, False,False,False]},

                           {"title": 'Ivory Coast',

                            "annotations": []}]),

                

                dict(label="Egypt",

                     method="update",

                     args=[{"visible": [False,False, False, False,True, False, False,False, False, False, False,False,False]},

                           {"title": 'Egypt' ,

                            "annotations": []}]),

                dict(label="Kenya",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, True, False,False, False, False, False,False,False]},

                           {"title": 'Kenya',

                            "annotations": []}]),

                dict(label="Mauritius",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, True,False, False, False, False,False,False]},

                           {"title": 'Mauritius',

                            "annotations": []}]),

                dict(label="Morocco",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, False,True, False, False, False,False,False]},

                           {"title": 'Morocco',

                            "annotations": []}]),

                dict(label="Nigeria",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, False,False, True, False, False,False,False]},

                           {"title": 'Nigeria',

                            "annotations": []}]),

                dict(label="South Africa",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, False,False, False, True, False,False,False]},

                           {"title": 'South Africa',

                            "annotations": []}]),

                dict(label="Tunisia",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, False,False, False, False, True,False,False]},

                           {"title": 'Tunisia',

                            "annotations": []}]),

                dict(label="Zambia",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, False,False, False, False, False,True,False]},

                           {"title": 'Zambia',

                            "annotations": []

                            }]),

                dict(label="Zimbabwe",

                     method="update",

                     args=[{"visible": [False,False, False, False,False, False, False,False, False, False, False,False,True]},

                           {"title":  'Zimbabwe',

                            "annotations": []}]),

                dict(label="ALL",

                     method="update",

                     args=[{"visible": [True,True, True, True,True, True, True,True, True, True, True,True,True]},

                           {"title":  'ALL',

                            "annotations": []}]),

                

            ]),

        )

    ])



# Set title

fig.update_layout(title_text="AFRICA")



fig.show()