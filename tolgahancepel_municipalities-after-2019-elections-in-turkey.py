import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import plotly as plotly

import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING
df = pd.read_excel("../input/gsyh.xls", header=None)

df.head(50)
cities = np.c_[df.iloc[:,1]]
clean = pd.DataFrame(data = cities)

clean = clean.dropna()

clean = clean.reset_index(drop=True)

clean.columns = ['cities']

clean.head()
clean.info()
pd.options.display.float_format = '{:}'.format

gdp = df[df[2] == 2017.0]

gdp = gdp.iloc[:,8]

gdp = gdp.reset_index(drop=True)

gdp.head()
clean['gdp'] = gdp
clean = clean.sort_values(by='gdp', ascending=False).reset_index(drop=True)

clean.head(10)
clean['rate'] = clean['gdp'] / clean.iloc[0,1] * 100

clean['rate'] = clean['rate'].astype(float).round(3)
# This data created manually because of is not published yet from TURKSTAT.



pparty = ["", #0

          "CHP", # 1

          "CHP", # 2

          "CHP", # 3

          "AKP", # 4

          "AKP", # 5

          "CHP", # 6

          "AKP", # 7

          "CHP", # 8

          "AKP", # 9

          "CHP", # 10

          "MHP", # 11

          "CHP", # 12

          "AKP", # 13

          "CHP", # 14

          "AKP", # 15

          "AKP", # 16

          "AKP", # 17

          "AKP", # 18

          "CHP", # 19

          "CHP", # 20

          "HDP", # 21

          "CHP", # 22

          "AKP", # 23

          "AKP", # 24

          "AKP", # 25

          "CHP", # 26

          "AKP", # 27

          "AKP", # 28

          "AKP", # 29

          "AKP", # 30

          "AKP", # 31

          "AKP", # 32

          "MHP", # 33

          "HDP", # 34

          "HDP", # 35

          "AKP", # 36

          "INDPT", # 37

          "AKP", # 38

          "AKP", # 39

          "CHP", # 40

          "AKP", # 41

          "MHP", # 42

          "AKP", # 43

          "CHP", # 44

          "AKP", # 45

          "AKP", # 46

          "AKP", # 47

          "AKP", # 48

          "MHP", # 49

          "CHP", # 50

          "HDP", # 51

          "AKP", # 52

          "AKP", # 53

          "CHP", # 54

          "AKP", # 55

          "MHP", # 56

          "AKP", # 57

          "AKP", # 58

          "MHP", # 59

          "CHP", # 60

          "AKP", # 61

          "MHP", # 62

          "MHP", # 63

          "AKP", # 64

          "AKP", # 65

          "CHP", # 66

          "HDP", # 67

          "CHP", # 68

          "HDP", # 69

          "AKP", # 70

          "AKP", # 71

          "MHP", # 72

          "CHP", # 73

          "HDP", # 74

          "MHP", # 75

          "HDP", # 76

          "AKP", # 77

          "AKP", # 78

          "TKP", # 79

          "CHP", # 80

          "MHP", # 81         

         ];

clean['pparty'] = pparty

clean.head(10)
test = clean[1:]

test.head()
colors=np.array(['#000000']*20)

colors[test['pparty'][:20] == 'CHP'] = '#ac0400'

colors[test['pparty'][:20] == 'AKP'] = '#f78302'

colors[test['pparty'][:20] == 'MHP'] = '#1c4ba0'

# No need to continue assign colors because there are only 2 political parties in first 20 row.



trace1=go.Bar(

            x=test['cities'][:20],

            y=test['gdp'][:20],

            text="TRY",

            marker=dict(

                color=colors,

            ))

data = [trace1]



layout = go.Layout(title ='GDP (Gross Domestic Product) in 20 Wealthiest City',

              xaxis = dict(title = 'Cities'),

              yaxis = dict(title = 'GDP (TRY)'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Sum rate of CHP

chp_sum = clean[clean['pparty'] == "CHP"]

chp_sum = sum(chp_sum['rate'])

chp_sum
# Sum rate of AKP

akp_sum = clean[clean['pparty'] == "AKP"]

akp_sum = sum(akp_sum['rate'])

akp_sum
# Sum rate of MHP

mhp_sum = clean[clean['pparty'] == "MHP"]

mhp_sum = sum(mhp_sum['rate'])

mhp_sum
# Sum rate of HDP

hdp_sum = clean[clean['pparty'] == "HDP"]

hdp_sum = sum(hdp_sum['rate'])

hdp_sum
# Sum rate of TKP

tkp_sum = clean[clean['pparty'] == "TKP"]

tkp_sum = sum(tkp_sum['rate'])

tkp_sum
# Sum rate of INDEPENDENT

bgmsz_sum = clean[clean['pparty'] == "INDPT"]

bgmsz_sum = sum(bgmsz_sum['rate'])

bgmsz_sum
labels = ['CHP','AKP','MHP','HDP','INDPT','TKP']

values = [chp_sum, akp_sum, mhp_sum, hdp_sum, bgmsz_sum, tkp_sum]

colors = ["#ac0400", "#f78302", "#1c4ba0", "#6c289f", "#90979e", "#90979e"]

trace = go.Pie(labels=labels, values=values,

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

data = [trace]

layout = go.Layout(title='Political Party Rates by GDP')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
chp_minucs = test[test['pparty'] == 'CHP']

chp_minucs.head()
trace1=go.Bar(

            x=chp_minucs['cities'],

            y=chp_minucs['gdp'],

            text="TRY",

            marker=dict(

                color='#ac0400',

            ))

data = [trace1]



layout = go.Layout(title ='Municipalities of CHP (Cumhuriyet Halk Partisi)',

              xaxis = dict(title = 'Cities'),

              yaxis = dict(title = 'GDP (TRY)'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
akp_minucs = test[test['pparty'] == 'AKP']

akp_minucs.head()
trace2=go.Bar(

            x=akp_minucs['cities'],

            y=akp_minucs['gdp'],

            text="TRY",

            marker=dict(

                color='#f78302',

            ))

data = [trace2]



layout = go.Layout(title ='Municipalities of AKP (Adalet ve Kalkınma Partisi)',

              xaxis = dict(title = 'Cities', tickangle = 45),

              yaxis = dict(title = 'GDP (TRY)'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
mhp_minucs = test[test['pparty'] == 'MHP']

mhp_minucs.head()
trace3=go.Bar(

            x=mhp_minucs['cities'],

            y=mhp_minucs['gdp'],

            text="TRY",

            marker=dict(

                color='#1c4ba0',

            ))

data = [trace3]



layout = go.Layout(title ='Municipalities of MHP (Milliyetçi Hareket Partisi)',

              xaxis = dict(title = 'Cities'),

              yaxis = dict(title = 'GDP (TRY)'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
hdp_minucs = test[test['pparty'] == 'HDP']

hdp_minucs.head()
trace4=go.Bar(

            x=hdp_minucs['cities'],

            y=hdp_minucs['gdp'],

            text="TRY",

            marker=dict(

                color='#6c289f',

            ))

data = [trace4]



layout = go.Layout(title ='Municipalities of HDP (Halkların Demokratik Partisi)',

              xaxis = dict(title = 'Cities'),

              yaxis = dict(title = 'GDP (TRY)'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
tkp_minucs = test[test['pparty'] == 'TKP']

tkp_minucs.head()
indpt_minucs = test[test['pparty'] == 'INDPT']

indpt_minucs.head()