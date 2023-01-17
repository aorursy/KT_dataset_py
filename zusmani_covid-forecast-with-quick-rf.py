# 786

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objs as go



import plotly as py

from plotly import tools

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dt = pd.read_csv("../input/pakistan-corona-virus-citywise-data/PK COVID-19-30apr.csv", encoding = "ISO-8859-1", parse_dates=["Date"])

print("Data Dimensions are: ", dt.shape)

print(dt.head)
dt.info()
dt['Travel_history'].unique

dt['Travel_history'].fillna('Unknown',  inplace=True)
dt = dt.sort_values('Date')

dt['Deaths']=dt['Deaths'].astype(int)

dt['Cases']=dt['Cases'].astype(int)

dt['Recovered']=dt['Recovered'].astype(int)



dt.loc[dt.Province == "khyber Pakhtunkhwa", "Province"] = "Khyber Pakhtunkhwa"

dt.loc[dt.Travel_history == "Tableegi Jamaat", "Travel_history"] = "Tableeghi Jamaat"
pdc = dt.groupby('Date')['Cases'].sum().reset_index()

pdd = dt.groupby('Date')['Deaths'].sum().reset_index()#.drop('Date', axis=1)

pdr = dt.groupby('Date')['Recovered'].sum().reset_index()#.reset_index()#.drop('Date', axis=1)



p = pd.DataFrame(pdc) 

p['Deaths'] = pdd['Deaths']

p['Recovered'] = pdr['Recovered']



#Cumulative Sum

p['Cum_Cases'] = p['Cases'].cumsum() 

p['Cum_Deaths'] = p['Deaths'].cumsum()

p['Cum_Recovered'] = p['Recovered'].cumsum()



del pdc, pdd, pdr 

p.head()

p['Dateofmonth'] = p['Date'].dt.day

p['Month'] = p['Date'].dt.month

p['Week'] = p['Date'].dt.week

p['Dayofweek'] = p['Date'].dt.dayofweek # 0 = monday.

p['Weekdayflg'] = (p['Dayofweek'] // 5 != 1).astype(float)

p['Month'] = p['Date'].dt.month

p['Quarter'] = p['Date'].dt.quarter

p['Dayofyear'] = p['Date'].dt.dayofyear

p.head(10)
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cases'],

                    mode='lines+markers',

                    name='Cases'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Recovered'],

                    mode='lines+markers',

                    name='Recoveries'))



fig.show()
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cum_Cases'],

                    mode='lines+markers',

                    name='Cases'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cum_Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cum_Recovered'],

                    mode='lines+markers',

                    name='Recoveries'))



fig.show()
px.scatter(p, x= 'Date', y = 'Cases', trendline = "ols")
from sklearn.ensemble import RandomForestClassifier

#model = RandomForestClassifier(n_estimators=200)

model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                      max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=250, random_state=None, n_jobs=1, verbose=0)
input_col = [#'Date',

# 'Cases',

# 'Deaths',

# 'Recovered',

# 'Cum_Cases',

# 'Cum_Deaths',

# 'Cum_Recovered',

 'Dateofmonth',

 'Month',

 'Week',

 'Dayofweek',

 'Weekdayflg',

 'Quarter',

 'Dayofyear']



output_cols = ['Cases', 'Deaths', 'Cum_Cases', 'Cum_Deaths'] 
X = p[input_col]

Y1 = p[output_cols[0]]
# Date Range for Prediction

pred_dates = np.arange('2019-05', '2019-06', dtype='datetime64[D]')

pred_range = pred_dates[0:6]

pred = pd.DataFrame(pred_range, columns=['Date'])

pred['Dateofmonth'] = pred['Date'].dt.day

pred['Month'] = pred['Date'].dt.month

pred['Week'] = pred['Date'].dt.week

pred['Dayofweek'] = pred['Date'].dt.dayofweek # 0 = monday.

pred['Weekdayflg'] = (pred['Dayofweek'] // 5 != 1).astype(float)

pred['Month'] = pred['Date'].dt.month

pred['Quarter'] = pred['Date'].dt.quarter

pred['Dayofyear'] = pred['Date'].dt.dayofyear

#pred.info()
model.fit(X,Y1)
X_test = pred[input_col]

prd = model.predict(X_test)
pred['Predicted_Cases'] = prd
pred