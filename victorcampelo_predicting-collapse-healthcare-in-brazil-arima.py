# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd                           # Manipulação de dados
import glob                                   # Importação de dados
import re                                     # Expressões regulares
import itertools                              # Ferramenta iteração
import numpy as np                            # Computação científica
import statsmodels.api as sm                  # Modelagem estatística
import seaborn as sns                         # Visualização de dados
import squarify                               # Visualização de treemaps

import matplotlib.pyplot as plt               # Visualização de dados
import matplotlib
plt.style.use('fivethirtyeight') 

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls                    # Ferramentas do plotly
import ipywidgets as widgets

import folium                                 # Visualização de mapas
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster

import warnings                               # Ignorar warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-pastel')
plt.rcParams['figure.figsize'] = (10,7)
covid = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
covid.head()
covid = covid.rename(columns={'Country/Region': 'location'})
covid.groupby('location')['Confirmed'].sum()
br = covid[covid['location'] == 'Brazil']
covid[covid['location'] == 'Brazil'][['Date','Confirmed']].plot.bar(figsize=(20,8), x='Date')
br = br.drop(['Lat', 'Long'], 1)
covid['Date'] = pd.to_datetime(covid['Date'], format='%m/%d/%y')
covid['Date']
sus = pd.read_csv('/kaggle/input/sus-brazil/sus_1000_hab.csv') # available beds in each city from Brazil 2010
sus = sus.drop(columns=['FID', 'gid', 'Censo','legenda', 'Descrição', 'classe'])
sus.head()
sus.describe()
sus.shape
sus['total_leitos'] = (sus['Pop_est_2009']/1000)*sus['razao_leitos_sus_1000_hab']
sus['total_leitos']
sus.head()
sus['total_leitos'].sum()
sus.groupby('UF')['razao_leitos_sus_1000_hab'].sum().plot.barh()
x = br['Confirmed']
x2 =  br['Deaths']
y = br['Date']
plt.figure(figsize=(25, 10))
plt.plot(y,x, label='Confirmed', color='red');
plt.plot(y,x2, label='Deaths', color='black');
plt.scatter(y, x, color='red')
plt.scatter(y, x2, color='blue')
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.grid()
plt.show()
br2 = br[['Date', 'Confirmed']]
br2.index = pd.to_datetime(br.Date, format='%m/%d/%y')
br2.drop(columns='Date', inplace=True)
br2 = br2[br2['Confirmed'] > 0]
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(br2.Confirmed.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# TRANSFORMANDO SÉRIE TEMPORAL
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(br2, model='multiplicative')
g = result.plot()
g.set_figwidth(20)
g.set_figheight(10)
# Import Libs

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# PLOT ACF Correlation

g = plot_acf(br2, title="ACF Correlation")
g.set_figheight(8)
g.set_figwidth(14)
br_diff = br2.diff()

g = plot_acf(br_diff, title="ACF Correlation (d = 1)")
g.set_figheight(8)
g.set_figwidth(14)
# PLOT CORRELAÇÃO ACF

br_diff2 = br2.diff().diff().diff()

g = plot_acf(br_diff2, title="ACF Correlation (d = 2)")
g.set_figheight(8)
g.set_figwidth(14)
result = adfuller(br_diff.Confirmed.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# DEFINE COMBINAÇÕES ENTRE p DE 0 A 3, d = 2, e q = 13
pdq = [(p, 2, 1) for p in range(0, 4)]

# DEFINE P e Q ENTRE 0 e 3
P = Q = range(0, 4)

# DEFINE COMBINAÇÕES ENTRE P, D e Q
seasonal_pdq = [(x[0], 1, x[1], 12) for x in list(itertools.product(P, Q))]

# CALCULANDO O MELHOR PARÂMETRO PARA O MODELO

scores = {}

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(br2,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            scores[param, param_seasonal] = results.aic
            
        except:
            continue
            
print("Melhoers parâmetros: ", min(scores, key=scores.get)," AUC: ", min(scores.values()))

# CRIANDO MODELO COM OS MELHOERS PARÂMETROS
# Make the model with the best paraments

mod = sm.tsa.statespace.SARIMAX(br2,
                                order=(0, 2, 1),
                                seasonal_order=(0, 1, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
# VALIDAÇÃO DO MODELO: PREVENDO OCORRÊNCIAS A PARTIR DE 2017

pred = results.get_prediction(start=pd.to_datetime('2020-02-29'), dynamic=False)
pred_ci = pred.conf_int()

ax = br2.plot(figsize=(14, 8))
pred.predicted_mean.plot(ax=ax, label='predict', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_title("Train to predite growing curve", fontsize=18)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Cases')
plt.legend()

plt.show()

# CALCULANDO MSE

y_forecasted = pred.predicted_mean
y_truth = br2["2020-02-29":].squeeze()

mse = ((y_forecasted - y_truth) ** 2).mean()

print('Mean Squared Error {}'.format(round(mse)))
# PREVENDO OCORRÊNCIAS PARA OS PRÓXIMOS TRÊS ANOS

pred_uc = results.get_forecast(steps=300)
pred_ci = pred_uc.conf_int()

ax = br2.plot(figsize=(14, 8))
pred_uc.predicted_mean.plot(ax=ax, label='predict')
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_title("Prediction of increase in new coronavirus confirming cases in 300 days", fontsize=18)
ax.set_xlabel('Date')
ax.set_ylabel('Number of news cases')

plt.legend()
plt.show()
xp = pred_uc.predicted_mean.index.tolist()
ylimit = []
for dt in br['Date']:
    ylimit.append(33000)
for dt in xp:
    ylimit.append(33000)
yp = pred_uc.predicted_mean
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)

trace = [go.Scatter(x = br['Date'], y = br['Confirmed'], name='Total of Cases')]
trace += [go.Scatter(x = xp,y = ylimit, name='Total of beds')]
trace += [go.Scatter(x = xp,y = yp, name='mean rise case')]
trace += [go.Scatter(x = pred_ci.index,y = pred_ci.iloc[:, 0], fill='tonexty', name='min')]
trace += [go.Scatter(x = pred_ci.index,y = pred_ci.iloc[:, 1], fill='tonexty', name='max')]

layout = dict(
    title='Total of coronavirus cases - Brazil',
    yaxis=dict(
    title='new cases'
    ),
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=12,
                     label='1yr',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)
fig = dict(data=trace, layout=layout)
py.iplot(fig)
