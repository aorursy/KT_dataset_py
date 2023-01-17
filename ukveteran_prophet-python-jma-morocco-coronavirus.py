import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation Libraries

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



# Date & Time

from datetime import date, datetime, timedelta



warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline



# plt.style.use('ggplot')

plt.style.use('seaborn-white')

font = {

    'family' : 'normal',

    'weight' : 'bold',

    'size'   : 13

}

plt.rc('font', **font)
df = pd.read_csv('../input/moroccocoronavirus/corona_morocco.csv')

df.head()
df.fillna(df.mean(), inplace=True)
df.head()
from fbprophet import Prophet
df1=df.rename(columns={"Date": "ds", "Confirmed": "y"})

df1
df2=df1.drop(["Deaths", "Recovered","Beni Mellal-Khenifra","Casablanca-Settat","Draa-Tafilalet","Dakhla-Oued Ed-Dahab","Fes-Meknes","Guelmim-Oued Noun","Laayoune-Sakia El Hamra","Marrakesh-Safi","Oriental","Rabat-Sale-Kenitra","Souss-Massa","Tanger-Tetouan-Al Hoceima"], axis = 1)

df2
m = Prophet()

m.fit(df2)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)