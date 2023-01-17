import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARIMA

from pandas.tseries.offsets import DateOffset



import matplotlib.pyplot as plt

%matplotlib inline



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',

                              'Parrot', 'Parrot'],

                   'Max Speed': [380., 370., 24., 26.]})

df
df.groupby(['Animal']).mean()
arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],

          ['Captive', 'Wild', 'Captive', 'Wild']]

index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))

df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},

                  index=index)

df
df.groupby(level=0).mean()
df.groupby(level="Type").mean()
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',

   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],

   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],

   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],

   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

df1 = pd.DataFrame(ipl_data)



df1
df1.groupby('Team').groups
df1.groupby(['Team','Year']).groups
df3 = pd.DataFrame({

    'user_id':[1,2,1,3,3,],

    'content_id':[1,1,2,2,2],

    'tag':['cool','nice','clever','clever','not-bad']

})



df3
df3.groupby("content_id")['tag'].apply(lambda tags: ','.join(tags))
df4 = pd.DataFrame({

    'value':[20.45,22.89,32.12,111.22,33.22,100.00,99.99],

    'product':['table','chair','chair','mobile phone','table','mobile phone','table']

})



df4
df4.groupby('product')['value'].sum()
df4.groupby('product')['value'].sum().to_frame().reset_index().sort_values(by='value')
df5 = pd.DataFrame({

    'value':[20.45,22.89,32.12,111.22,33.22,100.00,99.99],

    'product':['table','chair','chair','mobile phone','table','mobile phone','table']

})



df5
plt.clf()

df5.groupby('product').size().plot(kind='bar')

plt.show()
plt.clf()

df5.groupby('product').sum().plot(kind='bar')

plt.show()