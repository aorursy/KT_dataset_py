# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/delhi-house-price-prediction/MagicBricks.csv')

df
df.isnull().any(axis=0)
df = df[df.loc[:]!=0].dropna()

df.isnull().any(axis=0)
bhk_hist=np.random.normal(2.713500,1.182314,12000)

plt.hist(bhk_hist,20)

plt.xlabel('BHK')

plt.ylabel('frequency')

plt.show()
price_hist=np.random.normal(1.135153e+05,1.654280e+05,12000)

plt.hist(price_hist)

plt.xlabel('price')

plt.ylabel('frequency')

plt.show()
price_hist=np.random.normal(2.384167,1.413123,12000)

plt.hist(price_hist)

plt.xlabel('Bathroom')

plt.ylabel('frequency')

plt.show()
price_hist=np.random.normal(2101.322333,1624.840792,12000)

plt.hist(price_hist)

plt.xlabel('Sqft')

plt.ylabel('frequency')

plt.show()
Location_With_Price=df.groupby('Locality')['Price'].mean()

q=np.arange(0,880)

import plotly.graph_objects as go



fig = go.Figure(go.Bar(

            x=Location_With_Price,

            y=Location_With_Price.index,

            orientation='h'))



fig.show()