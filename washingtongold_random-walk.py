import random

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
for i in range(3):

    n=1000

    x = np.linspace(1,100,n)

    y = [50]

    for i in range(n-1):

        y.append(y[-1]+random.randint(-3,3))

    plt.figure(figsize=(15,6))

    sns.lineplot(x,y)

    plt.show()
from pandas_datareader import data as web

import datetime as dt

df = web.DataReader('IBM', 'yahoo',dt.datetime(2000,1,22),dt.datetime(2020,3,23))

plt.figure(figsize=(15,6))

data = np.array(df['Close'][:1000])

sns.lineplot(range(len(data)),data)
df['Close'][:1000]