# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



eqdata = pd.read_csv("../input/italy_earthquakes_from_2016-08-24_to_2016-11-30.csv").set_index('Time')
eqdata.index = pd.to_datetime(eqdata.index)

eqdata.index.dtype
mag = eqdata.Magnitude

(n, bins, patches) = plt.hist(mag, bins=4)

print(n)

print(bins)
eqdata.Magnitude.resample('2D').mean().plot()

plt.title("Time Series: Average Magnitude")

plt.ylabel("Magnitude")