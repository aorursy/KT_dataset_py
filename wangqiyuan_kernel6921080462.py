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
df =  pd.read_csv("/kaggle/input/tesla-stock-data-from-2010-to-2020/TSLA.csv")
df.head()
df.Close

plt.xlabel("days")         # ось абсцисс

plt.ylabel("price")    # ось ординат

plt.grid()              # включение отображение сетки

plt.plot(df.Close)
df.Date
df.High
plt.xlabel("days")         # ось абсцисс

plt.ylabel("price")    # ось ординат

plt.grid()              # включение отображение сетки

plt.plot(df.Close)

plt.plot(df.High)

plt.grid()  


df.Open

print(len(df.Close))

las100_days = df.High[2316:2415]

plt.plot(las100_days)