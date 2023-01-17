# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import torch

from torch import nn



import matplotlib.pyplot as plt

import pandas as pd

df1 = pd.read_csv(r'./../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv', usecols=[0,1])



table_val = df1.values
plt.plot(table_val[:, 1], label='oil price from 1986')



plt.xlabel('1986~2020',fontsize=14)

plt.ylabel('price',fontsize=14)
df1 = pd.read_csv(r'./../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv', usecols=[842, 844, 849])



data = df1.values
plt.plot(data[:, 0], label='world new case')



plt.xlabel('19/12/31~',fontsize=14)

plt.ylabel('price',fontsize=14)
plt.plot(data[:, 1], label='world new death')



plt.xlabel('19/12/31~',fontsize=14)

plt.ylabel('price',fontsize=14)