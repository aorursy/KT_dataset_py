# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/query_result.csv')
data2 = data[data['days_since_last_transact'] < 2000]
data2
hist1 = data2['days_since_last_transact'].hist(bins = 20)
hist1.set_xlabel('Days Since Last Transaction')
hist1.set_ylabel('#')
# ticks = np.arange(0, 2000, step = 100)
# hist1.set_xticks(ticks)



# hist1.set_xticks(ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], labels = [])
# hist1.xticks()

data2['bins'] = pd.cut(data2['days_since_last_transact'], 20)
data2.groupby('bins').count()['amount']
data2.groupby('bins').count()['amount']

