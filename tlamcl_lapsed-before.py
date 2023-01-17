# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/query2.csv')
data2 = data[data['days_since_last_transact'] < 2000]
hist1 = data2['days_since_last_transact'].hist(bins = 20)
hist1.set_xlabel('Days Since Last Transaction')
hist1.set_ylabel('#')
data2['bins'] = pd.cut(data2['days_since_last_transact'], 20)
data2.groupby('bins').count()['amount']
data2.groupby('bins').count()['amount']
