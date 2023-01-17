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

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



ts = pd.read_csv('../input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt



ts = pd.read_csv('../input/tesla-stock-data-from-2010-to-2020/TSLA.csv')



import warnings

warnings.filterwarnings('ignore')



#SEE IF NaN VALUES EXISTS

print(ts.isna().any())



#THIS ONE ADDS UP ALL THE NULL VALUES

print(ts.isnull().sum())



#FILLS THE NaN WITH ZERO

ts.fillna(0)



print(ts.describe(include="all"))

print(ts.shape)

print(ts.info())

print(ts.head())

print(ts.dtypes)



#PRINTS CORRELATION

print(ts.corr())





#TO SEE IF THERE A SPIKE OF VARIANCE OF THE VOLUME BEFORE THE RISE

ts['Volume'].rolling(50).var().plot()

plt.show()













#DATE BREAKDOWN



#FIRST CONVERT THE DATE COLUMN TO A DATETIME OBJECT

ts['Date'] = pd.to_datetime(ts['Date'])

print(ts.Date.describe())



#BAR GRAPH OF VOLUME BY DATES

vol_count = ts.Volume.count()

ax = ts.plot.bar(x='Date', y='Volume', rot=0)



#SCATTERPLOT OF ADJUSTED CLOSE BY DATE

ax = sns.scatterplot(x="Date", y="Adj Close", data=ts)