# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

    



# Any results you write to the current directory are saved as output.

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/restaurant-week-2018/restaurant_week_2018_final.csv')
data.head()
data.shape[0]
data.iloc[200:211]
data.loc[data.average_review.idxmax()][0]
data.loc[data.average_review.idxmin()][0]
data.query('review_count > 1000')
data.query('review_count > 1000').mean()
data.query('review_count < 1000').mean()
data.groupby("restaurant_main_type").mean()
data.groupby("restaurant_main_type").size().sort_values(ascending=False)
data.groupby("restaurant_main_type").mean().sort_values('value_review', ascending = False)
plotData = data.groupby("restaurant_main_type").mean().sort_values('value_review', ascending = False).value_review

plotData.plot.barh()
plotData2 = data.groupby("restaurant_main_type").mean().sort_values('value_review', ascending = False).average_review

plotData2.plot.barh()