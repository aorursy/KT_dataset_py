# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import sys

import re

df = pd.read_csv("../input/Traffic_Violations.csv")

df.replace(['Yes', 'No'], [1, 0])

df['Violation'] = pd.Series(1, index=df.index)



df['Date Of Stop'] = pd.to_datetime(df['Date Of Stop'], dayfirst=True)

df['year'], df['month'] = df['Date Of Stop'].dt.year, df['Date Of Stop'].dt.month



import matplotlib.pyplot as plt



df2 = df[['Violation','year']]



count = df2.groupby('year')

totalsum = count.aggregate(np.sum).unstack()

totalsum.plot(kind = 'bar', title = 'violation number per year')

plt.ylabel('count')

plt.show()
df3 = df[['Violation','Gender','year']]



count2 = df3.groupby(['year','Gender'])

totalsum = count2['Violation'].aggregate(np.sum).unstack()

totalsum.plot(kind = 'bar', title = 'violation number per year per gender')

plt.ylabel('count')

plt.show()
#Race-wise traffic violations

df4 = df[['Violation','Race','year']]



count3 = df4.groupby(['year','Race'])

totalsum = count3['Violation'].aggregate(np.sum).unstack()

totalsum.plot(kind = 'bar', title = 'violation number per year per race')

plt.ylabel('count')

plt.show()
df21 = df[['Violation','Make']]



count = df21.groupby('Make').filter(lambda x: len(x) > 5000).groupby('Make')

totalsum = count.aggregate(np.sum).unstack()

totalsum.plot(kind = 'bar', title = 'violation number per Make')

plt.ylabel('count')

plt.show()
#df5 = df[['Violation','Date Of Stop']]



#count4 = df5.groupby('Date Of Stop')

#totalsum = count4.aggregate(np.sum).unstack()

#totalsum.plot(kind = 'bar', title = 'This is a title')

#plt.ylabel('count')

#plt.show()



df5 = df

mask = (df5['Date Of Stop'] >= '2015-1-1') & (df5['Date Of Stop'] < '2016-1-1') & (df5['Fatal'] != 'No')

df5 = df5.loc[mask]



df5 = df5[['Violation','month']]

count4 = df5.groupby('month')

totalsum = count4.aggregate(np.sum).unstack()

totalsum.plot(kind = 'bar', title = 'Fatal accidents in 2015 per month')

plt.ylabel('count')

plt.show()