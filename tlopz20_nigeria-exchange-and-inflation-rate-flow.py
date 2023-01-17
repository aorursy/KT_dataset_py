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
data=pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
#Working with country, exchange rate and year

#CEY = country, exch_usd and Y  year

print(data.columns)
#Selecting the column for usage 

#CEY = Country, exch_usd and Year

CEY = data[['country', 'exch_usd', 'year', 'inflation_annual_cpi']]

CEY.head()
# Now working with just data from Nigeria starting from 1999 to 2014

NG = CEY.loc[CEY['country'] =='Nigeria']

NG90 = NG.tail(16)

NG90
# To check the Rate flow from 1999 to 2014 on a Bar chart



import matplotlib.pyplot as plt

color_list = ['red','black']

ax = NG90.plot('year','exch_usd', kind='bar',figsize=(20,8),width=0.8,color = color_list,edgecolor = None)

plt.title("exchange rate flow from 1999 to 2014", fontsize=16)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#To check the Inflation flow from 1999 to 2014



color_list = ['blue','black']

ax = NG90.plot('year','inflation_annual_cpi', kind='bar',figsize=(20,8),width=0.8,color = color_list,edgecolor = None)

plt.title("year per Inflation % from 1991 to 2016", fontsize=16)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#From the analysis 2007 seems to be the best yaer in the 2000's with reduction in exchchange rate and a reduction in Inflation.