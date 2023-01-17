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
df_reservoir_level = pd.read_csv('/kaggle/input/chennai_reservoir_levels.csv')

df_rainfall = pd.read_csv('/kaggle/input/chennai_reservoir_rainfall.csv')
df_reservoir_level.head()
df_reservoir_level.Date = pd.to_datetime(df_reservoir_level.Date)

df_reservoir_level['Year'] = df_reservoir_level['Date'].dt.year

df_reservoir_level['Month'] = df_reservoir_level['Date'].dt.month

df_reservoir_level['Month_name'] = df_reservoir_level['Date'].dt.month_name()

df_reservoir_level.head()
# Searched the google for the reservoirs maximum capacity

POONDI_max_cap = 3231

CHOLAVARAM_max_cap = 595

REDHILL_max_capS = 3300

CHEMBARAMBAKKAM_max_cap = 3645
df_reservoir_level_grouping = df_reservoir_level.groupby('Year').agg({'POONDI':'mean','CHOLAVARAM':'mean','REDHILLS':'mean','CHEMBARAMBAKKAM':'mean'})

df_reservoir_level_max_cap_ratio = df_reservoir_level_grouping.copy()

df_reservoir_level_max_cap_ratio['POONDI_%_Fill'] = (df_reservoir_level_max_cap_ratio['POONDI']/POONDI_max_cap)*100

df_reservoir_level_max_cap_ratio['CHOLAVARAM_%_Fill'] = (df_reservoir_level_max_cap_ratio['CHOLAVARAM']/CHOLAVARAM_max_cap)*100

df_reservoir_level_max_cap_ratio['REDHILLS_%_Fill'] = (df_reservoir_level_max_cap_ratio['REDHILLS']/REDHILL_max_capS)*100

df_reservoir_level_max_cap_ratio['CHEMBARAMBAKKAM_%_Fill'] = (df_reservoir_level_max_cap_ratio['CHEMBARAMBAKKAM']/CHEMBARAMBAKKAM_max_cap)*100

df_reservoir_level_max_cap_ratio['FINAL_Fill'] = (df_reservoir_level_max_cap_ratio['POONDI_%_Fill']+df_reservoir_level_max_cap_ratio['CHOLAVARAM_%_Fill']+df_reservoir_level_max_cap_ratio['REDHILLS_%_Fill']+df_reservoir_level_max_cap_ratio['CHEMBARAMBAKKAM_%_Fill'])/4

df_reservoir_level_max_cap_ratio.head()
df_reservoir_level_max_cap_ratio = df_reservoir_level_max_cap_ratio.reset_index()
df_rainfall.Date = pd.to_datetime(df_rainfall.Date)

df_rainfall['Year'] = df_rainfall['Date'].dt.year

df_rainfall['Month'] = df_rainfall['Date'].dt.month

df_rainfall['Month_name'] = df_rainfall['Date'].dt.month_name()

df_rainfall_grouped_by_year = df_rainfall.groupby('Year').agg({'POONDI':'mean','CHOLAVARAM':'mean','REDHILLS':'mean','CHEMBARAMBAKKAM':'mean'})
df_rainfall_grouped_by_year.head()
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import seaborn as sns
df_rainfall_grouped_by_year = df_rainfall_grouped_by_year.reset_index()
df_reservoir_level_max_cap_ratio_Xnew = np.linspace(df_reservoir_level_max_cap_ratio['Year'].min(),df_reservoir_level_max_cap_ratio['Year'].max(),500)

df_rainfall_grouped_by_year_Xnew = np.linspace(df_rainfall_grouped_by_year['Year'].min(),df_rainfall_grouped_by_year['Year'].max(),500)



fig = plt.figure(figsize=(24,16))



ax = fig.add_subplot(221)

ax2 = ax.twinx()

sns.set_style("darkgrid")

plt.xlabel('Year')

plt.title("POONDI")

ax.set_ylabel('Reservior Level(%)')

ax2.set_ylabel('Rainfall')

f = interp1d(df_reservoir_level_max_cap_ratio['Year'],df_reservoir_level_max_cap_ratio['POONDI_%_Fill'], kind='quadratic')

df_reservoir_level_max_cap_ratio_Ynew = f(df_reservoir_level_max_cap_ratio_Xnew)

ax.plot(df_reservoir_level_max_cap_ratio_Xnew,df_reservoir_level_max_cap_ratio_Ynew, label = 'Water Level',color='r')



f = interp1d(df_rainfall_grouped_by_year['Year'],df_rainfall_grouped_by_year['POONDI'], kind='quadratic')

df_rainfall_grouped_by_year_Ynew = f(df_rainfall_grouped_by_year_Xnew)

ax2.plot(df_rainfall_grouped_by_year_Xnew,df_rainfall_grouped_by_year_Ynew,color='r',linestyle='dashed',label = 'Rainfall')

ax.legend(loc="upper right")

ax2.legend(loc="upper left")





ax = fig.add_subplot(222)

ax2 = ax.twinx()

sns.set_style("darkgrid")

plt.xlabel('Year')

plt.title("CHOLAVARAM")

ax.set_ylabel('Reservior Level(%)')

ax2.set_ylabel('Rainfall')

f = interp1d(df_reservoir_level_max_cap_ratio['Year'],df_reservoir_level_max_cap_ratio['CHOLAVARAM_%_Fill'], kind='quadratic')

df_reservoir_level_max_cap_ratio_Ynew = f(df_reservoir_level_max_cap_ratio_Xnew)

ax.plot(df_reservoir_level_max_cap_ratio_Xnew,df_reservoir_level_max_cap_ratio_Ynew, label = 'Water Level',color='b')



f = interp1d(df_rainfall_grouped_by_year['Year'],df_rainfall_grouped_by_year['CHOLAVARAM'], kind='quadratic')

df_rainfall_grouped_by_year_Ynew = f(df_rainfall_grouped_by_year_Xnew)

ax2.plot(df_rainfall_grouped_by_year_Xnew,df_rainfall_grouped_by_year_Ynew,color='b',linestyle='dashed',label = 'Rainfall')

ax.legend(loc="upper right")

ax2.legend(loc="upper left")





ax = fig.add_subplot(223)

ax2 = ax.twinx()

sns.set_style("darkgrid")

plt.xlabel('Year')

plt.title("REDHILLS")

ax.set_ylabel('Reservior Level(%)')

ax2.set_ylabel('Rainfall')

f = interp1d(df_reservoir_level_max_cap_ratio['Year'],df_reservoir_level_max_cap_ratio['REDHILLS_%_Fill'], kind='quadratic')

df_reservoir_level_max_cap_ratio_Ynew = f(df_reservoir_level_max_cap_ratio_Xnew)

ax.plot(df_reservoir_level_max_cap_ratio_Xnew,df_reservoir_level_max_cap_ratio_Ynew, label = 'Water Level',color='m')



f = interp1d(df_rainfall_grouped_by_year['Year'],df_rainfall_grouped_by_year['REDHILLS'], kind='quadratic')

df_rainfall_grouped_by_year_Ynew = f(df_rainfall_grouped_by_year_Xnew)

ax2.plot(df_rainfall_grouped_by_year_Xnew,df_rainfall_grouped_by_year_Ynew,color='m',linestyle='dashed',label = 'Rainfall')

ax.legend(loc="upper right")

ax2.legend(loc="upper left")



ax = fig.add_subplot(224)

ax2 = ax.twinx()

sns.set_style("darkgrid")

plt.xlabel('Year')

plt.title("CHEMBARAMBAKKAM")

ax.set_ylabel('Reservior Level(%)')

ax2.set_ylabel('Rainfall')

f = interp1d(df_reservoir_level_max_cap_ratio['Year'],df_reservoir_level_max_cap_ratio['CHEMBARAMBAKKAM_%_Fill'], kind='quadratic')

df_reservoir_level_max_cap_ratio_Ynew = f(df_reservoir_level_max_cap_ratio_Xnew)

ax.plot(df_reservoir_level_max_cap_ratio_Xnew,df_reservoir_level_max_cap_ratio_Ynew, label = 'Water Level',color='g')



f = interp1d(df_rainfall_grouped_by_year['Year'],df_rainfall_grouped_by_year['CHEMBARAMBAKKAM'], kind='quadratic')

df_rainfall_grouped_by_year_Ynew = f(df_rainfall_grouped_by_year_Xnew)

ax2.plot(df_rainfall_grouped_by_year_Xnew,df_rainfall_grouped_by_year_Ynew,color='g',linestyle='dashed',label = 'Rainfall')

ax.legend(loc="upper right")

ax2.legend(loc="upper left")