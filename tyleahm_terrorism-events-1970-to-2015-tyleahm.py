# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
##Read in CSV

Terr = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1')

Terr.head()
##Select only the columns I want

Terr2 = Terr[['eventid','region_txt', 'country_txt', 'iyear', 'imonth', 'iday', 'doubtterr', 'targtype1_txt','latitude', 'longitude', 'specificity', 'attacktype1_txt']]

Terr2.head()
##Selecting only records from 2012 and up

Terr3 = Terr2.loc[Terr2['iyear'] >= 2012]

Terr3.head()
##County by year from 2012

sum_year = Terr3.groupby('iyear').count()
##Allows you to pull one pie section out

explode = (0, 0, 0, 0.1)

sum_year.iday.plot(kind='pie', explode = explode)
fig = plt.figure(figsize=(6,6), dpi=125)

ax = plt.subplot(111)

explode = (0, 0, 0, 0.1)



sum_year.iday.plot(kind='pie', explode = explode, ax=ax, autopct='%1.1f%%', startangle=270, fontsize=17)
sum_region = Terr3.groupby('region_txt').count()

sum_region.head()
fig = plt.figure(figsize=(8,8), dpi=125)

ax = plt.subplot(111)

##Percentages show in Pie chart , autopct='%1.1f%%'

sum_region.iday.plot(kind='pie', labels = None, ax=ax)

ax.legend(loc="best", labels=sum_region.iday.index)
ax = sum_region[['country_txt']].plot(kind='bar', title ="V comp",figsize=(6,6),legend=True, fontsize=12)

ax.set_xlabel("country_txt",fontsize=12)

ax.set_ylabel("targtype1_txt",fontsize=12)
sum_month = Terr3.groupby('imonth').count()
ax = sum_month[['country_txt']].plot(kind='bar', title ="V comp",figsize=(6,6),legend=True, fontsize=12)

ax.set_xlabel("month",fontsize=12)

ax.set_ylabel("count",fontsize=12)
sum_attack = Terr3.groupby('attacktype1_txt').count()

ax = sum_attack[['country_txt']].plot(kind='bar', title ="V comp",figsize=(6,6),legend=True, fontsize=12)

ax.set_xlabel("Attack Type",fontsize=12)

ax.set_ylabel("count",fontsize=12)
category_group=Terr3.groupby(['iyear','attacktype1_txt']).sum()

category_group.head()
ax = category_group['eventid'].plot(kind='bar', title ="V comp",figsize=(12,6),legend=True, fontsize=12)

ax.set_xlabel("Attack Type",fontsize=12)

ax.set_ylabel("count",fontsize=12)
Terr4 = Terr3[['eventid','iyear','attacktype1_txt']]

category_group2 =Terr4.groupby(['iyear','attacktype1_txt']).sum()

category_group2.unstack().head()