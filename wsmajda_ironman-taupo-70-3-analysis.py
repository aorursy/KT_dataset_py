# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure

import math



from datetime import datetime

from matplotlib.ticker import MaxNLocator

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

nRowsRead = 1000 

im = pd.read_csv('../input/ironman-703-taupo-2020/im-taupo.csv', delimiter=',', nrows = nRowsRead)

nRow, nCol = im.shape

del im['0']



#convert data from h:mm:ss to min



im.drop(im.filter(regex="Unname"),axis=1, inplace=True)

im['Gender'] = im['Gender'].astype(str).apply(lambda x: 'Male' if x.startswith('Male') else 'Female')    



time = im['Time'].astype(str).str.split(':')

im['Time'] = time.apply(lambda x: int(x[0]) * 60 + int(x[1]))



swim = im['Swim'].astype(str).str.split(':')

im['Swim'] = swim.apply(lambda x: int(x[0]) * 60 + int(x[1]))



t1 = im['T1'].astype(str).str.split(':')

im['T1'] = t1.apply(lambda x: int(x[0]) * 60 + int(x[1]))



bike = im['Bike'].astype(str).str.split(':')

im['Bike'] = bike.apply(lambda x: int(x[0]) * 60 + int(x[1]))



t2 = im['T2'].astype(str).str.split(':')

im['T2'] = t2.apply(lambda x: int(x[0]) * 60 + int(x[1]))



run = im['Run'].astype(str).str.split(':')

im['Run'] = run.apply(lambda x: int(x[0]) * 60 + int(x[1]))

#Regex on cat col

im['Cat'] = im['Cat'].str.extract(r'((([A-Z])\w+)|([\d][\d]-[\d][\d]))')

im.rename(columns={"Time": "Time(min)", "Swim": "Swim(min)", "T1": "T1(min)", "Bike": "Bike(min)", "T2": "T2(min)", "Run": "Run(min)"}, inplace=True)

im['BikeSpeed'] = [90/(i/60) for i in im['Bike(min)']]  

im['RunPace'] = [i/21.09 for i in im["Run(min)"]]

im['SwimPace'] = [(i*100)/1800 for i in im["Swim(min)"]]



im.head(30)
x = im[(im['Cat'].str.contains('Pro')) & (im['Gender'] == 'Male')]

y = im[(im['Cat'].str.contains('Pro')) & (im['Gender'] == 'Female')]



plt.figure(figsize=(20,10))



plt.hist(x['Time(min)'], alpha=0.5, color='blue', label="Male", bins=20)

plt.hist(y['Time(min)'], alpha=0.4, color='pink', label="Female", bins=20)



plt.legend(loc="upper right")

plt.xlabel('Time (min)')

plt.ylabel('Occurrences')

plt.tick_params(axis='both',labelsize=14)



plt.title('Histogram: Pro athlete - men vs woman')



plt.show 
x1 = im[(~im['Cat'].str.contains('Pro')) & (im['Gender'] == 'Male')]

y2 = im[(~im['Cat'].str.contains('Pro')) & (im['Gender'] == 'Female')]
plt.figure(figsize=(20,10))



plt.hist(x1['Time(min)'], alpha=0.5, color='blue', label="Male", bins=20)

plt.hist(y2['Time(min)'], alpha=0.75, color='pink', label="Female", bins=20)



plt.legend(loc="upper right")

plt.xlabel('Time (min)')

plt.ylabel('Occurrences')

plt.tick_params(axis='both',labelsize=14)



plt.title('Histogram: Amateurs athlete - men vs woman')



plt.show 
cats_u =  im['Cat'].unique()

x = im.groupby(['Cat'])['Gender'].count()

plt.figure(figsize=(20,10))

plt.bar(range(len(x)), x, label='Category', color='green')

plt.legend(loc="upper right")

plt.xlabel('Category')

plt.ylabel('Participants number in group')



ax = plt.subplot()

ax.set_xticks(range(len(x)))

ax.set_xticklabels(['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','Pro'])



plt.show()

im.groupby(['Cat', 'Gender'])['Gender'].count().unstack('Cat').plot.bar(figsize = (20,10),rot=0)

boxplot = im.boxplot(column=['Swim(min)', 'Bike(min)', 'Run(min)'], figsize=(12,8))

plt.figure(figsize=(20,15))

sns.set(style="whitegrid")

ax = sns.boxplot(x="Cat", y="Swim(min)", data=im, palette="Set3")

ax = sns.swarmplot(x="Cat", y="Swim(min)", data=im, color="0.25")
plt.figure(figsize=(20,15))

sns.set(style="whitegrid")

ax = sns.boxplot(x="Cat", y="Bike(min)", data=im, palette="Set3")

ax = sns.swarmplot(x="Cat", y="Bike(min)", data=im, color="0.25")
plt.figure(figsize=(20,15))

sns.set(style="whitegrid")

ax = sns.boxplot(x="Cat", y="Run(min)", data=im, palette="Set3")

ax = sns.swarmplot(x="Cat", y="Run(min)", data=im, color="0.25")
x = im[im['Cat'] == 'Pro']

plt.figure(figsize=(20,15))

ax = sns.barplot(x='BikeSpeed', y='Name', data=x)

# ax.set_xlabel('totalCount')
y = im[im['Cat'] != 'Pro'].head(30)

plt.figure(figsize=(20,15))

plt.xlim(30, 42)



ax = sns.barplot(x='BikeSpeed', y='Name', data=y)



  

plt.figure(figsize=(20,15))

ax = sns.barplot(x='RunPace'  , y='Name', data=x)
plt.figure(figsize=(20,15))

ax = sns.barplot(x='RunPace', y='Name', data=y)
plt.figure(figsize=(20,15))

ax = sns.barplot(x='SwimPace', y='Name', data=x)
plt.figure(figsize=(20,15))

ax = sns.barplot(x='SwimPace', y='Name', data=y)