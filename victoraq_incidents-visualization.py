
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
# calls = pd.read_csv('../input/police-department-calls-for-service.csv')

incidents = pd.read_csv('../input/police-department-incidents.csv')
incidents['Time'] = pd.to_datetime(incidents['Time'], format='%H:%M')
incidents['Date'] = pd.to_datetime(incidents['Date'])
incidents['hour'] = incidents['Time'].dt.hour
incidents['wday'] = incidents['Date'].dt.dayofweek
# calls.head()
incidents.head()
# calls['Original Crime Type Name'].value_counts()[:20].plot(kind='barh')
incidents['Category'].value_counts()[:20].plot(kind='barh')
thefts = incidents[incidents['Category'] == 'LARCENY/THEFT']
thefts_weekdays = thefts[thefts['wday'] < 5]
thefts_weekends = thefts[thefts['wday'] >= 5]

thefts.head()
fig, (ax1,ax2) = plt.subplots(ncols=2, nrows=1,figsize=(20,10))
ax1.set_xlim(37.65, 37.85)
ax1.set_ylim(-122.53,-122.35)
ax1.set_title('Weekdays')
ax1.scatter(thefts_weekdays['Y'],thefts_weekdays['X'], s=0.01, alpha=1)

ax2.set_xlim(37.65, 37.85)
ax2.set_ylim(-122.53,-122.35)
ax2.set_title('Weekends')
ax2.scatter(thefts_weekends['Y'],thefts_weekends['X'], s=0.01, alpha=1)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(14,4.5)

ax1.set_title('Weekdays')
sns.distplot(incidents[incidents['wday'] < 5]['hour'], label='Total', ax=ax1)
sns.distplot(thefts_weekdays['hour'], label='Lacerny/Thefts',color='r', ax=ax1)

ax2.set_title('Weekends')
sns.distplot(incidents[incidents['wday'] >= 5]['hour'], label='Total', ax=ax2)
sns.distplot(thefts_weekends['hour'], label='Lacerny/Thefts',color='r', ax=ax2)

ax1.set_xlim(0,24)
ax1.set_ylim(0,0.25)
ax1.xaxis.set_ticks(np.arange(0, 24))
ax1.legend()
ax2.set_xlim(0,24)
ax2.set_ylim(0,0.25)
ax2.xaxis.set_ticks(np.arange(0, 24))
ax2.legend()
