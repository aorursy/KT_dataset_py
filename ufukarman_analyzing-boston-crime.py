import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
md = pd.read_csv('../input/boston-crime-data/crime.csv', sep=',', engine='python')
md.shape
md.head()
md.columns
md.isnull().sum()
md.dropna(subset=['Lat'], inplace = True)
md.shape
md.describe()
md.groupby('YEAR').count()
md.groupby('YEAR').nunique()
md.drop_duplicates(['INCIDENT_NUMBER'], inplace=True)
md.groupby('YEAR').count()['INCIDENT_NUMBER']
yc = md.groupby('YEAR').count()['INCIDENT_NUMBER'].reset_index()
yc
mp = sns.barplot(x = 'YEAR' , y='INCIDENT_NUMBER', data = yc)
md.groupby(['YEAR','OFFENSE_CODE_GROUP']).count()['INCIDENT_NUMBER']
yd = md.groupby(['YEAR','MONTH']).count()
yd.head()
mnd = md[(md['MONTH']>=6) & (md['MONTH']<=9)]
mnd.head()
mnd.groupby('YEAR')['MONTH'].max()
mnd.groupby('YEAR')['MONTH'].min()
mnd.groupby('YEAR').count()['INCIDENT_NUMBER']
mnd.groupby(['YEAR','OFFENSE_CODE_GROUP']).count()['INCIDENT_NUMBER']
mnd.groupby(['YEAR','OFFENSE_CODE_GROUP']).count()['INCIDENT_NUMBER'].reset_index()
mnd.groupby('YEAR').nunique()
mnd.groupby(['YEAR' , 'MONTH']).count()
mnd.groupby(['DISTRICT' , 'YEAR']).count()
mnd.groupby(['YEAR' , 'DISTRICT']).count()['INCIDENT_NUMBER'].reset_index()
mnd.groupby(['DISTRICT' , 'YEAR']).count()['INCIDENT_NUMBER'].reset_index()
mnd.groupby(['DISTRICT' , 'OFFENSE_CODE_GROUP']).count()['INCIDENT_NUMBER'].reset_index()
ac = mnd.groupby('DISTRICT').count()['INCIDENT_NUMBER'].reset_index()
ac
sns.barplot(x = 'DISTRICT' , y='INCIDENT_NUMBER', data = ac)
mi = mnd.groupby('MONTH').count()['INCIDENT_NUMBER'].reset_index()
sns.barplot(x = 'MONTH' , y='INCIDENT_NUMBER', data = mi)
oi = mnd.groupby('OFFENSE_CODE_GROUP').count()['INCIDENT_NUMBER'].reset_index()
oi
mnd.groupby(['MONTH' , 'DISTRICT' , 'OFFENSE_CODE_GROUP']).count()['INCIDENT_NUMBER'].reset_index()
mnd.groupby(['OFFENSE_CODE_GROUP' , 'DISTRICT']).count()['INCIDENT_NUMBER'].reset_index()
label = mnd['OFFENSE_CODE_GROUP'].unique()
label
sizes = oi['INCIDENT_NUMBER'].values
sizes
fig1, ax1 = plt.subplots()
ax1.pie(sizes , labels=label)
di = mnd.groupby('DAY_OF_WEEK').count()['INCIDENT_NUMBER'].reset_index()
ax1.axis('equal')
di = di.sort_values('INCIDENT_NUMBER' , ascending=False)
di
explode = (0.1, 0.08,0.06,0.04,0.02,0,0)
plt.pie(di['INCIDENT_NUMBER'] , labels=di['DAY_OF_WEEK'], explode=explode , autopct='%1.1f%%', shadow=True)
plt.show()
mnd.describe()
mnd.describe(include=['O'])
mnd['DISTRICT'].loc[mnd['YEAR']==2015].value_counts().plot.bar()
mnd['DISTRICT'].loc[mnd['YEAR']==2016].value_counts().plot.bar()
mnd['DISTRICT'].loc[mnd['YEAR']==2017].value_counts().plot.bar()
mnd['DISTRICT'].loc[mnd['YEAR']==2018].value_counts().plot.bar()
i = 6



while i < 10:

    print('== ' + str(i) + ' ==')

    print(mnd['DISTRICT'].loc[mnd['MONTH']==i].value_counts())

    i +=1
mnd.groupby(['DISTRICT' , 'DAY_OF_WEEK']).count()['INCIDENT_NUMBER'].reset_index()
mnd.groupby(['DAY_OF_WEEK' , 'DISTRICT']).count()['INCIDENT_NUMBER'].reset_index()
mnd['DAY_OF_WEEK'].value_counts().plot.bar()
mnd['HOUR'].value_counts().plot.bar()
mnd['UCR_PART'].value_counts().plot.bar()
mnd.groupby(['DAY_OF_WEEK' , 'SHOOTING']).count()['INCIDENT_NUMBER'].reset_index()
mnd.groupby('YEAR').nunique()
mnd.describe(include=['O'])
hi = mnd.groupby(['HOUR']).count()['INCIDENT_NUMBER'].reset_index()
sns.barplot(x = 'HOUR' , y='INCIDENT_NUMBER', data = hi)
mnd[mnd['HOUR']==11]
mnd.groupby(['DISTRICT' , 'Lat' , 'Long']).count()['INCIDENT_NUMBER'].reset_index()
mmnd = mnd[mnd['DISTRICT']=='A1']
mmnd
amnd = mnd[mnd['DISTRICT']=='A15']

aamnd = mnd[mnd['DISTRICT']=='A7']

emnd = mnd[mnd['DISTRICT']=='E5']

eemnd = mnd[mnd['DISTRICT']=='E13']

eeemnd = mnd[mnd['DISTRICT']=='E18']

ddmnd = mnd[mnd['DISTRICT']=='D14']

cmnd = mnd[mnd['DISTRICT']=='C6']

bbmnd = mnd[mnd['DISTRICT']=='B3']

dmnd = mnd[mnd['DISTRICT']=='D4']

ccmnd = mnd[mnd['DISTRICT']=='C11']

bmnd = mnd[mnd['DISTRICT']=='B2']
A15 = amnd['OFFENSE_CODE_GROUP'].value_counts().head()

A7 = aamnd['OFFENSE_CODE_GROUP'].value_counts().head()

E5 = emnd['OFFENSE_CODE_GROUP'].value_counts().head()

E13 = eemnd['OFFENSE_CODE_GROUP'].value_counts().head()

E18 = eeemnd['OFFENSE_CODE_GROUP'].value_counts().head()

D14 = ddmnd['OFFENSE_CODE_GROUP'].value_counts().head()

C6 = cmnd['OFFENSE_CODE_GROUP'].value_counts().head()

B3 = bbmnd['OFFENSE_CODE_GROUP'].value_counts().head()

D4 = dmnd['OFFENSE_CODE_GROUP'].value_counts().head()

C11 = ccmnd['OFFENSE_CODE_GROUP'].value_counts().head()

B2 = bmnd['OFFENSE_CODE_GROUP'].value_counts().head()
A15
A7
E5
E13
E18
D14
C6
B3
B2
D4
C11
mnd.groupby('YEAR').nunique()