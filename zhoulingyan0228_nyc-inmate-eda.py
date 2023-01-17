import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import date
import math
df= pd.read_csv('../input/daily-inmates-in-custody.csv').drop('INMATEID',axis=1)
df['TS'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').timestamp())
df['YEAR'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').year)
df['MONTH'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').month)
df['DAY'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').day)
df['HOUR'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').hour)
df.describe(include='all')
df = df.drop(['DISCHARGED_DT', 'SEALED'],axis=1)
df['ISADULT'] = df['AGE']>18
plt.figure(figsize=(60, 60))
plt.subplot(3,2,1)
sns.violinplot(x="SRG_FLG", y="AGE", data=df);
plt.subplot(3,2,2)
sns.violinplot(x="RACE", y="AGE", data=df);
plt.subplot(3,2,3)
sns.violinplot(x="GENDER", y="AGE", data=df);
plt.subplot(3,2,4)
sns.violinplot(x="BRADH", y="AGE", data=df);
plt.subplot(3,2,5)
sns.violinplot(x="INFRACTION", y="AGE", data=df);
sns.factorplot(x="RACE", hue="GENDER", col="ISADULT", row='INFRACTION',
               data=df, kind="count");
sns.factorplot(x="YEAR", hue="RACE", 
               data=df, kind="count", order=np.arange(2015,2019));
sns.factorplot(x="MONTH", hue="YEAR", 
               data=df[df['YEAR']!=2018], kind="count", order=np.arange(1,13));
sns.factorplot(x="MONTH", hue="SRG_FLG", 
               data=df[df['YEAR']==2017], kind="count", order=np.arange(1,13));
countSeries = df[df['YEAR']>2010].groupby(pd.cut(df.loc[df['YEAR']>2010, 'TS'], 100)).size()
binSeries = pd.Series(countSeries.index.values)
dateSeries = binSeries.apply(lambda x: date.fromtimestamp(x.left))
tsPlotDf = pd.DataFrame()
tsPlotDf['datestr'] = dateSeries.values
tsPlotDf['count'] = countSeries.values
tsPlotDf['date'] = tsPlotDf.index.values

plt.plot(countSeries.values);
plt.plot([math.exp(x**2*7/10000) for x in range(100)], 'r--');
plt.yscale('log');
plt.xticks(range(0, 100, 24), dateSeries.values[range(0, 100, 24)], size='small');