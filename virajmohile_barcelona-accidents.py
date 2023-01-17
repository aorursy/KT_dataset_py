import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
d = pd.read_csv("../input/accidents_2017.csv")
d.head()
list(d)
d1 = d['Weekday'].value_counts()

d1 = pd.DataFrame(d1)

#d1.columns = ['Weekdays', 'Percentage']

y1 = d1

d1
d1 = d1 / sum(d1['Weekday'])
d1
d1.sort_values( by ='Weekday')
d1 = d1 * 100

#Percentages are prettier to look at:)
d1
s = np.std(d1['Weekday'])
s
m = np.mean(d1['Weekday'])
d1 = d1 - m
d1
d1 = d1 / s
d1
d.head()
d2 = d.iloc[:,[4,8]]

d2.head()
d3 = d2.loc[d2['Weekday'] == 'Friday']
d3.head()
pd.unique(d['Part of the day'])
d4 = d3['Part of the day'].value_counts()
d4 = pd.DataFrame(d4)
d4
d4 = d4 / sum(d4['Part of the day'])
d4 = d4 * 100
d4
d5 = d.iloc[:, 8]
d5.head()
d5 = d5.value_counts()
d5 = pd.DataFrame(d5)

d5

ei = d5

ei
d5 = d5 / sum(d5['Part of the day'])

d5 = d5 * 100
d5
d6 = d2.loc[d2['Weekday'] == 'Tuesday']
d6 = d6['Part of the day'].value_counts()
d6 = pd.DataFrame(d6)

d6
d6 = d6 / sum(d6['Part of the day'])

d6 = d6 * 100

d6
d.head()
d7 = d.iloc[:, 6]

d7 = pd.DataFrame(d7)

d7.head()
w1 = d7.loc[d7['Day']<=7]

w2 = d7.loc[(d7['Day']>7) & (d7['Day']<=14)]

w3 = d7.loc[(d7['Day']>14) & (d7['Day']<=21)]

w4 = d7.loc[(d7['Day']>21) & (d7['Day']<=31)]

w1 = w1.describe()

w4.describe()
w1 = 2342

w2 = 2346

w3 = 2508

w4 = 3143

w = w1 + w2 + w3 + w4
w1 = w1 / w

w1 = w1 * 100

w2 = w2 / w

w2 = w2 * 100

w3 = w3 / w

w3 = w3 * 100

w4 = w4 / w

w4 = w4 * 100

w1
w2
w3
w4
d.head()
e1 = d.loc[d['Serious injuries']!=0]
e1.head()
e1 = e1.iloc[:, [1,3,4,6,7,8,10]]
e1.head()
e2 = e1['Part of the day'].value_counts()
type(e2)

eii = e2
e2 = pd.DataFrame(e2)

e2 = e2 / sum(e2['Part of the day'])

e2 = e2 * 100
e2
e2
d5
ei
eii = pd.DataFrame(eii)

eii
e1W = eii['Part of the day'] / ei['Part of the day']
e1W * 100
ei
ei = sum(ei['Part of the day'])

ei
eii
eii = eii['Part of the day'] / ei
eii = eii * 100
eii = pd.DataFrame(eii)

eii
er = e1['Weekday'].value_counts()

er2 = er

er2 = pd.DataFrame(er2)
er = pd.DataFrame(er)

er
er = er['Weekday'] / sum(y1['Weekday'])

er = er * 100

er = pd.DataFrame(er)

er
er1 = er2['Weekday'] / y1['Weekday']

er1 = er1 * 100

er1 = pd.DataFrame(er1)

er1
er1 = er1.sort_values(by = ['Weekday'], ascending = False)
er1 = pd.DataFrame(er1)

#er1

er1 = er1.rename(columns = {'Weekday' : 'P(S_Accident | Weekday)'})

er1
er = er.rename(columns = {'Weekday' : 'P(S_Accident)'})

er
pd.concat([er, er1], axis = 1)