import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

df = pd.read_csv('../input/database.csv', dtype=object)

df.head()
df.shape #638454 rows and 24 columns
#Top cities with maximum crime incidences reported

df.City.value_counts()[:5]
city_count = df.City.value_counts()[:10]

ax = city_count.plot(kind = 'bar', rot = 35, title = 'Cities with highest number of crime incidents reported')
#Top 10 states with reported crime

df.State.value_counts()[:5]
state_count = df.State.value_counts()[:10]

ax = state_count.plot(kind = 'bar', rot = 35, title = 'States with highest number of crime incidents reported')

df['Crime Type'].value_counts()
df['Perpetrator Race'].value_counts()
race = df['Perpetrator Race'].value_counts()

race.plot.pie(figsize=(6, 6), title = 'Perpetrator race in Homicide data')
df['Weapon'].value_counts()[:5]
weapon = df['Weapon'].value_counts()[:10]

weapon.plot.pie(figsize=(6, 6), title = 'Top 10 Weapons used in Homicide')
df.groupby(['Year', 'State']).Incident.count().reset_index()