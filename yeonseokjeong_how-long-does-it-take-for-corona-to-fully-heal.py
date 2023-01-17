import pandas as pd
df = pd.read_csv('../input/coronavirusdataset/patient.csv')

df.head()
df['age'] = 2020-df['birth_year']
df_released = df[df['state'] == 'released']

df_released.head()
df_released = df_released.reset_index(drop = True)

df_released.tail()
df_released.info()
df_released = df_released[df_released.released_date.notna()]

df_released.tail()
date_cols = ["confirmed_date", "released_date", "deceased_date"]

for col in date_cols:

    df_released[col] = pd.to_datetime(df_released[col])
df_released["timedelta_to_release_since_confirmed"] = df_released["released_date"] - df_released["confirmed_date"]
time_to_recover_series = df_released["timedelta_to_release_since_confirmed"].apply(lambda timedelta: int(timedelta.days))

df_released["time_to_release_since_confirmed"] = df_released["timedelta_to_release_since_confirmed"].apply(lambda timedelta: int(timedelta.days))

time_to_recover_series[:5]
time_to_recover = time_to_recover_series.values

time_to_recover 
import seaborn as sns

import matplotlib.pyplot as plt
print('min_value: '+ str(time_to_recover.min()) + ' days')

print('max_value: '+ str(time_to_recover.max()) + ' days')

print('mean: '+ str(time_to_recover.mean()))

print('std: '+ str(time_to_recover.std()))
plt.figure(figsize=(10,5))

sns.distplot(time_to_recover, color = 'red')

plt.xlim(time_to_recover.min(),time_to_recover.max())

plt.xticks(range(time_to_recover.max()));

plt.xlabel('day');
plt.figure(figsize=(10,5))

sns.kdeplot(time_to_recover, color = 'red')

plt.xlim(time_to_recover.min(),time_to_recover.max())

plt.xticks(range(time_to_recover.max()));

plt.xlabel('day');
import scipy as sp
print('skewness: ' + str(sp.stats.skew(time_to_recover)))

print('kurtosis: ' + str(sp.stats.kurtosis(time_to_recover)))
plt.figure(figsize=(20,10))

ax = sns.barplot(data = df_released, x="age", y="time_to_release_since_confirmed",

                 saturation=1)

plt.title('The time it takes for the corona to heal completely with age');
df_deceased = df[df['state'] == 'deceased']

df_deceased.head()
df_deceased = df_deceased.reset_index(drop = True)

df_deceased.tail()
df_deceased.info()
date_cols = ["confirmed_date", "released_date", "deceased_date"]

for col in date_cols:

    df_deceased[col] = pd.to_datetime(df_deceased[col])

    

df_deceased["timedelta_to_decease_since_confirmed"] = df_deceased["deceased_date"] - df_deceased["confirmed_date"]

df_deceased["time_to_decease_since_confirmed"] = df_deceased["timedelta_to_decease_since_confirmed"].apply(lambda timedelta: int(timedelta.days))

df_deceased["time_to_decease_since_confirmed"][:5]
df_deceased = df_deceased[df_deceased["time_to_decease_since_confirmed"]>=0]

df_deceased = df_deceased.reset_index(drop = True)

df_deceased.head()
sns.kdeplot(data=df_deceased['age'],label='deceased', color='black', shade=True)

sns.kdeplot(data=df_released['age'],label='released', color='red', shade=True);