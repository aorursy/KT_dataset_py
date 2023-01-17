import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df= pd.read_csv('/kaggle/input/ae-attendances-england/AE_attendances_england_monthly.csv')

df.head()
UK = df.loc[:,['date', 'Total attendances' ]].groupby('date').sum().reset_index()
fig, ax = plt.subplots(figsize=(15,10)) 



sns.lineplot(UK.date, UK.iloc[:,1])

sns.despine()



ax.xaxis.set_major_locator(plt.MaxNLocator(30))

fig.autofmt_xdate()
london_hospitals = [n for n in list(df.Name.dropna().unique()) if 'London' in n]



london = df[df['Name'].isin(london_hospitals)]
fig, ax = plt.subplots(figsize=(15,10)) 



sns.lineplot(x='date', y='Total attendances', data=london, hue='Name')

sns.despine()



ax.xaxis.set_major_locator(plt.MaxNLocator(30))

fig.autofmt_xdate()