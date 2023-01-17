print('Ready to go!')
import pandas as pd # pandas, our library for representing tabular data

import matplotlib.pyplot as plt # matplotlib, our library for data viz

plt.rc('figure', figsize=(12, 8)) # default figure size

import seaborn as sns # additional data viz library



# load our data

df = pd.read_csv('/kaggle/input/lemonade_clean.csv')

df
df['sales']
df['sales'].describe()
plt.hist(df['sales'])
plt.hist(df['sales'], edgecolor='pink', color='darkred', bins=20)
plt.boxplot(df['sales'])
sns.barplot(data=df, y='sales', x='rainfall')
sns.barplot(data=df, y='rainfall', x='sales')
plt.scatter(data=df, y='sales', x='temperature')