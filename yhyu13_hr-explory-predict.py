import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
df = pd.read_csv('../input/HR_comma_sep.csv')
df.info()
df.head()
df['sales'].unique()
df['promotion_last_5years'].unique()
df['salary'].unique()
df.mean()
df.mean()['average_montly_hours']/30
corrmat = df.corr()

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
sns.factorplot("sales", col="salary", col_wrap=4, data=df, kind="count", size=8, aspect=1)
print(df[df['salary'] == 'low'].size)

print(df[df['salary'] == 'medium'].size)

print(df[df['salary'] == 'high'].size)
plt.subplot(1,2,1)

df.groupby('sales').mean()['satisfaction_level'].plot(kind='bar')

plt.subplot(1,2,2)

df.groupby('salary').mean()['satisfaction_level'].plot(kind='bar')