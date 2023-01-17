import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



data = pd.read_csv('../input/weather-istanbul-data-20092019/Istanbul Weather Data.csv')
data.describe()
print(data['Condition'].value_counts(dropna = False))
data.MaxTemp.plot(kind = 'hist',bins = 50,figsize = (15,15))

plt.xlabel('MaxTemp')

plt.show()
data[(data['AvgHumidity']>90) & (data['Rain']>10)]
data["rain_level"]=["high" if i > 10 else "low" for i in data.Rain]

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data[(data['rain_level']=="high")].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.plot(kind='scatter',x='MinTemp',y='Rain')

plt.show()