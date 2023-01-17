import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



%matplotlib inline



print("Setup Complete")
happines_filepath = "../input/world-happiness/2019.csv"



happines_data = pd.read_csv(happines_filepath, parse_dates=True, encoding = "cp1252")



happines_data.head()
happines_data.info()
happines_data.shape

print("There are {:,} rows ".format(happines_data.shape[0]) + "and {} columns in our data".format(happines_data.shape[1]))
happines_data.describe()
happines_data.isnull().sum()
happines_data.duplicated().sum()
happines_data.loc[happines_data['Healthy life expectancy'] > 1]
happines_data.loc[happines_data['Country or region'] == 'Bulgaria']
happines_data.sort_values(by="Generosity", ascending=False).head(10)
happines_data[happines_data['Score'] > 7.5]
print("There are {} countries that have a happiness score above 7.0 ".format(len(happines_data[happines_data['Score'] > 7])))
happines_data.loc[happines_data['Healthy life expectancy'] >= 0.7, 'Color'] = 'green'

happines_data.loc[(happines_data['Healthy life expectancy'] > 0.5) & (happines_data['Healthy life expectancy'] < 0.7), 'Color'] = 'blue'

happines_data.loc[happines_data['Healthy life expectancy'] <= 0.5, 'Color'] = 'red'
happines_data.head()
whr_color = happines_data.groupby('Color')
whr_color['Score'].describe().sort_values(by="mean",ascending=True).head(10)
fig, ax = plt.subplots()



ax.scatter(happines_data['Healthy life expectancy'], happines_data['Score'])



ax.set_title('WHR 2019')

ax.set_xlabel('Healthy life expectancy')

ax.set_ylabel('Score')
fig, ax = plt.subplots()



for i in range(len(happines_data['Score'])):

    ax.scatter(happines_data['Perceptions of corruption'][i], happines_data['Score'][i], color=happines_data['Color'][i])



ax.set_title('WHR 2019')

ax.set_xlabel('Perceptions of corruption')

ax.set_ylabel('Score')
fig, ax = plt.subplots()



ax.hist(happines_data['Healthy life expectancy'], bins=10, density=False, edgecolor='k', color='darkgreen', alpha=0.5)



ax.set_title('Healthy life expectancy')

ax.set_xlabel('Points')

ax.set_ylabel('Frequency')
sns.scatterplot(x='Perceptions of corruption', y='Score', hue='Score', data=happines_data)
sns.pairplot(happines_data, hue="Color", palette="husl")
happines_data.corr(method="pearson", min_periods=20)["Score"].sort_values(ascending=False)
happines_data.corr(method="pearson", min_periods=20)["Score"].abs().sort_values(ascending=False)
happines_data.corr(method="pearson", min_periods=20)
corr = happines_data.corr(method = "pearson")



f, ax = plt.subplots(figsize=(10, 10))



sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 

            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
happines_data[happines_data['Score'] > 4].shape[0]
happines_data[(happines_data['Score'] > 5.5) & (happines_data['Color'] == 'green')].shape[0]
float(len(happines_data[(happines_data['Score'] > 5.5) & (happines_data['Color'] == 'blue')]))/float(len(happines_data[happines_data['Score'] > 5.5]))