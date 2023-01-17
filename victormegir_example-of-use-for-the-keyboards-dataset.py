import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = [10, 10]

plt.rcParams.update({'font.size': 16})
keyboards = pd.read_csv('../input/keyboards-from-skroutzgr/all_keyboards.csv', index_col=0)



keyboards.head()
sns.distplot(keyboards['Price'])

plt.xlabel('Price in EU Euros');
sns.pairplot(keyboards);
keyboards['Gaming'] = keyboards['Switches'].notna()

keyboards['Gaming']
regular_avg_price = keyboards.loc[keyboards['Gaming'] == False]['Price'].mean()

gaming_avg_price = keyboards.loc[keyboards['Gaming'] == True]['Price'].mean()



print('Gaming keyboards average price: {:.2f} Euros. Regular keyboards average price: {:.2f} Euros.'.format(gaming_avg_price, regular_avg_price))

print('Gaming keyboards are on average {:.2f}% more expensive than regular keyboards.'.format((gaming_avg_price / regular_avg_price - 1) * 100))