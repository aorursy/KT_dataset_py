import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
shoes = pd.read_csv('../input/7210_1.csv')

shoes.head()
# Here only choose 5 columns interesting.

columns = ['brand', 'prices.amountMin', 'prices.amountMax', 'prices.isSale', 'prices.currency']
shoes = shoes[columns]

shoes.dropna(inplace=True)

shoes['prices.amountAverage'] = (shoes['prices.amountMin'] + shoes['prices.amountMax']) / 2
shoes['prices.currency'].value_counts()
# Show some not USD shoes

shoes[shoes['prices.currency'] != 'USD'].head(10)
# Transform different currency to USD

for row in shoes.itertuples():

    if row._5 == 'CAD':

        shoes['prices.amountMin'][row.Index] *= 0.73

        shoes['prices.amountMax'][row.Index] *= 0.73

        shoes['prices.amountAverage'][row.Index] *= 0.73

    elif row._5 == 'EUR':

        shoes['prices.amountMin'][row.Index] *= 1.1

        shoes['prices.amountMax'][row.Index] *= 1.1

        shoes['prices.amountAverage'][row.Index] *= 1.1

    elif row._5 == 'AUD':

        shoes['prices.amountMin'][row.Index] *= 0.75

        shoes['prices.amountMax'][row.Index] *= 0.75

        shoes['prices.amountAverage'][row.Index] *= 0.75

    elif row._5 == 'GPB':

        shoes['prices.amountMin'][row.Index] *= 1.3

        shoes['prices.amountMax'][row.Index] *= 1.3

        shoes['prices.amountAverage'][row.Index] *= 1.3
# Make sure it did the transformation

shoes[shoes['prices.currency'] != 'USD'].head(10)
data = shoes.groupby('brand')['prices.amountAverage'].mean().sort_values(ascending=False).head(10)

ax = data.plot(kind='barh', figsize=(10, 6))

ax.invert_yaxis()

plt.xlabel('Price(USD)')

plt.title('Most experience average price brand')
data = shoes.groupby('brand')['prices.amountAverage'].max().sort_values(ascending=False).head(10)

ax = data.plot(kind='barh', figsize=(10, 6))

ax.invert_yaxis()

plt.xlabel('Price(USD)')

plt.title('Most experience single price brand')
grouped = shoes.groupby('brand')['prices.amountAverage']

data = grouped.apply(lambda x:x.max() - x.min()).sort_values(ascending=False).head(10)

ax = data.plot(kind='barh', figsize=(10, 6))

ax.invert_yaxis()

plt.xlabel('Price(USD)')

plt.title('Most widest distributed price brand')
shoes['prices.amountAverage'].hist(bins=50, figsize=(10, 6))

plt.title('Prices Distribution across brands')
shoes['brand'].value_counts().head(10)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for idx, brand in enumerate(shoes['brand'].value_counts()[0:10].index):

    axs[idx//5, idx%5].hist(shoes[shoes['brand'] == brand]['prices.amountAverage'], bins=20)

    axs[idx//5, idx%5].set_title(brand)

plt.suptitle('Price Distributions of Specific Brand')

plt.tight_layout()

fig.subplots_adjust(top=0.88)