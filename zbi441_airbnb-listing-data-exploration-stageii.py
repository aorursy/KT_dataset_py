import pandas as pd

import seaborn as sns

import matplotlib 

listings=pd.read_csv('../input/listings.csv')
type(listings.iloc[0,60])
#Explorer all listed prices and identify the outliers

listings['price']=listings['price'].str.replace('$','')

listings['price']=pd.to_numeric(listings['price'],errors='coerce')
listings['price'].describe()
%matplotlib inline

price=listings['price']

sns.swarmplot(y=price.sample(300));
# Most of the prices listed are well below $500, consider only use price data below $500

listings_new=listings.query('price<500')
# Explorer the relationship between price and neighbourhood

import matplotlib.pyplot as plt

sns.boxplot(x='neighbourhood_cleansed',y='price',data=listings_new)

ax=plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

plt.show()
# Explorer the relationship between price and property type

sns.boxplot(x='property_type',y='price',data=listings_new)

ax=plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

plt.show()
price_property=listings_new[['price','property_type']]

price_property_mean=price_property.groupby('property_type').mean().round(2)

price_property_mean # check the mean listing price for each room type
# Explorer the relationship between price and number of bedrooms & bathrooms

room_price=listings_new[['bathrooms','bedrooms','price']]
import numpy as np

temp=pd.pivot_table(room_price, index=['bedrooms'],columns=['bathrooms'],aggfunc=[np.mean]).round(2)
temp
sns.heatmap(temp)
# Explorer the relationship between number of reviews and price

reviews=listings_new[['price','number_of_reviews']]

sns.jointplot('price','number_of_reviews',data=reviews)