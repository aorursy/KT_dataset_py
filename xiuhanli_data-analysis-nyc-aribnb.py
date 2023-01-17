#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns
#acquring data

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
print('shape:',data.shape)

print('colunms:',data.columns.tolist)

data.info()
#change 'last review' column to type 'datetime'

data['last_review'] = pd.to_datetime(data['last_review'])
total = data.isnull().sum().sort_values(ascending = False)

percentage = (data.isnull().sum()/data.count()).sort_values(ascending = False)

missing_data = pd.concat([total, percentage],axis = 1, keys = ['Total', 'Percentage']).sort_values('Total',ascending = False)

print(missing_data)
print(data[data.number_of_reviews==0.0].shape)
data.drop(['host_name','name'], axis = 1, inplace = True)



data.reviews_per_month = data.reviews_per_month.fillna(0)

earliest = min(data.last_review)

data.last_review = data.last_review.fillna(earliest)

data['last_review'] = data['last_review'].apply(lambda x:x.toordinal() - earliest.toordinal())
total = data.isnull().sum().sort_values(ascending = False)

percentage = (data.isnull().sum()/data.count()).sort_values(ascending = False)

missing_data = pd.concat([total, percentage],axis = 1, keys = ['Total', 'Percentage']).sort_values('Total',ascending = False)

print(missing_data)
total = len(data.host_id.unique())

percentage = total/data.shape[0]

print(f'unique num of host_id: {total}')

print(f'percentage : {percentage}')
data.drop(['host_id','id'], axis = 1, inplace = True)
print(data.neighbourhood_group.value_counts())

sns.catplot(x = 'neighbourhood_group', kind = 'count', data = data)

plt.show()
print(data.room_type.value_counts())

sns.catplot(x= 'room_type', kind = 'count', data = data)

plt.show()
fig,axes = plt.subplots(1,3,figsize = (21,6))

sns.distplot(data.latitude, ax = axes[0])

sns.distplot(data.longitude, ax = axes[1])

sns.scatterplot(x = data.latitude, y = data.longitude, ax = axes[2])

plt.show()
fig, ax = plt.subplots()

sns.distplot(data.minimum_nights, kde = False, rug = True)

ax.set_title('Counts of minimum nights')

ax.set_xlabel('minimum nights')

ax.set_ylabel('Counts')

ax.grid(which = 'major', axis = 'both')

plt.show()
print(data.minimum_nights.describe(percentiles = [.25,.5,.75,.99]))
fig, ax = plt.subplots()

sns.distplot(np.log1p(data.minimum_nights), kde = False, rug = True)

ax.set_yscale('log')

ax.grid(which = 'major', axis = 'both')

plt.show()
data.minimum_nights= np.log1p(data.minimum_nights)
fig, axes = plt.subplots(1,2, figsize = (18,6))

sns.distplot(data.reviews_per_month, kde = False, rug = True, ax = axes[0])

axes[0].set_xlabel('reviews per month')

axes[0].set_ylabel('counts') 

axes[0].set_title('Counts of reviews permonth')

axes[0].grid(which = 'major', axis = 'both')

sns.distplot(np.log1p(data.reviews_per_month),kde = False, rug = True, ax = axes[1])

axes[1].set_xlabel('log:reviews permonth')

axes[1].set_ylabel('log')

axes[1].set_title('Counts of reviews permonth')

axes[1].grid(which = 'major', axis = 'both')

plt.show()
print(data.reviews_per_month.describe(percentiles= [.25,.5,.75,.99]))
fig, ax = plt.subplots()

sns.distplot(data.availability_365)

ax.set_xlabel('Availability 365')

ax.set_xlim(0,365)

plt.show()

fig, ax = plt.subplots()

sns.distplot(data.calculated_host_listings_count,kde = False)

ax.set_xlabel('calculated host listings count')

ax.set_yscale('log')

plt.show()
fig, axes = plt.subplots(1,3, figsize = (21,6))

sns.distplot(data.price, ax = axes[0])

axes[0].set_xlabel('price')

sns.distplot(np.log1p(data.price), ax = axes[1])

axes[1].set_xlabel('log(1+price)')

sm.qqplot(np.log1p(data.price), fit = True, line = '45',ax= axes[2])

plt.show()
data.price = np.log1p(data.price)
corrmatrix = data.corr()

fig, ax = plt.subplots()

sns.heatmap(corrmatrix, vmax = 0.7, annot=True,cmap="RdBu_r", linewidth = 0.2)

plt.show()
sns.pairplot(data.select_dtypes(exclude = ['object']),height = 3.5)

plt.show()
fig,axes = plt.subplots(2,2, figsize = (20,20))

axes[0,0].scatter(x = data.number_of_reviews, y=np.exp(data.price)-1,alpha=.5 )

axes[0,0].set_xlabel('number_of_reviews')

axes[0,0].set_ylabel('price')

axes[0,1].scatter(x = data.number_of_reviews, y = data.minimum_nights, alpha = .5 )

axes[0,1].set_xlabel('number_of_reviews')

axes[0,1].set_ylabel('minimum_nights')

axes[1,0].scatter(x = data.number_of_reviews, y = data.latitude, alpha = .5 )

axes[1,0].set_xlabel('number_of_reviews')

axes[1,0].set_ylabel('latitude')

axes[1,1].scatter(x = data.number_of_reviews, y = data.longitude, alpha = .5 )

axes[1,1].set_xlabel('number_of_reviews')

axes[1,1].set_ylabel('longitude')

plt.show()