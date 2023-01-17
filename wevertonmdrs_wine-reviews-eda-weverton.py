# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
winereviews_filepath = "../input/wine-reviews/winemag-data-130k-v2.csv"

wine_reviews = pd.read_csv(winereviews_filepath, index_col = 0)



wine_reviews.head() # First 5 rows
def percentage(row,total):

    

    rate = float('{0:.4f}'.format((row.quantity/(total))*100))

    return rate

    

wine_per_country = wine_reviews.groupby('country').description.agg([len]).sort_values(by='len', ascending=False)

wine_per_country = wine_per_country.rename_axis('', axis = 'rows').rename_axis('country', axis = 'columns').rename(columns = {'len' : 'quantity'})



total_wine_country = wine_per_country.quantity.sum()

wine_per_country['percentage %'] = wine_per_country.apply(lambda row : percentage(row,total_wine_country), axis = 'columns')



wine_per_country.head()

plt.figure(figsize = (14,8))

plt.title('Top 10 countries with the most wine reviews')





sns.barplot(x = wine_per_country.iloc[0:10].index, y = wine_per_country.iloc[0:10,1])





plt.ylabel('Percentage %')

plt.xlabel('Countries')

plt.legend()
points_price_info = wine_reviews.describe()

points_price_info
plt.figure(figsize = (12,8))



plt.title('Distribution of wine evaluation')



sns.distplot(a = wine_reviews['points'], kde = False)
wine_per_country['mean points'] =  wine_reviews.groupby(['country']).points.mean().sort_values(ascending = False)

wine_per_country.head()

french_wine = wine_reviews.copy()



def french_wine_dataset(row):

    

    if row.country == 'France':

        return 'France'

    else:

        return 'Not_France'



french_wine['country'] = french_wine.apply(lambda row : french_wine_dataset(row), axis = 'columns')

french_wine.head(10)
plt.figure(figsize = (8,10))



sns.set_style('darkgrid')

sns.lmplot(x = 'points', y = 'price', hue = 'country', data = french_wine)
sns.lmplot(y = 'points', x = 'price', hue = 'country', lowess = True, data = french_wine)

# sns.regplot(x=french_wine['price'], y=french_wine['points'],hue = french_wine['country'],lowess = True)
is_french = french_wine.loc[french_wine.country == 'France']

is_french.head()
CB_info = is_french.describe() 

CB_info
def CB_points_price(row):

    if row.points == 'NaN' or row.price == 'NaN':

        return 'NaN'

    return row.points/row.price



def CB_status(row):



    if row.CB_points_price <= 2.16: 

        return 'Low'

    elif row.CB_points_price <= 3.24: #2*(2.1596) + (2.1596/2)

        return 'Medium'

    elif row.CB_points_price <= 5.4: # (2*(2.1596))+(2.1596/2) 

        return 'High' 

    else:

        return 'Very High'



is_french['CB_points_price'] = is_french.apply(lambda row : CB_points_price(row), axis = 'columns')

is_french['CB_status'] = is_french.apply(lambda row : CB_status(row), axis = 'columns')

is_french
plt.figure(figsize = (14,10))



sns.scatterplot(y = is_french['points'], x = is_french['price'], hue = is_french['CB_status'])
wine_per_province = is_french.groupby('province').CB_points_price.mean().sort_values(ascending = False)

wine_per_province
plt.figure(figsize = (14,6))



plt.title('Top 10 Provinces with best cost benefit')

plt.ylabel('Province')

plt.xlabel('Cost benefit ratio')

plt.legend()



sns.barplot(y = wine_per_province.index, x = wine_per_province.iloc[:])
province_points_price = is_french.loc[(is_french.points >= 95)&(is_french.price < 40)]



province_points_price = province_points_price.groupby('province').description.agg([len]).sort_values(by = 'len', ascending = False)

province_points_price = province_points_price.rename_axis('', axis = 'rows').rename_axis('province', axis = 'columns').rename(columns = {'len' : 'quantity'})



total_quantity_province = province_points_price.quantity.sum()



province_points_price

plt.figure(figsize = (12,8))



plt.title('French province whose wines have evaluation >= 95 and their prices are < 40')





sns.barplot(x = province_points_price.iloc[:].index, y = ((province_points_price.iloc[: ,0])/total_quantity_province)*100)

plt.xlabel('Province')

plt.ylabel('Percentage %')

plt.legend()

best_french_wines = is_french.loc[is_french.points == 100]

best_french_wines
best_french_wine = best_french_wines.loc[best_french_wines.price.idxmin()]

best_french_wine