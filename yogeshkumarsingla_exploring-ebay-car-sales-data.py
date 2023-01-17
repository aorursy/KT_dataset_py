# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

autos = pd.read_csv('../input/used-cars-database/autos.csv',encoding='Latin-1')
autos.info()

autos.head()
print(autos.columns)
column_names =  ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest',

       'vehicle_type', 'registration_year', 'gearbox', 'power_PS', 'model',

       'odometer', 'registration_month', 'fuel_type', 'brand',

       'unepaired_damage', 'ad_created', 'nr_of_pictures', 'postal_code',

       'last_seen']

autos.columns = column_names

autos.head()
autos.describe()
autos['price'] = autos['price'].astype(str).str.replace("$","")

autos['price'] = autos['price'].astype(str).str.replace(",","")

autos['price'] = autos['price'].astype(int)

autos['odometer'] = autos['odometer'].astype(str).str.replace("km","")

autos['odometer'] = autos['odometer'].astype(str).str.replace(",","")

autos['odometer'] = autos['odometer'].astype(int)

autos.rename({"odometer":"odometer_km"},axis=1,inplace=True)

autos.head()
autos.info()
print(autos['price'].unique().shape)

print(autos['odometer_km'].unique().shape)

print(autos.describe())
print(autos['price'].value_counts().sort_index(ascending=True).head())

print(autos['odometer_km'].value_counts().sort_index(ascending=True).head())

print(autos['price'].value_counts().sort_index(ascending=False).head())

print(autos['odometer_km'].value_counts().sort_index(ascending=False).head())
autos = autos.loc[autos['price'].between(1,99999998),:]
print(autos['price'].value_counts().sort_index(ascending=True).head())

print(autos['odometer_km'].value_counts().sort_index(ascending=True).head())

print(autos['price'].value_counts().sort_index(ascending=False).head())

print(autos['odometer_km'].value_counts().sort_index(ascending=False).head())
print(autos['date_crawled'].str[:10].value_counts(normalize=True, dropna=False).sort_index().head())

print(autos['ad_created'].str[:10].value_counts(normalize=True, dropna=False).sort_index().head())

print(autos['last_seen'].str[:10].value_counts(normalize=True, dropna=False).sort_index().head())

print(autos['registration_year'].describe())
autos = autos.loc[autos['registration_year'].between(1900,2016),:]
print(autos['registration_year'].describe())
brands = autos['brand'].value_counts().index[0:6]
brand_dict ={}

for b in brands:

    brand_dict[b] = autos.loc[autos['brand']==b,'price'].mean()

print(brand_dict)
brand_series = pd.Series(brand_dict)

brand_series
df = pd.DataFrame(brand_series,columns=['mean_price'])

df