%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import tarfile

import urllib

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('darkgrid')
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"

# HOUSING_PATH = os.path.join("datasets", "housing")

# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"



def fetch_housing_data(housing_url=HOUSING_URL):

#     os.makedirs(housing_path, exist_ok=True)

#     tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url)

    housing_tgz = tarfile.open('housing.tgz')

    housing_tgz.extractall('housing.tgz')

    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)

df = load_housing_data()
df.head()
df.describe()
df.info()
df['bedrooms'] = round(df['total_bedrooms'] / df['households'])

df['rooms'] = round(df['total_rooms'] / df['households'])

df['bedrooms'].fillna(np.median, inplace=True, axis=0)

df['total_bedrooms'].fillna(np.median, inplace=True, axis=0)
df
df['total_bedrooms'] = pd.to_numeric(df['total_bedrooms'], errors='coerce')

df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['price'] = df['median_house_value']

df.drop('median_house_value', axis=1,inplace=True) 
df['median_income'] = df['median_income'] * 10000
df.head()
sns.relplot(x='median_income', y='price', data=df, height=8, )

plt.show()
sns.relplot(x='median_income', y='price', data=df, 

            hue='housing_median_age', height=8, palette='jet')

plt.show()
sns.relplot(x='latitude', y='longitude', data=df, 

           hue='population', palette='jet', height=8)

plt.show()
sns.relplot(x='latitude', y='longitude', data=df, 

            hue='price', height=8, palette='jet')

plt.xlabel('latitude',fontsize=16)

plt.ylabel('longitude', fontsize=16)

plt.show()
sns.relplot(x='latitude',y='longitude', data=df,

           hue='price' ,size='median_income', palette='jet', height=10)

plt.show()
df
sns.relplot('rooms', 'price', data=df, 

            kind='line', height=6, sort=True)

plt.show()
rooms = df[df['rooms'] <= 20]

df.pivot_table(rooms, 'rooms').iloc[:20,[7]]
sns.relplot('bedrooms', 'price', data=df,

           height=6, kind='line', sort=True)

plt.show()
rooms = df[df['bedrooms'] <= 20]

df.pivot_table(rooms, 'bedrooms').iloc[:12, [6]]
sns.relplot('ocean_proximity', 'population', data=df, 

            kind='line', sort=True,

            height=6, palette='jet')

plt.show()
sns.relplot('ocean_proximity', 'median_income', data=df,

           kind='line', sort=True, height=6, palette='jet')

plt.show()
sns.relplot('ocean_proximity', 'price', data=df,

           kind='line', sort=True, height=6, palette='jet')

plt.show()
sns.relplot('ocean_proximity', 'rooms', data=df,

           kind='line', sort=True, height=6, palette='jet')

plt.show()
sns.relplot('ocean_proximity', 'housing_median_age', data=df,

           kind='line', sort=True, height=6, palette='jet')

plt.show()
sns.relplot('ocean_proximity', 'bedrooms', data=df,

           kind='line', sort=True, height=6, palette='jet')

plt.show()