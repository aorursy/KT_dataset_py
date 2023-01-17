import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



%matplotlib inline
df_train = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')

df_test = pd.read_csv('../input/used-cars-price-prediction/test-data.csv')
df_train_orig = df_train.copy()

df_test_orig = df_test.copy()
print("Skew ", df_train['Price'].skew())

print("kurt ", df_train['Price'].kurt())
#A trial to check log of target label to avoid skew & kurt

df_test1 = np.log1p(df_train['Price'].values)
df_test1 = df_test1.reshape(-1,1)

df_test1 = pd.DataFrame(df_test1, columns=['PriceNew'])
print("Skew ", df_test1['PriceNew'].skew())

print("kurt ", df_test1['PriceNew'].kurt())
df_train.head()
df_test.sample(5)
print(df_train.shape)

print(df_test.shape)
df_test.info()
df_train.describe()
miss_percent = (df_train.isnull().sum() / len(df_train)) * 100

missing = pd.DataFrame({"percent":miss_percent, 'count':df_train.isnull().sum()}).sort_values(by="percent", ascending=False)

missing.loc[missing['percent'] > 0]
miss_percent = (df_test.isnull().sum() / len(df_test)) * 100

missing = pd.DataFrame({"percent":miss_percent, 'count':df_test.isnull().sum()}).sort_values(by="percent", ascending=False)

missing.loc[missing['percent'] > 0]
df_train.drop(df_train.columns[0], axis=1, inplace=True)

df_test.drop(df_test.columns[0], axis=1, inplace=True)
df_train['brand_name'] = df_train['Name'].apply(lambda x: str(x).split(" ")[0])

df_test['brand_name'] = df_test['Name'].apply(lambda x: str(x).split(" ")[0])
df_train.drop(columns=["Name"], axis=1, inplace=True)

df_test.drop(columns=["Name"], axis=1, inplace=True)
#df_train.loc[df_train['brand_name'] == 'Maruti']['Seats'].mode()[0]

def fill_na_with_mode(ds, brandname):

  fill_value = ds.loc[ds['brand_name'] == brandname]['Seats'].mode()[0]

  condit = ((ds['brand_name'] == brandname) & (ds['Seats'].isnull()))

  ds.loc[condit, 'Seats'] = ds.loc[condit, 'Seats'].fillna(fill_value)
car_brand = ['Maruti','Hyundai','BMW','Fiat','Land','Ford','Toyota','Honda','Skoda','Mahindra']

for c in car_brand:

  fill_na_with_mode(df_train, c)

  fill_na_with_mode(df_test, c)
import re



df_train['Mileage_upd'] = df_train['Mileage'].apply(lambda x: re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', r'\1', str(x)))

df_train['Engine_upd'] = df_train['Engine'].apply(lambda x: re.sub(r'(\d+)\s(CC)', r'\1', str(x)))

df_train['Power_upd'] = df_train['Power'].apply(lambda x: re.sub(r'(\d+\.?\d+?)\s(bhp)', r'\1', str(x)))



df_test['Mileage_upd'] = df_test['Mileage'].apply(lambda x: re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', r'\1', str(x)))

df_test['Engine_upd'] = df_test['Engine'].apply(lambda x: re.sub(r'(\d+)\s(CC)', r'\1', str(x)))

df_test['Power_upd'] = df_test['Power'].apply(lambda x: re.sub(r'(\d+\.?\d+?)\s(bhp)', r'\1', str(x)))
df_train['Mileage_upd'] = pd.to_numeric(df_train['Mileage_upd'], errors='coerce')

df_train['Engine_upd'] = pd.to_numeric(df_train['Engine_upd'], errors='coerce')

df_train['Power_upd'] = pd.to_numeric(df_train['Power_upd'], errors='coerce')



df_test['Mileage_upd'] = pd.to_numeric(df_test['Mileage_upd'], errors='coerce')

df_test['Engine_upd'] = pd.to_numeric(df_test['Engine_upd'], errors='coerce')

df_test['Power_upd'] = pd.to_numeric(df_test['Power_upd'], errors='coerce')
df_train.drop(columns=['Mileage', 'Engine', 'Power'], inplace=True)

df_test.drop(columns=['Mileage', 'Engine', 'Power'], inplace=True)
df_train.drop(df_train[df_train['brand_name'] == 'Smart'].index, axis=0, inplace=True)

df_test.drop(df_test[df_test['brand_name'] == 'Hindustan'].index, axis=0, inplace=True)
#Function to replace na value with mode of that specific brand

def fill_na_with_mode(ds, brandname, colname):

  fill_value = ds.loc[ds['brand_name'] == brandname][colname].mode()[0]

  condit = ((ds['brand_name'] == brandname) & (ds[colname].isnull()))

  ds.loc[condit, colname] = ds.loc[condit, colname].fillna(fill_value)
miss_Mileage_col = df_train.loc[df_train['Mileage_upd'].isnull()]['brand_name'].unique()

miss_Engine_col = df_train.loc[df_train['Engine_upd'].isnull()]['brand_name'].unique()

miss_Power_col = df_train.loc[df_train['Power_upd'].isnull()]['brand_name'].unique()



for x in miss_Mileage_col:

  fill_na_with_mode(df_train, x, 'Mileage_upd')

for y in miss_Engine_col:

  fill_na_with_mode(df_train, y, 'Engine_upd')

for z in miss_Power_col:

  fill_na_with_mode(df_train, z, 'Power_upd')
miss_ts_Mileage_col = df_test.loc[df_test['Mileage_upd'].isnull()]['brand_name'].unique()

miss_ts_Engine_col = df_test.loc[df_test['Engine_upd'].isnull()]['brand_name'].unique()

miss_ts_Power_col = df_test.loc[df_test['Power_upd'].isnull()]['brand_name'].unique()



for x in miss_ts_Mileage_col:

  fill_na_with_mode(df_test, x, 'Mileage_upd')

for y in miss_ts_Engine_col:

  fill_na_with_mode(df_test, y, 'Engine_upd')

for z in miss_ts_Power_col:

  fill_na_with_mode(df_test, z, 'Power_upd')
zero_mileage_col = df_train.loc[df_train['Mileage_upd'] == 0.0]['brand_name'].unique()



for m in zero_mileage_col:

  fill_zero = df_train.loc[df_train['brand_name'] == m]['Mileage_upd'].mode()[0]

  m1 = ((df_train['brand_name'] == m) & (df_train['Mileage_upd'] == 0.0))

  df_train.loc[m1, 'Mileage_upd'] = fill_zero
zero_mileage_col2 = df_test.loc[df_test['Mileage_upd'] == 0.0]['brand_name'].unique()



for m in zero_mileage_col2:

  fill_zero = df_test.loc[df_test['brand_name'] == m]['Mileage_upd'].mode()[0]

  m1 = ((df_test['brand_name'] == m) & (df_test['Mileage_upd'] == 0.0))

  df_test.loc[m1, 'Mileage_upd'] = fill_zero
m1 = (df_train['Seats'] == 0.0)

df_train.loc[m1, 'Seats'] = 5.0
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.distplot(df_train['Price'])



plt.subplot(1,2,2)

sns.boxplot(y=df_train['Price'])

plt.show()
fig = plt.figure(figsize=(20,18))

fig.subplots_adjust(hspace=0.2, wspace=0.2)

fig.add_subplot(2,2,1)

g1 = sns.countplot(x='brand_name', data=df_train)

loc,labels = plt.xticks()

g1.set_xticklabels(labels,rotation=90)

fig.add_subplot(2,2,2)

g2 = sns.countplot(x='Fuel_Type', data=df_train)

loc,labels = plt.xticks()

g2.set_xticklabels(labels,rotation=0)

fig.add_subplot(2,2,3)

g3 = sns.countplot(x='Transmission', data=df_train)

loc,labels = plt.xticks()

g3.set_xticklabels(labels,rotation=0)

fig.add_subplot(2,2,4)

g4 = sns.countplot(x='Owner_Type', data=df_train)

loc,labels = plt.xticks()

g4.set_xticklabels(labels,rotation=0)

plt.show()
fig = plt.figure(figsize=(15,15))

fig.subplots_adjust(hspace=0.2, wspace=0.2)

ax1 = fig.add_subplot(2,2,1)

plt.xlim([0, 100000])

p1 = sns.scatterplot(x="Kilometers_Driven", y="Price", data=df_train)

loc, labels = plt.xticks()

ax1.set_xlabel('Kilometer')



ax2 = fig.add_subplot(2,2,2)

#plt.xlim([0, 100000])

p2 = sns.scatterplot(x="Mileage_upd", y="Price", data=df_train)

loc, labels = plt.xticks()

ax2.set_xlabel('Mileage')



ax3 = fig.add_subplot(2,2,3)

#plt.xlim([0, 100000])

p3 = sns.scatterplot(x="Engine_upd", y="Price", data=df_train)

loc, labels = plt.xticks()

ax3.set_xlabel('Engine')



ax4 = fig.add_subplot(2,2,4)

#plt.xlim([0, 100000])

p4 = sns.scatterplot(x="Power_upd", y="Price", data=df_train)

loc, labels = plt.xticks()

ax4.set_xlabel('Power')



plt.show()
fig = plt.figure(figsize=(18,5))

fig.subplots_adjust(hspace=0.3, wspace=0.3)



ax1 = fig.add_subplot(1,2,1)

sns.scatterplot(x='Price', y="Year", data=df_train)

ax1.set_xlabel('Price')

ax1.set_ylabel('Year')

ax1.set_title('Year vs Price')



ax2 = fig.add_subplot(1,2,2)

sns.scatterplot(x='Price', y='Kilometers_Driven', data=df_train)

ax2.set_ylabel('kilometer')

ax2.set_xlabel('Price')

ax2.set_title('Kilometer vs Price')

plt.show()
df_train.drop(df_train[df_train['Kilometers_Driven'] >= 6500000].index, axis=0, inplace=True)
df_vis_1 = pd.DataFrame(df_train.groupby('brand_name')['Price'].mean())

df_vis_1.plot.bar()

plt.show()
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(1,2,1)

sns.boxplot(x='Owner_Type', y='Price', data=df_train)

ax1.set_title('Owner vs Price')



ax2 = fig.add_subplot(1,2,2)

sns.boxplot(x='brand_name', y='Price', data=df_train)

loc,labels = plt.xticks()

ax2.set_xticklabels(labels, rotation=90)

ax2.set_title('Brand vs Price')

plt.show()
fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(1,3,1)

sns.boxplot(x='Seats', y='Price', data=df_train)

ax1.set_title('Seats vs Price')



ax2 = fig.add_subplot(1,3,2)

sns.boxplot(x='Transmission', y='Price', data=df_train)

ax2.set_title('Transmission vs Price')



ax3 = fig.add_subplot(1,3,3)

sns.boxplot(x='Fuel_Type', y='Price', data=df_train)

ax3.set_title('Fuel vs Price')



plt.show()
import datetime

now = datetime.datetime.now()

df_train['Year_upd'] = df_train['Year'].apply(lambda x : now.year - x)

df_test['Year_upd'] = df_test['Year'].apply(lambda x : now.year - x)
df_train.drop(columns=['Year'], axis=1, inplace=True)

df_test.drop(columns=['Year'], axis=1, inplace=True)
df_train.drop(columns=['New_Price'], axis=1, inplace=True)

df_test.drop(columns=['New_Price'], axis=1, inplace=True)
df_train.drop(columns=['Location'], axis=1, inplace=True)

df_test.drop(columns=['Location'], axis=1, inplace=True)
df_train_norm = pd.get_dummies(df_train, drop_first=True)

df_test_norm = pd.get_dummies(df_test, drop_first=True)
df_train_norm['Price_upd'] = np.log1p(df_train_norm['Price'].values)
df_train_norm.drop(columns=['Price'], axis=1, inplace=True)
df_train_X = df_train_norm.drop(columns=['Price_upd'], axis=1)

df_train_y = df_train_norm[['Price_upd']]
df_train_X = (df_train_X - df_train_X.mean())/df_train_X.std()

df_test_norm = (df_test_norm - df_test_norm.mean())/df_test_norm.std()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



lm = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_train_y, test_size=0.22, random_state=1)

reg = lm.fit(X_train, y_train)
y_predict = reg.predict(X_test)

y_predict
from sklearn.metrics import r2_score



r2_score(y_predict, y_test)
reg.score(X_test,y_test)