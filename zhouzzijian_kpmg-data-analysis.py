# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
xls = pd.ExcelFile('/kaggle/input/KPMG_VI_New_raw_data_update_final.xlsx')

df = pd.read_excel(xls, "Transactions")
# clean column names

headers = df.iloc[0]

df  = pd.DataFrame(df.values[1:], columns=headers)



df.head(10)
df.describe()
# filter out not approved transactions

df = df[df['order_status'] == 'Approved']

df.head(10)
# brands

fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(ax=ax, x="brand", data=df)
# product class



fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['product_class'].value_counts(), labels=['high','medium','low'], autopct='%1.1f%%',startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
# order channel



fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['online_order'].value_counts(), labels=['True', 'False'], autopct='%1.1f%%',startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
# product size



fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['product_size'].value_counts(), labels=['large', 'medium', 'small'], autopct='%1.1f%%',startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
# distribution of list prices

sns.distplot(df.loc[:,'list_price'])
# distribution of standard costs

sns.distplot(df.loc[:,'standard_cost'])
# product class among the online shoppers

fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(ax=ax, x="product_class", data=df[df['online_order'] == True])
# product class among the offline shoppers

fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(ax=ax, x="product_class", data=df[df['online_order'] == False])
# product size among the online shoppers

fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(ax=ax, x="product_size", data=df[df['online_order'] == True])
# product size among the offline shoppers

fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(ax=ax, x="product_size", data=df[df['online_order'] == False])
# correlation between list prices and standard costs

fig, ax = plt.subplots(figsize=(12,7))

df.plot(ax=ax, x='list_price', y='standard_cost', style='o')
# correlation between product size and product class



fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(df['product_size'],hue=df['product_class'])
fig, ax = plt.subplots(figsize=(12,7))

sns.countplot(df['product_class'],hue=df['product_size'])