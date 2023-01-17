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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
df.head(10)
df = df.set_index('Invoice ID')
df = df.drop('Tax 5%', axis = 1)
df = df.drop('gross margin percentage', axis = 1)
df
df = df.drop('Branch', axis = 1)
df['City'].value_counts().sort_values(ascending = False)
df['Product line'].value_counts().sort_values(ascending = False)
df_revenue_by_city = df.groupby('City').agg([np.sum, np.median])
df_revenue_by_city[['gross income']].T
df_revenue_by_city[['gross income', 'Unit price', 'Quantity']]
## Now onwards looking at revenue data only for Naypyitaw
df_revenue_by_product = df[df['City']== "Naypyitaw"].groupby('Product line').agg([sum, np.median])
df_revenue_by_product.sort_values(by = ('Quantity', 'sum'), ascending = False)
df_revenue_by_product.sort_values(by = ('gross income', 'sum'), ascending = False)
df_revenue_by_product.sort_values(by = ('Rating', 'median'), ascending = False)
df_revenue_by_gender = df.groupby('Gender').agg([np.median, sum])
df_revenue_by_gender.T
df_revenue_by_mem = df.groupby('Customer type').agg([np.median, sum])
df_revenue_by_mem.T
df[df['City'] == "Naypyitaw"].groupby(['Customer type','Gender']).size()
df[df['City'] == "Naypyitaw"].groupby(['Gender', 'Customer type', 'Product line']).size().sort_values(ascending=False)
df_revenue_by_pay = df[df['City'] == "Naypyitaw"].groupby('Payment')
df_revenue_by_pay.size().sort_values(ascending = False)
df[df['City'] == "Naypyitaw"].groupby(['Gender', 'Customer type', 'Payment']).size().sort_values(ascending=False)