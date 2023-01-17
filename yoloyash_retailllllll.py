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
import numpy as np
import pandas as pd

import datetime
df = pd.read_excel('/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')
df.head()
df.Country.value_counts().head(15)
# looking at the number of purchases per Country

df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year

dt_df = df.groupby(['Year', 'Month']).sum()
total = dt_df.sort_values('Amount')
total
df.UnitPrice.describe()
# some values are negative, which logically means that there are 
# some refund transactions
df = df[df.Country=='United Kingdom']
df.head(5)
# making a separate attribute based on total amount
df['Amount'] = df.Quantity * df.UnitPrice
df.head()
df = df[~(df['Amount']<0)]
df.head()
df = df[~(df.CustomerID.isnull())]
# df.head()
# remove all purchases in which the Customer ID is missing
# RFM - Recency, Frequency and Monetary Value

reference_date = df.InvoiceDate.max()
reference_date = reference_date + datetime.timedelta(days=1)
df['days_since_last_purch'] = reference_date - df.InvoiceDate
df['days_since_last_purch_num'] = df['days_since_last_purch'].astype('timedelta64[D]')
df['days_since_last_purch_num'].head()
# making a history of customer's last transactions

customer_history = df.groupby('CustomerID').min