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
data = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding='ISO-8859-1', dtype={'CustomerID': str})
data.head(10)
data.dtypes
data.shape
missing_count = data.isna().sum().rename('count')

missing_description = missing_count.to_frame()

missing_description['percentage'] = missing_description['count'] / data.shape[0] * 100

missing_description



# 0.26% data in Description is missing data

# 24.9% data in CustomerID is missing data
na_desc = data[data['Description'].isna()]

na_desc.head(10)
na_desc['CustomerID'].value_counts()



# all the values in column ``CustomerID`` are also missing data, too.
na_desc.describe()



# values in column ``UnitPrice` are 0.
(desc_data['Description'].str.len() == 0).sum()

# no empty string
na_customerid = data[data['CustomerID'].isna()]
na_desc['CustomerID'].isna().sum()

# those customers may be the unregistered customers.
data.dropna(inplace=True)
data.isna().sum()
data['InvoiceNo'].map(lambda x: x[0]).value_counts()



# 8,905 rows in column ``InvoiceNo`` which strings starts with 'C'
data['IsCancelled'] = data['InvoiceNo'].str.startswith('C')

data['IsCancelled'].value_counts() / data.shape[0] * 100
data[data['IsCancelled'] == True].describe()



# values in column ``Quantity`` are all less than 0.

# data in ``InvoiceNo`` which strings start with C represent the cancelled orders.
data = data[data['IsCancelled'] == False].copy()

data.drop(columns=['IsCancelled'], inplace=True)
data.shape
data['InvoiceDate'].min(), data['InvoiceDate'].max()
data['CustomerID'].nunique()
data['InvoiceNo'].nunique()
country_count = (

    data[['InvoiceNo', 'Country']].groupby('Country')['InvoiceNo'].nunique()

    .sort_values(ascending=False).to_frame('count')

)

country_count['percentage'] = country_count['count'] / country_count['count'].sum() * 100
country_count.head(10)



# 89.8% of orders are from United Kingdom
country_count['percentage'].head(10).plot.bar()
stockcode_desc = data.groupby('StockCode')['Description'].unique().to_frame()

stockcode_desc['n_unique'] = stockcode_desc['Description'].map(len)

stockcode_desc['n_unique'].value_counts()



# there is no one-to-one relationship between ``StockCode`` and ``Description``
stockcode_desc[stockcode_desc['n_unique'] != 1]
data["InvoiceDate"] = pd.to_datetime(data['InvoiceDate'])
data['InvoiceDate'].describe()
data['InvoiceDate'].max() - data['InvoiceDate'].min()
data['InvoiceDate_year'] = data['InvoiceDate'].dt.year

data['InvoiceDate_month'] = data['InvoiceDate'].dt.month

data['InvoiceDate_hour'] = data['InvoiceDate'].dt.hour

data['InvoiceDate_year_month'] = (

    data['InvoiceDate'].dt.year.astype('str') + '_' +

    data['InvoiceDate'].dt.month.astype('str').str.zfill(2)

)

data['InvoiceDate_dayofweek'] = data['InvoiceDate'].dt.dayofweek
data.groupby('InvoiceDate_year_month')['InvoiceNo'].nunique().plot.bar()



# sales in November get the higest sales, but decrease in December. 

# 可以解讀為，因應節慶的到來，會提早在11月下單購買。導致節慶當月銷量下降。
data.groupby('InvoiceDate_dayofweek')['InvoiceNo'].nunique().plot.bar()



# there is no orders on Saturday (5).
invoiceno_hour = data.groupby('InvoiceDate_hour')['InvoiceNo'].nunique()

invoiceno_hour.index

invoiceno_hour.plot.bar()



# there is higest number of orders at 12:00pm
data['Description'].nunique()
desc_counts = data['Description'].value_counts().rename('').to_frame()

desc_counts = desc_counts['percentage'] = missing_description['count'] / data.shape[0] * 100

missing_description
data['Description'].value_counts().head(20).plot.bar()
def compute_recency(data, dt):

    assert isinstance(data, pd.DataFrame)

    assert isinstance(dt, pd.Timestamp)

    return (dt - data.groupby('CustomerID')['InvoiceDate'].max()).dt.days





def compute_frequency(data):

    assert isinstance(data, pd.DataFrame)

    data = data[['CustomerID', 'InvoiceDate']].drop_duplicates()

    output = (

        data.groupby('CustomerID')

        .apply(lambda subdf: subdf['InvoiceDate'].sort_values().diff().mean())

        .fillna(pd.Timedelta(seconds=0))

    )

    return output





def compute_monetary(data):

    assert isinstance(data, pd.DataFrame)

    assert (data['total_price'] >= 0).all()

    return data.groupby('CustomerID')['total_price'].mean()





def compute_rfm(data):

    assert isinstance(data, pd.DataFrame)

    dt = data['InvoiceDate'].max()

    r_series = compute_recency(data, dt)

    f_series = compute_frequency(data)

    m_series = compute_monetary(data)

    output = pd.concat([

            r_series.rename('recency'),

            f_series.rename('frequency'),

            m_series.rename('monetary'),

            pd.qcut(x=-r_series, q=4, labels=False).rename('r_score') + 1,

            pd.qcut(x=-f_series, q=4, labels=False, duplicates='drop').rename('f_score') + 1,

            pd.qcut(x=m_series, q=4, labels=False).rename('m_score') + 1

        ], axis=1)

    output.reset_index(drop=False, inplace=True)

    output['rfm_score'] = output['r_score'] + output['f_score'] + output['m_score']

    return output





data['total_price'] = data['UnitPrice'] * data['Quantity']

customer_rfm = compute_rfm(data=data)
customer_rfm.head(10)
customer_rfm.groupby('rfm_score')['CustomerID'].nunique()
customer_rfm.groupby('rfm_score')['recency'].describe()
customer_rfm.groupby('rfm_score')['frequency'].describe()
customer_rfm.groupby('rfm_score')['monetary'].describe()