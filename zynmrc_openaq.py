# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery



client = bigquery.Client()

open_aq_data = client.dataset('openaq', project='bigquery-public-data')

type(open_aq_data)
[x_axis.table_id for x_axis in client.list_tables(open_aq_data)]
data_gaq = client.get_table(open_aq_data.table('global_air_quality'))

type(data_gaq)
sql = """SELECT * FROM `bigquery-public-data.openaq.global_air_quality`"""

df = client.query(sql).to_dataframe()

type(df)
df.head(n=10)
df.tail(n=10)
array_ts = df['timestamp']

print("Earliest ts is " + str(array_ts.min()))

print("Latest ts is " + str(array_ts.max()))
df_ger = df[df['country'] == 'DE']

df_ger.head(n=10)
print('Earliest ts in Germany ' + str(df_ger['timestamp'].min()))

print('Latest ts in Germany ' + str(df_ger['timestamp'].max()))