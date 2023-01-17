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
import pandas as pd 
from google.cloud import bigquery 
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data","openaq")
bq_assistant.list_tables()
bq_assistant.head("global_air_quality", num_rows = 3 )
bq_assistant.table_schema("global_air_quality")
#### simple query to run the table using to_dataframe()
Query = """
select location, city, country, value, timestamp 
from `bigquery-public-data.openaq.global_air_quality`
where pollutant = "pm10" and timestamp>"2017-04-01"
order by value desc 
limit 1000
"""

client = bigquery.Client()
query_job = client.query(Query)
df = query_job.to_dataframe()
df.head(10)
client = bigquery.Client()
query_job = client.query(Query)
rows = list(query_job.result(timeout=30))
for row in rows[:10]:
    print(row)
type(rows[0])
list(rows[0].keys())
list(rows[0].values())
df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

                  
                  
df.head(3)
df.info()
df['value'].plot()