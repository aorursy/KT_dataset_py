# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# create a helper object for our bigquery dataset
dataopenaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# print a list of all the tables in this dataset
dataopenaq.list_tables()
dataopenaq.table_schema("global_air_quality")
dataopenaq.head("global_air_quality")
# query to select all the items from the "city" column where the "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

# the query_to_pandas_safe will only return a result if it's less than 1GB, this is important as I only get 5TB of data per month for free with Kaggle
us_cities = dataopenaq.query_to_pandas_safe(query)

# What five cities have the most measurements taken there? THIS IS USED TO PRINT WHAT IS NOW A NEW DATAFRAME THANKS TO THE QUERY ABOVE
us_cities.city.value_counts().head()