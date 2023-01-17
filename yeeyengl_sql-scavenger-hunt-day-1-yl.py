import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country = 'US'
           """
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query = """SELECT country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'
           """
countries_not_ppm = open_aq.query_to_pandas_safe(query)
countries_not_ppm.country.value_counts()
query = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
           """
pollutant_is_null = open_aq.query_to_pandas_safe(query)
pollutant_is_null.pollutant.value_counts()

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
US
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
