# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
hacker_news.head("full" )
# describe table. look at data types.
hacker_news.table_schema("full")
# select and estimate the size
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """
hacker_news.estimate_query_size(query)
job_post_score = hacker_news.query_to_pandas_safe(query)
job_post_score.tail(n=20)
open_aq = bq_helper.BigQueryHelper("bigquery-public-data", "openaq")
open_aq.list_tables()
open_aq.table_schema('global_air_quality')
open_aq.head("global_air_quality")
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query = '''SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
'''
nppm = open_aq.query_to_pandas_safe(query)
nppm.country.unique()
query = '''SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
'''
zero_poll = open_aq.query_to_pandas_safe(query)
zero_poll.pollutant.unique()