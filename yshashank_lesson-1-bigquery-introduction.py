# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                        dataset_name = "hacker_news")
#What are the tables in this dataset?
hacker_news.list_tables()
#Print schema information of any particular table among those listed above
hacker_news.table_schema("full")
#Print the first 5 rows of the data set to see if it matches the Schema description
hacker_news.head("full")
#How to select particular column ans rows?
hacker_news.head("full", selected_columns="by", num_rows=10)
#How to check the size of a Big Query?
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)
#Run a query only if it is less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
#Repeat the above query without a max limit
job_post_scores =hacker_news.query_to_pandas_safe(query)
#Job_post_scores is the data frame to work with
job_post_scores.score.mean()
#A quick walkthrough of queries covered
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")