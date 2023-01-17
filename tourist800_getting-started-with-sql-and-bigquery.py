# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import bq_helper
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "hacker_news")
#open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")

# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
#open_aq.list_tables()

# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")
hacker_news.head("full")
hacker_news.head("full",selected_columns = "by",num_rows = 10)
# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """ SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """


# check how big this query will be
hacker_news.estimate_query_size(query)
# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)
# average score for job posts
job_post_scores.score.mean()

