import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import bq_helper
#Here's how we can use the BQHelper library to pull datasets/tables from BigQuery
ga_bq_train = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", 
                                       dataset_name = "ga_train_set")
ga_bq_test = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", 
                                       dataset_name = "ga_test_set")
ga_bq_test.list_tables()
#Take a look at the test set schema
ga_bq_test.table_schema('ga_sessions_20170802')
#Here's an example of how we might query the BQ train set tables to summarize total transaction revenue per user, where fullVisitorId is unique per user.
all_train_summary_query = """SELECT  fullVisitorId, coalesce(SUM( totals.transactionRevenue ),0) AS total_transactionrevenue_per_user
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
  GROUP BY fullVisitorId
"""
all_train_summary = ga_bq_train.query_to_pandas_safe(all_train_summary_query)
#Take a look at some basic analytics of the transaction revenue pulled from our query
all_train_summary.describe()
#How many of the users spent any money?
len(all_train_summary[all_train_summary.total_transactionrevenue_per_user > 0])
#Let's take a look at the average transaction revenue for each user who spent money.

avg_transrev_query = """SELECT
( SUM(total_transactionrevenue_per_visitorid) / SUM(total_visits_per_visitorid) ) AS
avg_revenue_by_user_per_visit
FROM (
SELECT
fullVisitorId,
SUM( totals.visits ) AS total_visits_per_visitorid,
SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_visitorid
FROM
`kaggle-public-datasets.ga_train_set.ga_sessions_*`
WHERE
totals.visits > 0
AND totals.transactionRevenue IS NOT NULL
GROUP BY
fullVisitorId );
        """
avg_transrev = ga_bq_train.query_to_pandas_safe(avg_transrev_query, max_gb_scanned=10)
avg_transrev.head()
#Here we'll create another query to pull a few specific datapoints per user.
all_train_summary_query2 = """SELECT  fullVisitorId, 
SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_user, 
SUM(totals.pageviews) AS total_pagesviews_per_user,
SUM(totals.visits ) AS total_visits_per_user,
SUM(totals.timeOnSite ) AS total_timeonsite_per_user
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
  GROUP BY fullVisitorId
"""
all_train_summary2 = ga_bq_train.query_to_pandas_safe(all_train_summary_query2)
all_train_summary2.head()
#Now let's drop the data from that query into a dataframe.
#This is the start of how you might pull features that you've queried out of BQ to begin modelling.
full_df = ga_bq_train.query_to_pandas(all_train_summary_query2)
full_df.describe()