import bq_helper
from bq_helper import BigQueryHelper
import pandas as pd
google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="data:google_analytics_sample")
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
bq_assistant.list_tables()
bq_assistant.head("ga_sessions_20160801", num_rows=3)
tableSchema = bq_assistant.table_schema("ga_sessions_20160801")
tableSchema
# Side note: Pulling specific table values just like any other Pandas dataframe
tableSchema[tableSchema["name"].str.contains("transactions")]
query1 = """SELECT
device.browser,
SUM ( totals.transactions ) AS total_transactions
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
GROUP BY
device.browser
ORDER BY
total_transactions DESC;
        """
response1 = google_analytics.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
source,
total_visits,
total_no_of_bounces,
( ( total_no_of_bounces / total_visits ) * 100 ) AS bounce_rate
FROM (
SELECT
trafficSource.source AS source,
COUNT ( trafficSource.source ) AS total_visits,
SUM ( totals.bounces ) AS total_no_of_bounces
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
GROUP BY
source )
ORDER BY
total_visits DESC;
        """
response2 = google_analytics.query_to_pandas_safe(query2)
response2.head(10)
query3 = """
SELECT
( SUM(total_pageviews_per_user) / COUNT(users) ) AS
avg_pageviews_per_user
FROM (
SELECT
fullVisitorId AS users,
SUM(totals.pageviews) AS total_pageviews_per_user
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
totals.transactions >= 1
GROUP BY
users )
"""

response3 = google_analytics.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
query4 = """
SELECT
( SUM(total_pageviews_per_user) / COUNT(users) ) AS
avg_pageviews_per_user
FROM (
SELECT
fullVisitorId AS users,
SUM(totals.pageviews) AS total_pageviews_per_user
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
totals.transactions IS NULL
GROUP BY
users )
"""

response4 = google_analytics.query_to_pandas_safe(query4,max_gb_scanned=10)
response4.head(10)
query5 = """
SELECT
(SUM(total_transactions_per_user) / COUNT(users)) AS
avg_total_transactions_per_user
FROM (
SELECT
fullVisitorId as users,
SUM(totals.transactions) AS total_transactions_per_user
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
totals.transactions IS NOT NULL
GROUP BY
users )
"""

response5 = google_analytics.query_to_pandas_safe(query5,max_gb_scanned=10)
response5.head(10)
query6 = """SELECT
( SUM(total_transactionrevenue_per_user) / SUM(total_visits_per_user) ) AS
avg_revenue_by_user_per_visit
FROM (
SELECT
fullVisitorId,
SUM( totals.visits ) AS total_visits_per_user,
SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_user
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
totals.visits > 0
AND totals.transactions >= 1
AND totals.transactionRevenue IS NOT NULL
GROUP BY
fullVisitorId );
        """
response6 = google_analytics.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(10)
query7 = """SELECT
fullVisitorId,
visitId,
visitNumber,
hits.hitNumber AS hitNumber,
hits.page.pagePath AS pagePath
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`,
UNNEST(hits) as hits
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
hits.type="PAGE"
ORDER BY
fullVisitorId,
visitId,
visitNumber,
hitNumber;
        """
response7 = google_analytics.query_to_pandas_safe(query7, max_gb_scanned=10)
response7.head(10)