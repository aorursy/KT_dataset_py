# offical google cloud library is recommended

from google.cloud import bigquery

import pandas as pd
# initiate bigquery client

bq = bigquery.Client()
# let's start with first table in the dataset

query = """

SELECT

    *

FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`

LIMIT 10

"""

# NOTE: Don't use SELECT * in your queries.
# make the query and get the result to a dataframe

result = bq.query(query).to_dataframe()
# let's see what this data contains

result
keyed_query = """

SELECT

    device.browser as browser,

    SUM ( totals.visits ) AS total_visits

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`

GROUP BY

    device.browser

ORDER BY

    total_visits DESC

"""

keyed_df = bq.query(keyed_query).to_dataframe()

keyed_df
unnested_query = """

SELECT

    fullVisitorId,

    visitId,

    visitNumber,

    hits.hitNumber AS hitNumber,

    hits.page.pagePath AS pagePath

FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`, UNNEST(hits) as hits

WHERE

    hits.type="PAGE"

ORDER BY

    fullVisitorId,

    visitId,

    visitNumber,

    hitNumber

"""

unnested_query_df = bq.query(unnested_query).to_dataframe()

unnested_query_df
parameterized_query = """

SELECT

    device.browser as browser,

    SUM ( totals.visits ) AS total_visits

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

WHERE

    _TABLE_SUFFIX BETWEEN '%s' AND '%s'

GROUP BY

    device.browser

ORDER BY

    total_visits DESC

"""

start_date = '20170101'

end_date = '20171231'

parameterized_query_df = bq.query((parameterized_query % (start_date, end_date))).to_dataframe()

parameterized_query_df
sub_query = """

SELECT

    ( SUM(total_transactionrevenue_per_user) / SUM(total_visits_per_user) ) AS avg_revenue_by_user_per_visit

FROM (

    SELECT

        fullVisitorId,

        SUM( totals.visits ) AS total_visits_per_user,

        SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_user

    FROM

        `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE

        _TABLE_SUFFIX BETWEEN '20170701' AND '20170731' AND totals.visits > 0

        AND totals.transactions >= 1 AND totals.transactionRevenue IS NOT NULL

    GROUP BY

        fullVisitorId)

"""

sub_query_df = bq.query(sub_query).to_dataframe()

sub_query_df
# NOTE: We are not using any existing data source in this example query, we are generating our data source and then using the result as data source.

udf_query = """

CREATE TEMPORARY FUNCTION multiplyInputs(x FLOAT64, y FLOAT64)

RETURNS FLOAT64

LANGUAGE js AS '''

  return x*y;

''';



CREATE TEMPORARY FUNCTION divideByTwo(x FLOAT64)

RETURNS FLOAT64

LANGUAGE js AS '''

  return x/2;

''';



WITH numbers AS

  (SELECT 1 AS x, 5 as y

  UNION ALL

  SELECT 2 AS x, 10 as y

  UNION ALL

  SELECT 3 as x, 15 as y)



SELECT x,

  y,

  multiplyInputs(divideByTwo(x), divideByTwo(y)) as half_product

FROM numbers;

"""

udf_df = bq.query(udf_query).to_dataframe()

udf_df