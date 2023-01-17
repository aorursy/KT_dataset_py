import pandas as pd
from google.cloud import bigquery
import bq_helper
OpenAQ=bq_helper.BigQueryHelper('bigquery-public-data','openaq')
OpenAQ.list_tables()
OpenAQ.head('global_air_quality')
OpenAQ.table_schema('global_air_quality')
QUERY1 = """
        SELECT distinct(country)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """
print('QUERY1 size estimate= {0:.3f} GB.'.format(OpenAQ.estimate_query_size(QUERY1)))
query1=OpenAQ.query_to_pandas_safe(QUERY1)
query1.head(10)
QUERY2 = """
        SELECT distinct(pollutant)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """

print('QUERY2 size estimate= {0:.3f} GB.'.format(OpenAQ.estimate_query_size(QUERY2)))
query2=OpenAQ.query_to_pandas_safe(QUERY2)
query2.head(10)
# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 1000
        """
print('QUERY size estimate = {0:.3f} GB.'.format(OpenAQ.estimate_query_size(QUERY)))
df = OpenAQ.query_to_pandas_safe(QUERY)
df.head(10)
client = bigquery.Client()
query_job = client.query(QUERY)
rows = list(query_job.result(timeout=30))
for row in rows[:3]:
    print(row)
type(rows[0])
list(rows[0].keys())
list(rows[0].values())
df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df.head(3)
df.info()
df['value'].plot()