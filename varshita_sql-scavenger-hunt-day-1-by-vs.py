import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 1000
        """
# my trial query
#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
MY_QUERY = """ SELECT location, pollutant, value, timestamp
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = "pm10" AND timestamp > "2017-01-01"
            ORDER BY value DESC
            LIMIT 1000
"""
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
df.head(3)
#check the size of MY_QUERY
bq_assistant.estimate_query_size(QUERY)
#now run MY_QUERY
df2 = bq_assistant.query_to_pandas(MY_QUERY)
df2.head(10)
df2['value'].plot()
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
#What are all the U.S. cities in the OpenAQ dataset?
MY_QUERY2 = """ SELECT COUNT(value) AS no_of_cities, city
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE country = "US"
                GROUP BY city
                ORDER BY COUNT(value) DESC
                """
us_cities = bq_assistant.query_to_pandas(MY_QUERY2)
us_cities.head(5)
MY_QUERY3 = """ SELECT country, count(value) AS INSTANCES
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE pollutant != "ppm"
                GROUP BY country
                ORDER BY count(value) DESC
                """
bq_assistant.estimate_query_size(MY_QUERY3)
countries_notmeasuring_in_ppm = bq_assistant.query_to_pandas(MY_QUERY3)
countries_notmeasuring_in_ppm.head(50)
countries_notmeasuring_in_ppm.size
MY_QUERY4 = """ SELECT pollutant, COUNT(value) AS no_of_zero_instances, sum(value)
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value = 0
                GROUP BY pollutant
                HAVING sum(value) < 1000
                ORDER BY COUNT(value) DESC
"""
bq_assistant.estimate_query_size(MY_QUERY4)
pollutant_value_zero = bq_assistant.query_to_pandas(MY_QUERY4)
pollutant_value_zero.head(10)