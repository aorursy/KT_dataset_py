import pandas as pd
import numpy as np
import bq_helper
aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "OpenAQ")
aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
notppm = aq.query_to_pandas_safe(query1)
notppm.head()
query2 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zeropollutant = aq.query_to_pandas_safe(query2)
zeropollutant.head()
