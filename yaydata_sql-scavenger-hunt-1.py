import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # help functions for Big Query
openaq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                               dataset_name = "openaq")
# View the contents of the dataset.
openaq.head("global_air_quality")
# Examine schema.
openaq.table_schema("global_air_quality")
# Create a query to answer question 1.
# Countries have many cities so they appear in the dataset many times. We only want each non-ppm
#   country to appear once in the results so use SELECT DISTINCT. Note that this assumes that all
#   cities in the same country use the same units to report pollution.
query1 = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# check how big this query will be, size is returned in gigabites
openaq.estimate_query_size(query1)
# Create a csv of countries that doesn't use ppm to measure air quality.
# Limit the query size to 0.25 gigs
non_ppm = openaq.query_to_pandas_safe(query1, max_gb_scanned=0.25)
non_ppm.to_csv("non_ppm.csv")
# Create a query to answer question 2.
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0"""

# check how big this query will be, size is returned in gigabites
openaq.estimate_query_size(query1)
# Create a csv of pollutants that have a pollutant value of 0.
# Limit the query size to 0.25 gigs
zero_pollutant = openaq.query_to_pandas_safe(query2, max_gb_scanned=0.25)
zero_pollutant.to_csv("zero_pollutant.csv")