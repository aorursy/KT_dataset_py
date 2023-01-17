import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
air_online = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq.global_air_quality")

query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# check how big this query will be
air_online.estimate_query_size(query)
# only run this query if it's less than 100 MB
no_ppm_countries=air_online.query_to_pandas_safe(query, max_gb_scanned=0.1)
no_ppm_countries.country.unique()
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """

# check how big this query will be
air_online.estimate_query_size(query2)
zero_pollutant=air_online.query_to_pandas_safe(query2, max_gb_scanned=0.1)
zero_pollutant.pollutant.unique()