import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# set up connection and review the structure of dataset
pollution = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                     dataset_name="openaq")
pollution.list_tables()
# look at the table
pollution.head("global_air_quality")
# construct first query and estimate the size
# get the countries/units where unit is not "ppm"
query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """
pollution.estimate_query_size(query)
# run the query
countries = pollution.query_to_pandas_safe(query, max_gb_scanned=0.001)
# checking the result
countries.head()
# getting a list of unique countries
print(countries.country.unique())
print("Number of countries with non-ppm units {}".format(countries.country.unique().shape[0]))
# second query to get pollutants with value equal to zero
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""
pollution.estimate_query_size(query)
# execute the query
pollutants = pollution.query_to_pandas_safe(query, max_gb_scanned=0.003)
# output the values
print(pollutants.pollutant.unique())
print("Number of pollutants with zero value {}".format(pollutants.pollutant.unique().shape[0]))