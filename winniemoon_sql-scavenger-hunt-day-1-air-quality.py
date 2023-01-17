
# Import our bq_helper package
import bq_helper 

# Import other packages
import numpy as np
import pandas as pd
# Create a helper object for the bigquery dataset OpenAQ
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# Print a list of all the tables in the OpenAQ dataset
openaq.list_tables()
# print information on all the columns in the "global_air_quality" table
# in the OpenAQ dataset
openaq.table_schema("global_air_quality")
# Preview the first couple lines of the "global_air_quality" table
openaq.head("global_air_quality")
# Note: use DISTINCT to remove duplicate names of countries
query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# check how big this query will be
openaq.estimate_query_size(query1)
# Check out the countries not using ppm as measure unit 
# Run only if the query is smaller than 10 MB
countries_not_ppm = openaq.query_to_pandas_safe(query1, max_gb_scanned=0.001)

# Check first few results
print(countries_not_ppm.head())
# Print results
print("There are", len(countries_not_ppm), "countries that do not use ppm as unit of measure")
print("These countries are:")
for c in countries_not_ppm['country']:
    print(c)
# Note: use DISTINCT to remove duplicate names of pollutants
query2 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """

openaq.estimate_query_size(query2)
# Check out the pollutants with value equal exactly to 0 
# Run only if the query is smaller than 10 MB
pollutant_zero = openaq.query_to_pandas_safe(query2, max_gb_scanned=0.001)

# Check first few results
print(pollutant_zero.head())
# Print results
print("There are", len(pollutant_zero), "pollutants with value of exactly zero")
print("The pollutants with value of zero are:")
for p in pollutant_zero['pollutant']:
    print(p)
