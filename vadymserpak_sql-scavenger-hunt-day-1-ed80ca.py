# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
import bq_helper as bq
# ccreate db_helper
db_helper = bq.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")
db_helper
# get the tabels from the database
db_helper.list_tables()
# Check the table schema
db_helper.table_schema('global_air_quality')
db_helper.head('global_air_quality')
# Check the query size
query_not_ppm = """SELECT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != "ppm"
"""
db_helper.estimate_query_size(query_not_ppm)
db_helper.query_to_pandas_safe(query_not_ppm, max_gb_scanned=0.0001)
countries_not_ppm = db_helper.query_to_pandas_safe(query_not_ppm, max_gb_scanned=0.001)
countries_not_ppm.head()
# What five countries have the most measurements taken there?
countries_not_ppm.country.value_counts().head()
# get the countries with ppm units
query_ppm = """SELECT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit = "ppm"
"""
countries_ppm = db_helper.query_to_pandas_safe(query_ppm, max_gb_scanned=0.001)
countries_ppm.country.value_counts().head()
# save our dataframes as a .csv 
countries_not_ppm.to_csv("countries_not_ppm.csv")
countries_ppm.to_csv("countries_ppm.csv")
query_poll_0 = """SELECT country, city, pollutant
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value = 0
"""
db_helper.estimate_query_size(query_poll_0)
cities_poll_0 = db_helper.query_to_pandas_safe(query_poll_0, max_gb_scanned=0.001)
cities_poll_0.head()
# in what city the most measurements gave 0?
cities_poll_0.city.value_counts().head()
# what pollution is most often equal 0?
cities_poll_0.pollutant.value_counts().head()
# in what country the most measurements gave 0?
cities_poll_0.country.value_counts().head()
cities_poll_0.to_csv("cities_poll_0")