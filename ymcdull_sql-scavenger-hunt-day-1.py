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
query = """SELECT city, country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
type(us_cities)
us_cities.shape
us_cities.pollutant.unique()
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'pm10' and pollutant != 'pm25'
        """
df1 = open_aq.query_to_pandas_safe(query)
query = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
df2 = open_aq.query_to_pandas_safe(query)
df2