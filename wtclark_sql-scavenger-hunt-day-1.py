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
# create a query to select all the countries where the unit IS NOT measured in ppm
# would also like to see the measurement as well so unit will be selected as well
query1 = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' ORDER BY country
        """
# create a 'safe' python dataframe that does not exceed one gigabyte
countries_not_ppm = open_aq.query_to_pandas_safe(query1)
# take a look at the first ten outputs of country and unit
countries_not_ppm.head(10)
# create another query for which pollutants have a value of exactly 0?
# bring in the value as well
query2 = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 ORDER BY pollutant
        """
# create a 'safe' python dataframe that does not exceed one gigabyte
pollutant_zero = open_aq.query_to_pandas_safe(query2)
# display the first ten outputs
pollutant_zero.head(10)