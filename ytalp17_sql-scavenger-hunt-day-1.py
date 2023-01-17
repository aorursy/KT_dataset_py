# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# query to select countries that use a unit other than ppm to measure any type of pollution 
# use SELECT DISTINCT in order to print a country name only once.
# " != "  means not equal

query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# the query_to_pandas_safe will only return a result if it's less than one gigabyte (by default)
countries_not_ppm = open_aq.query_to_pandas_safe(query)
# after seeing pandas_safe working on console, we can compile our query
countries_not_ppm
# query to select pollutants with exactly zero "0" value 
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
        """
pollutant_zero = open_aq.query_to_pandas_safe(query)
pollutant_zero