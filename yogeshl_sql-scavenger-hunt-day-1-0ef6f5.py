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
#query to retrieve countries that do use ppm as a unit
query1="""SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm' 
"""

#query to retrieve pollutants that have a value of zero at some point in time
query2="""SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.00
"""

#run the above queries to fetch and store your results
countries_not_using_ppm = open_aq.query_to_pandas_safe(query1)
pollutants_zero_value = open_aq.query_to_pandas_safe(query2)

#display the heads of the data frames
countries_not_using_ppm.head()
pollutants_zero_value.head()

#save to csv files
countries_not_using_ppm.to_csv("countries_not_using_ppm.csv")
pollutants_zero_value.to_csv("pollutants_zero_value.csv")