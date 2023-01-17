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
#first question
#countries that do not use ppm to measure any tipe of pollution
#it means that the unit must be different from p

query_unit = """
                SELECT country, unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
                GROUP BY country, unit
             """

countries =  open_aq.query_to_pandas_safe(query_unit)

#Saved it as a .csv
countries.to_csv("countries_without_ppm.csv")
countries.head()
#Second Question
#Pollutants with a value of 0
query_pollutants = """
                     SELECT location, city, country, timestamp, pollutant, value
                     FROM `bigquery-public-data.openaq.global_air_quality`
                     WHERE value = 0 
                     
                   """
pollutants = open_aq.query_to_pandas_safe(query_pollutants)
#Saved it as a .csv
pollutants.to_csv("pollutants_equals_to_0.csv")
pollutants.head()