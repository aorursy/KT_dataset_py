#import package with helper instructions
import bq_helper

#create a helper object
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                  dataset_name = "openaq")

# print all tables
open_aq.list_tables()
# print first rows
open_aq.head("global_air_quality")
# Select the items from "city" column where country is "US"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

# run query only if it returns result less than 1 gigabyte
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Select items form "country" column where unit is not equal ppm.
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# Run query safely
countries = open_aq.query_to_pandas_safe(query)
# Check results
# print unique values
countries.country.unique()
# Select items fron "pollutant" column where the "value" equals 0.00
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

# Run query safely
pollutants = open_aq.query_to_pandas_safe(query)
# Print unique values
pollutants.pollutant.unique()