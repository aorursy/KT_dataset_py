# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print  the tables in this dataset 
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Check the Schema of the table 
open_aq.table_schema("global_air_quality")


# Running the query for problem statement A
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# Estimating the approximate size of query executed
open_aq.estimate_query_size(query)
# Checking the output of query and converting to data frame
reported_countries = open_aq.query_to_pandas_safe(query)
# print 10 lines of output to check the data under reported_countries
reported_countries.head(10)
# Running the query for problem statement B
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
#Checking the approximate size of the query executed
open_aq.estimate_query_size(query2)
#Checking the output of query2 and converting to data frame
reported_pollutants = open_aq.query_to_pandas_safe(query2)
# print 10 lines of output to check the data under reported_pollutants
reported_pollutants.head(10)