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
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name ="openaq")

#print a list of tables in the open_aq dataset
open_aq.list_tables()

#check the structure 
open_aq.head("global_air_quality")

#query to count the number of rows in the dataset
query_table_size = """ select count(*)
                        from `bigquery-public-data.openaq.global_air_quality` """

#check how big this query will be
open_aq.estimate_query_size(query_table_size)

#run query
table_size = open_aq.query_to_pandas_safe(query_table_size)
table_size


#QUESTION 1
# query to bring back all unique countries who use a measuring unit other than "ppm"
query_question2 = """ select distinct(country), unit
                        from `bigquery-public-data.openaq.global_air_quality`
                        where unit != "ppm" """
#size check
open_aq.estimate_query_size(query_question2)

# Execute query and drop it into a dataframe
nonPPM_countries = open_aq.query_to_pandas_safe(query_question2)


# QUESTION 2
# Query to return pollutants with a value of "0"
query_question3 = """ Select country, pollutant, unit, value
    from `bigquery-public-data.openaq.global_air_quality`
    where value = 0 """

#Size check
open_aq.estimate_query_size(query_question3)

# Execute query and drop it into a dataframe
zero_emmissions = open_aq.query_to_pandas_safe(query_question3)

