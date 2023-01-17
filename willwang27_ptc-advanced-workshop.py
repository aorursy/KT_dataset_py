# importing bq_helper package

import bq_helper
# import the dataset using our bq_helper package

crime_dataset = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 

                                            dataset_name="chicago_crime")
# lists all tables within the medicare_dataset

# print information on all the columns in the "full" table in the medicare dataset

crime_dataset.table_schema("crime")
# this query looks in the crime table in the hacker_news

# dataset, then gets the score column from every row where 

# the type column has "job" in it.

query = """SELECT fbi_code

            FROM `bigquery-public-data.chicago_crime.crime`

            WHERE year = 2020 """



# check how big this query will be

crime_dataset.estimate_query_size(query)
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

my_query = """SELECT city

              FROM `bigquery-public-data.openaq.global_air_quality`

              WHERE country = 'US'

           """
# the query_to_pandas_safe will only return a result if it's less

# than one gigabyte (by default)

us_cities = open_aq.query_to_pandas_safe(my_query)
# What five cities have the most measurements taken there?

us_cities.city.value_counts().head()
# write your own code here (click above if you're stuck)

query1 = """

         """
pollutants = open_aq.query_to_pandas_safe(query1)
pollutants.head()
# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("comments")
# query to pass to 

query = """SELECT parent, COUNT(id)

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            HAVING COUNT(id) > 10

        """
# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# import package with helper functions 

import bq_helper



# create a helper object for this dataset

world_bank_dataset = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="world_bank_health_population")

world_bank_dataset.list_tables()