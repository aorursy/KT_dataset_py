# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



# print all the tables in this dataset (there's only one!)

open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset

open_aq.head("global_air_quality")
open_aq.head('global_air_quality')
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
query = """SELECT city, country

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """
query = """SELECT *

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")



# this query looks in the full table in the hacker_news

# dataset, then gets the score column from every row where 

# the type column has "job" in it.

query = """SELECT score

            FROM `bigquery-public-data.hacker_news.full`

            WHERE type = "job" """



# check how big this query will be

hacker_news.estimate_query_size(query)

# only run this query if it's less than 100 MB

hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the scores of job postings (if the 

# query is smaller than 1 gig)

job_post_scores = hacker_news.query_to_pandas_safe(query)
# average score for job posts

job_post_scores.score.mean()

type(query)