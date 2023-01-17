# Importing libraries for converting fetched data to data frame
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# Understanding the schema of the table
hacker_news.table_schema("full")
# print the first couple rows of the table ("full")
hacker_news.head("full",50)
# Running the query for stories ("id" column) for each type (in the "type" column) in the full table and aranging the output in ascending order based on the values in type field.
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY type 
                    """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
# Estimating the size of the query
hacker_news.estimate_query_size(query)
# Checking the output for counts against the most popular values in the type field.
popular_stories.head()
# Checking for the comments that have been deleted in the Comments table.

hacker_news.table_schema("comments")



# print the first couple rows of the table ("full")
hacker_news.head("comments",50)
# Running the query for count of deleted comments in the comment table.
query2 = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
                    """
# Estimating the size of the query
hacker_news.estimate_query_size(query2)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head()