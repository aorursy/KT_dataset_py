import bq_helper 
import pandas as pd
import numpy as np
hacker_news = bq_helper.BigQueryHelper(active_project= 'bigquery-public-data', 
                                       dataset_name = 'hacker_news')
hacker_news.list_tables()
hacker_news.table_schema('comments')
hacker_news.head('comments',num_rows=10)
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY parent
            ORDER BY COUNT(id) DESC
            """

hacker_news.estimate_query_size(query)
popular_stories=hacker_news.query_to_pandas_safe(query, max_gb_scanned=1)
popular_stories.head()
type(popular_stories)
popular_stories[popular_stories.parent==5371725]
# I want to find how many distinct commenters commented on a story. 
# Which story got the highest number of distinct commenters?  
query = """SELECT parent, COUNT(`by`) AS num_commenter 
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent, `by`
           ORDER BY COUNT(`by`) DESC
           """

hacker_news.estimate_query_size(query)
highest_commenters= hacker_news.query_to_pandas_safe(query)
highest_commenters.head()
# save our dataframe as a .csv 
popular_stories.to_csv("popular_stories.csv")
hacker_news.table_schema('full')
#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT type, COUNT(id) AS num_stories 
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
           ORDER BY COUNT(id) DESC
        """
hacker_news.estimate_query_size(query)
story_types=hacker_news.query_to_pandas_safe(query)
story_types.head(10)
hacker_news.head('full',num_rows=10)
query = """SELECT COUNT(deleted) AS num_deleted
           FROM `bigquery-public-data.hacker_news.full`
           WHERE type='comment' AND deleted=True
        """
hacker_news.estimate_query_size(query)
deleted_stories=hacker_news.query_to_pandas_safe(query)
deleted_stories.head()
# CONTIF only counts the number of True instances of an expression
query = """SELECT COUNTIF(deleted)
           FROM `bigquery-public-data.hacker_news.full`
           WHERE type='comment'
        """
hacker_news.estimate_query_size(query)
deleted_stories2=hacker_news.query_to_pandas_safe(query)
deleted_stories2.head()