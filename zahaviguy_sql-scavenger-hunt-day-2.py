# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
#hacker_news.head("comments")
# Let's look at the full table
hacker_news.head("full")
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_types = hacker_news.query_to_pandas_safe(query)
story_types
# query to pass to 
query = """SELECT COUNT(id), deleted
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments
# Bonus question :) 
query = """SELECT AVG(score) as average_score
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING type = "story"
        """
avg_score = hacker_news.query_to_pandas_safe(query)
avg_score