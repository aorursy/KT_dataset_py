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
# Your code goes here :)
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT COUNT(id), type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
num_of_stories = hacker_news.query_to_pandas_safe(query)
num_of_stories
# How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT COUNT(id), deleted
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments
# Optional extra credit
query = """SELECT MIN(time_ts), author
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            HAVING MIN(time_ts) > '2014-01-01 00:00:00+00:00'
        """
first_comment = hacker_news.query_to_pandas_safe(query)
first_comment
