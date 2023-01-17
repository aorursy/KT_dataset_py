# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id) as Number
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.head("comments")
query_3 = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE score IS NOT NULL
            GROUP BY type
        """
average_score = hacker_news.query_to_pandas_safe(query_3)
average_score.head()
query_2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted IS NOT True      
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query_2)
deleted_comments.head()
hacker_news.head("full")
# Your code goes here :)
query_1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """
popular_types = hacker_news.query_to_pandas_safe(query_1)
popular_types.head()

