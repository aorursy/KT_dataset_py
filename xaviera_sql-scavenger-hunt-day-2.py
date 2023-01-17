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
hacker_news.head("full")
# creating the query for counting each type of story
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(id) > 10
        """
story_types = hacker_news.query_to_pandas_safe(query)
print(story_types)
hacker_news.table_schema("comments")
# creating the query for counting each type of story
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_number = hacker_news.query_to_pandas_safe(query)
print(deleted_number)
# creating the query for counting each type of story
query = """SELECT deleted, COUNTIF(ranking > 10)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
top_deleted_number = hacker_news.query_to_pandas_safe(query)
print(top_deleted_number)