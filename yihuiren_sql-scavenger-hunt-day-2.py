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
popular_stories.sort_values(by=['f0_'],ascending=False).head(10)
hacker_news.head('full')
# Your code goes here :)
# How many stories (use the "id" column) 
# are there of each type (in the "type" column) in the full table?
query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count_by_type = hacker_news.query_to_pandas_safe(query)

count_by_type.sort_values(by='f0_',ascending=False).head()
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column 
# in the comments table will have the value "True".)

# query to pass to 
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted
        """
count_deleted = hacker_news.query_to_pandas_safe(query)
count_deleted.head()
# query to pass to 
query = """SELECT type, MAX(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
max_score_per_type = hacker_news.query_to_pandas_safe(query)
max_score_per_type.sort_values(by=['f0_'],ascending=False).head()