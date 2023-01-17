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
# print the first couple rows of the "full" table
hacker_news.head("full")
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type """
# Run query and store in pandas dataframe
stories_by_type = hacker_news.query_to_pandas_safe(query1)

# Check first few results
stories_by_type.head()
hacker_news.head("comments")
# Group by deleted column (True and None) and then filter group with deleted = 'True'
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted 
            HAVING deleted = True"""
# Run query and store in pandas dataframe
deleted_comments = hacker_news.query_to_pandas_safe(query2)

# Check results
deleted_comments.head()
# Alternative code for question 2 using SUM() instead of COUNT()
# Note: deleted is a boolean type so convert into numbers where True = 1
#       then sum of deleted will give the total number of deleted = True
query3 = """SELECT deleted, SUM(CAST(deleted AS INT64))
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted 
            HAVING deleted = True"""
# Run query and store in pandas dataframe
deleted_comments2 = hacker_news.query_to_pandas_safe(query3)

# Check results
deleted_comments2.head()
