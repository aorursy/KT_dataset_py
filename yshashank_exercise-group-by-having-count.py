# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
#hacker_news.head("full")
query1 = """ SELECT type, COUNT(id) FROM `bigquery-public-data.hacker_news.full` GROUP BY type """
#hacker_news.estimate_query_size(query1)
hacker_news.query_to_pandas_safe(query1)
# Your Code Here
query2 = """ SELECT deleted, COUNT(id) FROM `bigquery-public-data.hacker_news.comments` GROUP BY deleted HAVING COUNT(deleted) > 0 """
#hacker_news.estimate_query_size(query2)
hacker_news.query_to_pandas_safe(query2)
# Your Code Here
query3 = """ SELECT deleted, COUNTIF(deleted) FROM `bigquery-public-data.hacker_news.comments` GROUP BY deleted """
#hacker_news.estimate_query_size(query3)
hacker_news.query_to_pandas_safe(query3)