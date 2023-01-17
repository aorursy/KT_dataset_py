# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the first couple rows of the "comments" table
hacker_news.head("full")
q1 = """SELECT type as Type, COUNT(id) as CountStories
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """
s1 = hacker_news.query_to_pandas_safe(q1)
s1
hacker_news.head("comments")
q2 = """SELECT deleted, COUNT(id) as Count
        FROM `bigquery-public-data.hacker_news.comments`
        WHERE deleted = True
        GROUP BY deleted
        """
s2 = hacker_news.query_to_pandas_safe(q2)
s2
q3 = """SELECT deleted, SUM(id) as Count
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY deleted
        """
s3 = hacker_news.query_to_pandas_safe(q3)
s3