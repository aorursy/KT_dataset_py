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
query1 = """SELECT COUNT(id) as No_Of_Stories, type as Type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY 1 DESC;
"""
ans1 = hacker_news.query_to_pandas_safe(query1)
ans1.head()
query2 = """SELECT COUNT(id) as Total_Deleted_Comments
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True;        
"""
ans2 = hacker_news.query_to_pandas_safe(query2)
ans2.head()
# Using SUM aggregate function
query3 = """SELECT SUM(using_case) as Total_Deleted_Comments 
            FROM (SELECT (CASE WHEN deleted = True THEN 1
                          ELSE 0
                          END) as using_case
                    FROM `bigquery-public-data.hacker_news.comments`
            );
"""
ans3 = hacker_news.query_to_pandas_safe(query3)
ans3.head()