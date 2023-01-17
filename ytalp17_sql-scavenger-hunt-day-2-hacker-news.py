# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# How many stories are there of each type in the full table? Question 1
# print the first couple rows of the "full" table
hacker_news.head("full")
# query to find number of stories in each type on full data
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# is it safe to run this query ?
number_of_stories = hacker_news.query_to_pandas_safe(query)

#run the query
number_of_stories

# How many comments have been deleted? Question 2

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to find number of deleted comments on "comments" table
query = """SELECT  COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
# is it safe to run this query ?
number_of_deleted_comments = hacker_news.query_to_pandas_safe(query)

#run the query
number_of_deleted_comments
#alternative aggregate function Q-2
query = """SELECT  COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.comments`
        """
# is it safe to run this query ?
number_of_deleted_comments = hacker_news.query_to_pandas_safe(query)

#run the query
number_of_deleted_comments
