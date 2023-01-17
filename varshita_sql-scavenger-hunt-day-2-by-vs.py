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

#Question1: How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
MY_QUERY = """ SELECT COUNT(id) AS No_of_Stories, type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
"""

#estimating the query size
hacker_news.estimate_query_size(MY_QUERY)

#running the query
types_of_stories = hacker_news.query_to_pandas_safe(MY_QUERY)

#checking head elements
types_of_stories
#Question2: How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

MY_QUERY2 = """  SELECT COUNT(id) AS count_of_msg, deleted AS Deleted
                FROM `bigquery-public-data.hacker_news.comments`
                WHERE deleted = TRUE
                GROUP BY deleted
"""

#estimating the query size
hacker_news.estimate_query_size(MY_QUERY2)

#running the query
deleted_msg = hacker_news.query_to_pandas_safe(MY_QUERY2)

#checking head elements
deleted_msg

#Question3: Optional extra credit: read about aggregate functions other than COUNT() and modify one of the queries you wrote above to use a different aggregate function.
#list of story submitters with their average score!

MY_QUERY3 = """  SELECT AVG(score) AS Average_Story_Score, COUNT(score) AS No_of_stories, `by` AS User_Name
                FROM `bigquery-public-data.hacker_news.stories`
                GROUP BY `by`
                HAVING AVG(score) > 0 AND COUNT(score) > 1
                ORDER BY AVG(score) DESC
"""

#estimating the query size
hacker_news.estimate_query_size(MY_QUERY3)

#running the query
submitter_average_score = hacker_news.query_to_pandas_safe(MY_QUERY3)

#checking head elements
submitter_average_score