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
#First of all let's see how the full table is structured 

hacker_news.head("full")

query_num_stories = """
                    SELECT type, COUNT(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY type
                    HAVING COUNT(id) > 10
                    """
#run the query safely 

num_stories = hacker_news.query_to_pandas_safe(query_num_stories)

#print the first rows
num_stories.head()

query_deleted_comments = """
                            SELECT COUNT(id)
                            FROM `bigquery-public-data.hacker_news.comments`
                            WHERE deleted = True
                         """
num_del_comments = hacker_news.query_to_pandas_safe(query_deleted_comments)

#let's print the result
num_del_comments.head()
query_max_score =   """
                    SELECT author ,MAX(ranking)
                    FROM `bigquery-public-data.hacker_news.comments`
                    GROUP BY author 
                    HAVING MAX(ranking) >100
                    ORDER BY MAX(ranking) DESC
                    """
max_score = hacker_news.query_to_pandas_safe(query_max_score)
max_score.head()
