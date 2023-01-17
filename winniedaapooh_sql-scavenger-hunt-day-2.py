# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

hacker_news.list_tables()
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

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

all_types = hacker_news.query_to_pandas_safe(query)

print(all_types)


#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """

delete_comments = hacker_news.query_to_pandas_safe(query)

print(delete_comments)

# Your code goes here :)

query = """ SELECT author, MIN(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            HAVING min(ranking) > 0
            order by min(ranking) 
            limit 199
        """

maximum_ranking = hacker_news.query_to_pandas_safe(query)

print(maximum_ranking)
