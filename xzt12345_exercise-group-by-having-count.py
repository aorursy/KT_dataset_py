# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex3 import *



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("comments")
# Your Code Here



prolific_commenters_query = """SELECT author,COUNT(id) AS NumPosts

                               FROM `bigquery-public-data.hacker_news.comments`

                               GROUP BY author

                               HAVING COUNT(id) > 10000

                            """   

prolific_commenters = hacker_news.query_to_pandas_safe(prolific_commenters_query)

# print(prolific_commenters)



q_1.check() 
prolific_commenters_query = """SELECT author, COUNT(id) NumPosts

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY author

            HAVING COUNT(id) > 10000

        """

prolific_commenters = hacker_news.query_to_pandas_safe(prolific_commenters_query)

# print(prolific_commenters)



q_1.check() 
q_1.solution()
# Write your query here and figure out the answer
# Submit the answer by setting the following value

num_deleted_posts = """SELECT COUNT(id)  num_deleted_posts

            FROM `bigquery-public-data.hacker_news.comments`

            HAVING deleted = True

        """



q_2.check()
deleted_posts_query = """SELECT COUNT(id) num_deleted_posts

            FROM `bigquery-public-data.hacker_news.comments`

            WHERE deleted = True

        """

deleted_posts_results = hacker_news.query_to_pandas_safe(deleted_posts_query)

q_2.check()
q_2.solution()