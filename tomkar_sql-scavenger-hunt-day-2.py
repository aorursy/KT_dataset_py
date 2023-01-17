# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
query = """SELECT COUNT(1) total, COUNT(DISTINCT id) as stories
            FROM `bigquery-public-data.hacker_news.full`
        """

counts = hacker_news.query_to_pandas_safe(query)
counts
query = """SELECT type, COUNT(*) as stories
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

story_type = hacker_news.query_to_pandas_safe(query)
tot = story_type.stories.sum()
story_type['Percent']= (story_type['stories']/tot)*100
story_type
import matplotlib.pyplot as plot

p1 = plot.bar(story_type.type, story_type.stories, 0.4)

plot.ylabel('# of stories')
plot.title('Number of stories by story category')

plot.show()
### count for each value of deleted
query = """SELECT deleted, count(*) as deleted_news
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY deleted
        """
del1 = hacker_news.query_to_pandas_safe(query)

del1['Percent']= (del1['deleted_news']/tot)*100
del1

query = """SELECT COUNT(1) total,
           SUM(CASE WHEN deleted = True THEN 1 ELSE 0 END) as deleted_news
           FROM `bigquery-public-data.hacker_news.full`
        """

del2 = hacker_news.query_to_pandas_safe(query)
del2