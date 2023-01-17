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
## Question 1 - stories by type
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
print(hacker_news.estimate_query_size(query))
story_types = hacker_news.query_to_pandas_safe(query)
print(story_types)

ax = story_types.plot(kind='barh')
_= ax.set_yticklabels(story_types['type'])
## Question 2 - how many comments were deleted
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
print(hacker_news.estimate_query_size(query))
deleted_comments = hacker_news.query_to_pandas_safe(query)
print("{} comments were deleted or {:2.2f}% of the whole amount".format(
    deleted_comments.loc[deleted_comments["deleted"] == True, "f0_"][0],
    deleted_comments.loc[deleted_comments["deleted"] == True, "f0_"][0]/deleted_comments["f0_"].sum()*100
))
## Question 3 (optional) - average length of comments by hour
query = """SELECT EXTRACT(HOUR FROM TIMESTAMP_SECONDS(time)), AVG(LENGTH(text))
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY EXTRACT(HOUR FROM TIMESTAMP_SECONDS(time))
        """
print(hacker_news.estimate_query_size(query))
stories_length = hacker_news.query_to_pandas_safe(query, max_gb_scanned=3.1)
ax = stories_length.plot(x="f0_", y="f1_", kind="scatter")
_= ax.set_xlabel("Hour")
_= ax.set_ylabel("Avg length of comments")

# Looks like a dip around lunch time, eh?