import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", 
                                       dataset_name="hacker_news")
print(hacker_news.list_tables())


from google.cloud import bigquery
client = bigquery.Client()

PROJECT_NAME = 'bigquery-public-data'
DATASET_NAME = 'hacker_news'

dataset = client.get_dataset(client.dataset(DATASET_NAME, project=PROJECT_NAME))

BYTES_PER_GB = 2**30
for t in hacker_news.list_tables():
    table = client.get_table(dataset.table(t))
    print('Table (%s) with size (%s) GB'%(t, table.num_bytes/BYTES_PER_GB))
# select from
query = """
SELECT title, time
FROM `bigquery-public-data.hacker_news.stories` 
WHERE `by` = "emyy"
"""
# print(estimate_gigabytes_scanned(query, client))
result = hacker_news.query_to_pandas_safe(query)
result.head(20)
query = """
SELECT score, COUNT(1) AS times
FROM `bigquery-public-data.hacker_news.stories` 
WHERE `by` = "emyy"
GROUP BY 1
"""
result = hacker_news.query_to_pandas_safe(query)
result.head(20)
query = """
SELECT title, time AS unix_time, time_ts AS time_stamp
FROM `bigquery-public-data.hacker_news.stories` 
WHERE `by` = "emyy"
ORDER BY 2 desc
"""
result = hacker_news.query_to_pandas_safe(query)
result.head(20)
query = """
WITH good_contribution as 
(
SELECT 
    COUNT(1) AS counts, 
    author
    
FROM `bigquery-public-data.hacker_news.stories` 
WHERE score > 100
GROUP BY 2
HAVING COUNT(1) > 100
ORDER BY 1 desc
)
SELECT author, counts FROM good_contribution WHERE counts > 150
"""
result = hacker_news.query_to_pandas_safe(query)
result.head(20)
query = """
SELECT
    a.title, a.score, a.url, a.time, a.time_ts,
    COUNT(b.id) as comments
FROM
(
SELECT 
   *
FROM `bigquery-public-data.hacker_news.stories` 
WHERE author = 'ColinWright' and score > 100
) AS a
INNER JOIN `bigquery-public-data.hacker_news.comments` b
ON a.id = b.parent
GROUP BY
    a.title, a.score, a.url, a.time, a.time_ts
ORDER BY 2 DESC
    
"""
result = hacker_news.query_to_pandas_safe(query)
result.head(20)
query = """
SELECT author, COUNT(1) AS times
FROM `bigquery-public-data.hacker_news.stories` 
GROUP BY 1
ORDER BY 2 DESC
"""
result = hacker_news.query_to_pandas_safe(query)
result.head(20)
# visulize result set
import matplotlib.pyplot as plt
# remove the "None" author 
result[1:].boxplot(column='times')
result[1:].hist(column='times', bins=100, bottom=1)

result[1:].times.describe()
# let's see how many contribution is above mean value
result[result.times > 8.75][1:].count()/210368
query = """
SELECT 
    SUM(yc_flag) as yc_counts, 
    count(1) as counts, 
    (SUM(yc_flag)/count(1)) as yc_ratio,
    AVG(score) as mean_score,
    year, 
    month 
FROM
(
SELECT 
    IF(REGEXP_CONTAINS(title, r"( YC |Y-combinator|ycombinator|YCombinator|Y-Combinator)"), 1, 0) AS yc_flag,
    score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
)
GROUP BY 5, 6
ORDER BY 5, 6
"""
result = hacker_news.query_to_pandas_safe(query)
result.head(40)
result.plot(y=['yc_ratio'], x=['year', 'month'], figsize=(20,10))
query = """
SELECT 
    title, 
    author,
    url, 
    id,
    score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"( YC |Y-combinator|ycombinator|YCombinator|Y-Combinator)")
ORDER BY 6, 7
"""
yc_result = hacker_news.query_to_pandas_safe(query)
yc_result.head(20)
# let's look at average YC topic score
fig, ax = plt.subplots(figsize=(20,10))
yc_result.groupby(['year', 'month']).mean()['score'].plot(ax=ax)
result.groupby(['year', 'month']).mean()['mean_score'].plot(ax=ax)
query = """
SELECT 
    AVG(score) as mean_score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"( FB |fb|Facebook|facebook)")
GROUP BY 2, 3
ORDER BY 2, 3
"""
fb_result = hacker_news.query_to_pandas_safe(query)
fb_result.head(20)
query = """
SELECT 
    AVG(score) as mean_score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"(AMAZON|amazon|Amazon)")
GROUP BY 2, 3
ORDER BY 2, 3
"""
amazon_result = hacker_news.query_to_pandas_safe(query)
amazon_result.head(20)
query = """
SELECT 
    AVG(score) as mean_score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"(Apple|apple|APPLE)")
GROUP BY 2, 3
ORDER BY 2, 3
"""
apple_result = hacker_news.query_to_pandas_safe(query)
apple_result.head(20)
query = """
SELECT 
    AVG(score) as mean_score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"(Google|GOOGLE|google)")
GROUP BY 2, 3
ORDER BY 2, 3
"""
google_result = hacker_news.query_to_pandas_safe(query)
google_result.head(20)
query = """
SELECT 
    AVG(score) as mean_score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"(Netflix|NETFLIX|netflix)")
GROUP BY 2, 3
ORDER BY 2, 3
"""
netflix_result = hacker_news.query_to_pandas_safe(query)
netflix_result.head(20)
query = """
SELECT 
    AVG(score) as mean_score,
    EXTRACT(ISOYEAR from time_ts) as year,
    EXTRACT(MONTH from time_ts) as month
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"(LINKIN|Linkin|linkin)")
GROUP BY 2, 3
ORDER BY 2, 3
"""
linkin_result = hacker_news.query_to_pandas_safe(query)
linkin_result.head(20)
fig, ax = plt.subplots(figsize=(20,10))
yc_result.groupby(['year', 'month']).mean()['score'].plot(ax=ax, label='yc', legend=True)
google_result.groupby(['year', 'month']).mean()['mean_score'].plot(ax=ax, label='google', legend=True)
fb_result.groupby(['year', 'month']).mean()['mean_score'].plot(ax=ax, label='facebook', legend=True)
apple_result.groupby(['year', 'month']).mean()['mean_score'].plot(ax=ax, label='apple', legend=True)
linkin_result.groupby(['year', 'month']).mean()['mean_score'].plot(ax=ax, label='linkin', legend=True)
netflix_result.groupby(['year', 'month']).mean()['mean_score'].plot(ax=ax, label='netflix', legend=True)



c = yc_result.groupby('author', as_index=False).count()
c.boxplot(column='id')
import spacy
nlp = spacy.load('en')


%%time

def find_entity(title):
    
    parsed_title = nlp(title)
    entity_arr = []
    for num, entity in enumerate(parsed_title.ents):
        entity_arr.append(entity.text)
    return  ' '.join(entity_arr)

yc_result["entity"]= yc_result["title"].apply(find_entity)
yc_result[yc_result.entity!=''].head()
entity_result = yc_result[yc_result.entity!=''].groupby('entity', as_index=False).agg({'id':'size', 'score':'sum'})
entity_result.sort_values('id', ascending=False)

entity_result[entity_result.id>5].plot(x="entity",y=["score", 'id'],
                     kind="bar",figsize=(14,6),
                     title='Top stories topic').set_ylabel("counts")

