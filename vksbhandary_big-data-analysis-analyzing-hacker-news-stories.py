from google.cloud import bigquery

import pandas as pd

import bq_helper 

import numpy as np

%matplotlib inline

from wordcloud import WordCloud

import matplotlib.pyplot as plt



import spacy

nlp = spacy.load('en')



# create a helper object for our bigquery dataset

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")



client = bigquery.Client()



%matplotlib inline
hacker_news.head("stories")
## So we will declare the query variable

## we will be using same name for queries in entire notebook hope this doesnt create confusion

query = """

SELECT MIN(score) as min_score, MAX(score) as max_score, AVG(score) as avg_score

FROM `bigquery-public-data.hacker_news.stories`

"""

hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

rows = list(iterator)
# Transform the rows into a nice pandas dataframe

unique_scores = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

unique_scores.fillna(0, inplace=True)

# Look at the first 10 top scores

unique_scores.head()
query = """

SELECT 

SUM(score) as total_score, author,

SUM(descendants) as total_decendants,

count(id) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

GROUP BY author

ORDER BY total_score DESC

LIMIT 30

"""

hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

rows = list(iterator)
# Transform the rows into a nice pandas dataframe

top_users = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

top_users.fillna(0, inplace=True)

# Look at the first 10 top scores

top_users.sort_values("total_score")[::-1][:10].plot(x="author",y=["total_score", "total_decendants", "total_stories"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top users stats').set_ylabel("Score")
query = """

SELECT 

COUNT(*) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

"""

hacker_news.estimate_query_size(query)
query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)
count = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
print("Total number of stories {}".format(count.total_stories.values[0]))
query = """

SELECT 

SUM(score) as total_score, author,

SUM(descendants) as total_decendants,

count(id) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

WHERE author IS NOT NULL

GROUP BY author

ORDER BY total_stories DESC

LIMIT 30

"""

hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

rows = list(iterator)
# Transform the rows into a nice pandas dataframe

most_freq_users = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
# Look at the first 10 top scores

most_freq_users.sort_values("total_stories")[::-1][:10].plot(x="author",y=["total_score", "total_decendants", "total_stories"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top users stats').set_ylabel("Score")
print("Total number of stories {} posted by 30 most active users".format(most_freq_users.total_stories.sum()))
print("Percentage stories posted by 30 most active users : {}%".format(most_freq_users.total_stories.sum()*100/count.total_stories.values[0]))
# Writing query string for our specific need here

query = """

SELECT score, title

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"Y Combinator|YCombinator")

ORDER BY score  DESC

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

HNrows = list(iterator)
# Transform the rows into a nice pandas dataframe

HN_scores = pd.DataFrame(data=[list(x.values()) for x in HNrows], columns=list(HNrows[0].keys()))

HN_scores.sort_values("score")[::-1][:10].plot(x="title",y=["score"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top scoring YC Stories').set_ylabel("Score")
print("Total number of stories with mention of YC {}".format(len(HN_scores)))
print("Avg score of stories with mention of YC {}".format(HN_scores.score.mean()))
# Writing query string for our specific need here

query = """

SELECT AVG(score) as avg_score

FROM `bigquery-public-data.hacker_news.stories`

WHERE id NOT IN (SELECT id FROM `bigquery-public-data.hacker_news.stories` WHERE REGEXP_CONTAINS(title, r"Y Combinator|YCombinator")) 

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

Non_HN_score = list(iterator)
Non_HN_df = pd.DataFrame(data=[list(x.values()) for x in Non_HN_score], columns=list(Non_HN_score[0].keys()))

print("Total average score of stories without mention of HN {}".format(Non_HN_df.avg_score.values[0]))
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(U|u)ber")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

uber_stories_per_year = list(iterator)
uber_stories_df = pd.DataFrame(data=[list(x.values()) for x in uber_stories_per_year], columns=list(uber_stories_per_year[0].keys()))

uber_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Uber stories per year').set_ylabel("Score")


query ="""

SELECT count(*) as total_stories, SUM(score) as total_score, 

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(H|h)ewlett(-)(P|p)ackard|HP|hp")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

hp_stories_per_year = list(iterator)
hp_stories_df = pd.DataFrame(data=[list(x.values()) for x in hp_stories_per_year], columns=list(hp_stories_per_year[0].keys()))

hp_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='HP stories per year').set_ylabel("Score")

query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(T|t)esla")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

tesla_stories_per_year = list(iterator)
tesla_stories_df = pd.DataFrame(data=[list(x.values()) for x in tesla_stories_per_year], columns=list(tesla_stories_per_year[0].keys()))

tesla_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Tesla stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(Y|y)ahoo")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

yahoo_stories_per_year = list(iterator)
yahoo_stories_df = pd.DataFrame(data=[list(x.values()) for x in yahoo_stories_per_year], columns=list(yahoo_stories_per_year[0].keys()))

yahoo_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Yahoo stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(F|f)acebook")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

fb_stories_per_year = list(iterator)
fb_stories_df = pd.DataFrame(data=[list(x.values()) for x in fb_stories_per_year], columns=list(fb_stories_per_year[0].keys()))

fb_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Facebook stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(T|t)witch.tv")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

twitch_stories_per_year = list(iterator)
twitch_stories_df = pd.DataFrame(data=[list(x.values()) for x in twitch_stories_per_year], columns=list(twitch_stories_per_year[0].keys()))

twitch_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Twitch stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(R|r)eddit")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

reddit_stories_per_year = list(iterator)
reddit_stories_df = pd.DataFrame(data=[list(x.values()) for x in reddit_stories_per_year], columns=list(reddit_stories_per_year[0].keys()))

reddit_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Reddit stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(Z|z)illabyte")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

zillabyte_stories_per_year = list(iterator)
zillabyte_stories_df = pd.DataFrame(data=[list(x.values()) for x in zillabyte_stories_per_year], columns=list(zillabyte_stories_per_year[0].keys()))

zillabyte_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Zillabyte stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(K|k)icksend")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

kicksend_stories_per_year = list(iterator)
kicksend_stories_df = pd.DataFrame(data=[list(x.values()) for x in kicksend_stories_per_year], columns=list(kicksend_stories_per_year[0].keys()))

kicksend_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Kicksend stories per year').set_ylabel("Score")
companies_list = ["scribd","weebly" ,"songkick","ninite", "startuply","airbnb"]



query ="""

SELECT title, score, time_ts

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"Scribd |Weebly |Songkick |Ninite |Startuply |AirBnb ") 

ORDER BY time_ts

"""

hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

hn_ompanies = list(iterator)
print ("Total stories by all of these companies",len(hn_ompanies))
companies_stories_df = pd.DataFrame(data=[list(x.values()) for x in hn_ompanies], columns=list(hn_ompanies[0].keys()))

companies_stories_df["title"] =companies_stories_df["title"].str.lower()
companies_stories_df['year'] = companies_stories_df['time_ts'].dt.year
for company in companies_list:

    comp_df = companies_stories_df.loc[companies_stories_df["title"].str.contains(company)]

    if len(comp_df)>0 :

        agg_d = {'score':'sum'}

        comp_plot = comp_df.groupby(["year"], as_index=False).agg(agg_d)

        comp_plot["counts"] = comp_df.groupby(["year"], as_index=False).size().reset_index(name='counts').counts

        comp_plot.plot(x="year",y=["score","counts"],

                     kind="bar",figsize=(14,6),

                     title=company+' stories per year').set_ylabel("Score")

        plt.show()
query ="""

SELECT title

FROM `bigquery-public-data.hacker_news.stories`

ORDER BY score DESC

LIMIT 1000

"""

hacker_news.estimate_query_size(query)
query_job = client.query(query)

iterator = query_job.result(timeout=30)

top_stories_hn = list(iterator)
top_stories_df = pd.DataFrame(data=[list(x.values()) for x in top_stories_hn], columns=list(top_stories_hn[0].keys()))

top_stories_df.head()
%%time



def find_entity(title):

    

    parsed_title = nlp(title)

    entity_arr = []

    for num, entity in enumerate(parsed_title.ents):

        entity_arr.append(entity.text)

    return  ' '.join(entity_arr)



top_stories_df["entity"]= top_stories_df["title"].apply(find_entity)

top_stories_df = top_stories_df.loc[~ (top_stories_df['entity'] == "")]

top_stories_df.head()

top_entities = top_stories_df.groupby(["entity"]).size().reset_index(name='counts')

top_entities_p =top_entities.sort_values(['counts'])[::-1][:15]
top_entities_p.plot(x="entity",y=["counts"],

                     kind="bar",figsize=(14,6),

                     title='Top stories topic').set_ylabel("counts")
cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(top_entities.entity.values))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off');

plt.show()