import pandas as pd # package for high-performance, easy-to-use data structures and data analysis

import numpy as np # fundamental package for scientific computing with Python

from pandas import Series



import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

import plotly.plotly as py1

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

from mpl_toolkits.basemap import Basemap

from numpy import array

from matplotlib import cm



# import cufflinks and offline mode

import cufflinks as cf

cf.go_offline()



from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

import base64



from sklearn import preprocessing

# Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings("ignore")



# Print all rows and columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
import bq_helper

import seaborn as sns

import matplotlib.pyplot as plt

import wordcloud

from bq_helper import BigQueryHelper



stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="stackoverflow")
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
survey_data = pd.read_csv('../input/stackoverflow/survey_results_public.csv')
# Users that are either students or without computer science bachelor degree

temp = survey_data['Student'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Community percentage of students', hole = 0.3, color = ['#A9FE00','#FF8102','#FF1302'])
# Users that are either students or without  bachelor degree

temp = survey_data['UndergradMajor'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Community percentage of majors', hole = 0.3)
print(bq_assistant.list_tables())
query1 = """SELECT

  EXTRACT(YEAR FROM creation_date) AS Year,

  COUNT(*) AS Number_of_New_registers

FROM

  `bigquery-public-data.stackoverflow.users`

GROUP BY

  Year

HAVING

  Year > 2008 AND Year <= 2018

ORDER BY

  Year;

        """
stackOverflow.head('users', num_rows = 5)
stackOverflow.head('posts_questions', num_rows = 5)
bq_assistant.estimate_query_size(query1)
new_users = stackOverflow.query_to_pandas_safe(query1)

new_users
# Users that are either students or without  bachelor degree

new_users.iplot(kind='bar', x ='Year', y='Number_of_New_registers', xTitle = 'Year', title='Number of new user in the Community')
# query new users from 2016 to 2018

query2 = """ SELECT user_id, register_date FROM

    (SELECT

  id as user_id,

  date(creation_date) AS register_date,

  EXTRACT(YEAR FROM creation_date) AS Year

    FROM

  `bigquery-public-data.stackoverflow.users`)

WHERE Year > 2016 AND Year <= 2018;

        """
bq_assistant.estimate_query_size(query2)
new_users = stackOverflow.query_to_pandas_safe(query2)

new_users.head()
new_users.to_csv('new_users.csv')
# query all post from 2016 to 2018

query3 = """ SELECT post_id, user_id, accepted_answer_id,comment_count, answer_count, post_date FROM

    (SELECT

  id as post_id,

  date(creation_date) AS post_date,

  EXTRACT(YEAR FROM creation_date) AS Year,

  accepted_answer_id,

  comment_count,

  answer_count,

  owner_user_id as user_id

    FROM

  `bigquery-public-data.stackoverflow.posts_questions`)

WHERE Year > 2016 AND Year <= 2018;

        """
bq_assistant.estimate_query_size(query3)
# new post from 

new_posts = stackOverflow.query_to_pandas_safe(query3)

new_posts.head()
new_posts.to_csv('new_posts.csv')
query1 = """SELECT

  EXTRACT(YEAR FROM creation_date) AS Year,

  COUNT(*) AS Number_of_Questions,

  ROUND(100 * SUM(IF(answer_count > 0, 1, 0)) / COUNT(*), 1) AS Percent_Questions_with_Answers

FROM

  `bigquery-public-data.stackoverflow.posts_questions`

GROUP BY

  Year

HAVING

  Year > 2008 AND Year <= 2018

ORDER BY

  Year;

        """
bq_assistant.estimate_query_size(query1)
answered_questions = stackOverflow.query_to_pandas_safe(query1)

answered_questions.head(5)
ax = sns.barplot(x="Year",y="Percent_Questions_with_Answers",data=answered_questions).set_title("What is the percentage of questions that have been answered over the years?")
query1 = """SELECT

  EXTRACT(YEAR FROM creation_date) AS Year,

  COUNT(*) AS Number_of_Questions,

  SUM(IF(answer_count > 0, 1, 0)) AS Number_Questions_with_Answers

FROM

  `bigquery-public-data.stackoverflow.posts_questions`

GROUP BY

  Year

HAVING

  Year > 2008 AND Year <= 2018

ORDER BY

  Year;

        """



answered_questions = stackOverflow.query_to_pandas_safe(query1)

answered_questions.head(5) 
answered_questions.plot(x="Year",y=["Number_of_Questions","Number_Questions_with_Answers"], 



                    kind="bar",figsize=(14,6), 



                    title='What is the total number of questions and questions that have been answered over the years?')
tag_js_query = '''

    select id, tags

        from `bigquery-public-data.stackoverflow.posts_questions`

            where extract(year from creation_date) > 2016 and

            tags like '%python%'

'''

tags_raw = stackOverflow.query_to_pandas_safe(tag_js_query)

tags_raw.head()



rows_list = []

for _, rows in tags_raw.iterrows():

    tag = rows.tags.split('|')

    for t in tag:

        if t != 'python':

            row = {'question_id': rows.id, 'tag': t}

            rows_list.append(row)

tags_per_question = pd.DataFrame(rows_list)

tags_per_question.head()
query3 = """SELECT 

    REGEXP_EXTRACT(tags, "tensorflow") AS Tag, 

    EXTRACT(YEAR FROM creation_date) AS Year, 

    COUNT(*) AS Number_Spark_Questions

FROM 

    `bigquery-public-data.stackoverflow.posts_questions`

GROUP BY

  Tag, Year

HAVING

  Year > 2008 AND Year <= 2018 AND Tag IS NOT NULL

ORDER BY

  Year;

"""



bq_assistant.estimate_query_size(query3)
spark_questions = stackOverflow.query_to_pandas_safe(query3)

spark_questions.head(5)
ax = sns.barplot(x="Year",y="Number_Spark_Questions",data=spark_questions,palette="coolwarm").set_title("What is the number of questions about Apache Spark over years?")
query4 = """SELECT tags

FROM 

    `bigquery-public-data.stackoverflow.posts_questions`

where extract(year from creation_date) > 2016 and

            tags like '%python%';

"""



alltags = stackOverflow.query_to_pandas_safe(query4)



def remove_python(x):

    A = []

    for i in x.split('|'):

        if i != 'python':

            A.append(i)

    return ''.join(A)



tags = ' '.join(alltags.tags.apply(remove_python)).lower()

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=300,

                            relative_scaling=.5).generate(tags)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('stackOverflow.png')

plt.imshow(cloud);
query5 = """SELECT AVG(comment_count) AS Number_Comments, 

    score AS Score, 

    EXTRACT(YEAR FROM creation_date) AS Year

FROM 

    `bigquery-public-data.stackoverflow.posts_answers`

GROUP BY 

    Score, Year

ORDER BY

    Score;

"""



scores_answers = stackOverflow.query_to_pandas_safe(query5)

scores_answers.head(5)
plt.figure(figsize=(20,10))

plt.scatter(scores_answers["Year"], scores_answers["Score"], c=scores_answers["Number_Comments"], alpha=0.3, cmap='viridis')

plt.xlabel("Year")

plt.ylabel("Score")

plt.title("How average score of answers is evolving over years?")

plt.colorbar();  # show color scale
stackOverflow.head('posts_questions', num_rows = 2)
query1 = """

SELECT

  id,

  title,

  body,

  accepted_answer_id,

  answer_count,

  comment_count,

  score,

  tags,

  view_count

FROM

  (SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions`

WHERE

  EXTRACT(YEAR FROM creation_date) = 2017

LIMIT 500000)

        """
bq_assistant.estimate_query_size(query1)
# new post from 

posts_2017 = stackOverflow.query_to_pandas_safe(query1, max_gb_scanned=26)

posts_2017.head()