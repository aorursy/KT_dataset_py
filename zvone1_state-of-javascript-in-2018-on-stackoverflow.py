import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import exponnorm
import numpy as np
from bq_helper import BigQueryHelper
stack_overflow = BigQueryHelper(active_project="bigquery-public-data", dataset_name="stackoverflow")
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
bq_assistant.list_tables()
total_questions_count_query = '''
    select count(*) as total_number_of_questions 
        from `bigquery-public-data.stackoverflow.posts_questions` 
        where extract(year from creation_date) = 2018
'''
total_questions_count = stack_overflow.query_to_pandas_safe(total_questions_count_query)
total_questions_count
total_js_questions_count_query = '''
    select count(*) as number_of_javascript_questions 
        from `bigquery-public-data.stackoverflow.posts_questions` 
        where extract(year from creation_date) = 2018 and
        tags like '%javascript%'
'''
total_js_questions_count = stack_overflow.query_to_pandas_safe(total_js_questions_count_query)
total_js_questions_count
questions_per_day_query = '''
    select count(id) as q_count, extract(day from creation_date) as day, extract(month from creation_date) as month
        from `bigquery-public-data.stackoverflow.posts_questions` 
        where extract(year from creation_date) = 2018 and
        tags like '%javascript%'
        group by day, month
'''
questions_per_day = stack_overflow.query_to_pandas_safe(questions_per_day_query)
questions_per_day.head()
questions_per_day.q_count = questions_per_day.q_count.values.astype(int)
pivoted_table = questions_per_day.pivot('day', 'month', 'q_count')

plt.figure(figsize=(16,12))
sns.heatmap(data=pivoted_table, annot=True, fmt='.0f', linewidths=.5)
answers_to_js_count_query = '''
    select id, accepted_answer_id, answer_count, comment_count
    from `bigquery-public-data.stackoverflow.posts_questions`
    where extract(year from creation_date) = 2018 and
    tags like '%javascript%'
'''
answers_to_js_count = stack_overflow.query_to_pandas_safe(answers_to_js_count_query)
answers_to_js_count.head()
votes_and_views_js_count_query = '''
    select id, favorite_count, view_count
    from `bigquery-public-data.stackoverflow.posts_questions`
    where extract(year from creation_date) = 2018 and
    tags like '%javascript%'
'''
votes_and_views_js_count = stack_overflow.query_to_pandas_safe(votes_and_views_js_count_query)
votes_and_views_js_count = votes_and_views_js_count.fillna(0)
votes_and_views_js_count.head()
answers_to_js_count['has_accepted_answer'] = answers_to_js_count.apply(lambda r: ~np.isnan(r.accepted_answer_id), axis=1)
answers_to_js_count.groupby('has_accepted_answer').count()
plt.figure(figsize=(20,10))
sns.distplot(answers_to_js_count.
             answer_count.values,
             fit=exponnorm,
             bins=20,
             axlabel='Number of answers',
             kde=False,
             rug=True)
plt.figure(figsize=(20,10))
sns.distplot(answers_to_js_count.comment_count,
             bins=40,
             fit=exponnorm,
             axlabel='Number of comments',
             kde=False,
             rug=True)
tag_js_query = '''
    select id, tags
        from `bigquery-public-data.stackoverflow.posts_questions`
            where extract(year from creation_date) = 2018 and
            tags like '%javascript%'
'''
tags_raw = stack_overflow.query_to_pandas_safe(tag_js_query)
tags_raw.head()

rows_list = []
for _, rows in tags_raw.iterrows():
    tag = rows.tags.split('|')
    for t in tag:
        if t != 'javascript':
            row = {'question_id': rows.id, 'tag': t}
            rows_list.append(row)
tags_per_question = pd.DataFrame(rows_list)
tags_per_question.head()
tag_count = tags_per_question.groupby('tag').count().sort_values(by='question_id', ascending=False)
tag_count.head(20)
plt.figure(figsize=(20,10))
sns.barplot(x=tag_count.index[0:20], y=tag_count.question_id[0:20])
top20tags = tag_count.head(20).index.values
answers_tags = pd.merge(answers_to_js_count, tags_per_question, left_on='id', right_on='question_id', how='left')
views_tags = pd.merge(votes_and_views_js_count, tags_per_question, left_on='id', right_on='question_id', how='left')
answers_tag_grouped = answers_tags[answers_tags.tag.isin(top20tags)][['tag', 'answer_count', 'comment_count', 'has_accepted_answer']].groupby('tag')
avg_answer_count = answers_tag_grouped.mean().sort_values('has_accepted_answer', ascending=False)
avg_answer_count
answers_top20_tags = answers_tags[answers_tags.tag.isin(top20tags)]
views_top20_tags = views_tags[views_tags.tag.isin(top20tags)]
plt.figure(figsize=(20, 8))
sns.stripplot(data=answers_top20_tags, x='tag', y='answer_count', jitter=True)
tag_scatter_data = answers_tags[['tag', 'answer_count', 'comment_count']].groupby('tag').sum()
tag_scatter_data = tag_count.join(tag_scatter_data, how='left')
tag_scatter_data['tag'] = tag_scatter_data.index.values

sns.jointplot('answer_count', 'comment_count', data=tag_scatter_data, height=10, kind='reg')
avg_views = views_top20_tags[['tag', 'favorite_count', 'view_count']].groupby('tag').mean()
avg_views = avg_views.sort_values('view_count', ascending=False)

f = plt.figure(figsize=(16, 12))

ax = f.add_subplot(2, 1, 1)
plt.xticks(rotation=90)
sns.barplot(x = avg_views.index, y=avg_views.view_count, ax=ax)
plt.subplots_adjust(hspace=1)

avg_views = avg_views.sort_values('favorite_count', ascending=False)
ax = f.add_subplot(2, 1, 2)
plt.xticks(rotation=90)
sns.barplot(x = avg_views.index, y=avg_views.favorite_count, ax=ax)
merged_avg_answer_view = pd.merge(avg_views, avg_answer_count, left_index=True, right_index=True)
merged_avg_answer_view = merged_avg_answer_view[['view_count', 'answer_count']].copy()
merged_avg_answer_view['tag'] = merged_avg_answer_view.index.values
merged_avg_answer_view['ratio'] = merged_avg_answer_view.view_count/merged_avg_answer_view.answer_count
merged_avg_answer_view.sort_values('ratio', ascending=False)
sns.jointplot('answer_count', 'view_count', merged_avg_answer_view, height=6, kind='kde')
plt.figure(figsize=(16,8))
sns.scatterplot(x='answer_count', y='view_count', hue='tag', data=merged_avg_answer_view, palette='hls', s=100)
user_location_query = '''
    select u.location, q.tags
    from `bigquery-public-data.stackoverflow.posts_questions` q
    left join `bigquery-public-data.stackoverflow.users` u on q.owner_user_id = u.id
        where extract(year from q.creation_date) = 2018 and
        q.tags like '%javascript%'
'''

geo_locations = stack_overflow.query_to_pandas_safe(user_location_query)
geo_locations.head()
geo_tag_list = []
for _, rows in geo_locations.iterrows():
    tag = rows.tags.split('|')
    for t in tag:
        if t != 'javascript':
            row = {'location': rows.location, 'tag': t}
            geo_tag_list.append(row)

geo_tag_data = pd.DataFrame(geo_tag_list)
geo_tag_data.head()
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
angular_loc = geo_tag_data[geo_tag_data.tag == 'angular'].groupby('location').count().sort_values('tag', ascending=False).head(10)
react_loc = geo_tag_data[geo_tag_data.tag == 'reactjs'].groupby('location').count().sort_values('tag', ascending=False).head(10)
vue_loc = geo_tag_data[geo_tag_data.tag == 'vue.js'].groupby('location').count().sort_values('tag', ascending=False).head(10)

display_side_by_side(angular_loc, react_loc, vue_loc)
node_loc = geo_tag_data[geo_tag_data.tag == 'node.js'].groupby('location').count().sort_values('tag', ascending=False).head(10)
php_loc  = geo_tag_data[geo_tag_data.tag == 'php'].groupby('location').count().sort_values('tag', ascending=False).head(10)
display_side_by_side(node_loc, php_loc)
