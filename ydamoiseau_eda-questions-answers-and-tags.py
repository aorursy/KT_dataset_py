import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib as plt

import matplotlib.patches as mpatches

%matplotlib inline

from datetime import datetime, timedelta

import warnings

from matplotlib import pyplot

warnings.filterwarnings("ignore")

# print(os.listdir("../input"))
# Loading the email data and establishing some basic information

questions = pd.read_csv('../input/questions.csv', parse_dates=['questions_date_added'])

answers = pd.read_csv('../input/answers.csv', parse_dates=['answers_date_added'])

min_q_date = min(questions['questions_date_added'])

max_q_date = max(questions['questions_date_added'])

min_a_date = min(answers['answers_date_added'])

max_a_date = max(answers['answers_date_added'])

print('{:,} questions were added between {} and {}'.format(questions.shape[0], min_q_date.strftime('%Y-%m-%d'), max_q_date.strftime('%Y-%m-%d')))

print('and')

print('{:,} answers were added between {} and {}'.format(answers.shape[0], min_a_date.strftime('%Y-%m-%d'), max_a_date.strftime('%Y-%m-%d')))
# Count of answers

temp = answers.groupby('answers_question_id').size()

questions['questions_answers_count'] = pd.merge(questions, pd.DataFrame(temp.rename('count')), left_on='questions_id', right_index=True, how='left')['count'].fillna(0).astype(int)

# First answer for questions

firstansw = answers[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').min()

quest = questions.copy()

quest['questions_first_answers'] = pd.merge(quest, pd.DataFrame(firstansw), left_on='questions_id', right_index=True, how='left', indicator=True)['answers_date_added']

# Last answer for questions

lastansw = answers[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').max()

quest['questions_last_answers'] = pd.merge(quest, pd.DataFrame(lastansw), left_on='questions_id', right_index=True, how='left')['answers_date_added']

# Days required to answer the question

answ = answers.copy()

temp = pd.merge(quest, answers, left_on='questions_id', right_on='answers_question_id')

answers['time_delta_answer'] = (temp['answers_date_added'] - temp['questions_date_added'])

# Days until first answer

quest['time_until_1stanswer'] = (quest['questions_first_answers'] - quest['questions_date_added']).dt.days

ax_data = pd.DataFrame(quest['time_until_1stanswer'])

tempZ = quest['time_until_1stanswer'].dropna()

tempZ = tempZ.astype('int')

ax = sns.distplot(tempZ, kde=False, color="orange")

ax.set(xlabel='Time until 1st Answer (in days)')

#ax_data = (quest['questions_first_answers'] - quest['questions_date_added']).dt.days.rename('')

ax = ax_data.plot(kind='box', showfliers=False, grid=True, vert=False, figsize=(18, 5))

ax.set(xlabel='Time in Days', ylabel='', title='Days until FIRST Answer is Posted')

print(ax_data.describe())
temp5 = quest['questions_answers_count'].dropna()

temp5 = temp5.astype('int')

sns.set(style="white", palette="bright", color_codes=True)

# Set up the matplotlib figure

sns.despine(left=True)

# Setting axis and figure size

sns.set(rc={'figure.figsize':(10,5)})

# Plot a simple histogram with binsize determined automatically

ax = sns.distplot(temp5, kde=False, color="orange")

ax.set(xlabel='Number of Answers')
ax = temp5.plot(kind='box', showfliers=False, grid=True, vert=False, figsize=(18, 5))

ax.set(xlabel='Number of Answers', ylabel='', title='Number of Answers')

print(temp5.describe())
morethan10 = len(quest.loc[quest['questions_answers_count']>10])

print('There are {:,} instances of more than 10 answers.'.format(morethan10,))
tag_questions = pd.read_csv('../input/tag_questions.csv')

tags = pd.read_csv('../input/tags.csv')

q_t = questions.merge(right=tag_questions, how='left', left_on='questions_id', right_on='tag_questions_question_id')

q_tag = q_t.merge(right=tags, how='left', left_on='tag_questions_tag_id', right_on='tags_tag_id')

q_a_tag = q_tag.merge(right=answers, how='left', left_on='questions_id', right_on='answers_question_id')

tagnames = q_tag['tags_tag_name'].nunique()

notag = q_tag['questions_id'][q_tag['tags_tag_name'].isnull()]

print('Across the {:,} questions {} unique tags were used; {} questions had no tag'.format(questions.shape[0], tagnames, len(notag)))
# Distribution of tags 

q_tag.groupby(['questions_id'])['tags_tag_name'].nunique().describe()
sns.set(style='whitegrid', palette='bright', color_codes=True)

# Draw a violinplot of the number of tags per question

# sns.swarmplot(x=q_tag.groupby(['questions_id'])['tags_tag_name'].nunique(),data=q_tag)

ax = sns.distplot(q_tag.groupby(['questions_id'])['tags_tag_name'].nunique(), hist=True, kde=False, rug=True, bins=40)

ax.set(xlabel='Number of Unique Tags', ylabel='Number of Questions', title='Distribution of Unique Tags')

sns.despine(left=True)
# Adding count of tags to the questions dataframe

tagcount = tag_questions.groupby('tag_questions_question_id').size()

quest['questions_tag_count'] = pd.merge(quest, pd.DataFrame(tagcount.rename('tagcount')), left_on='questions_id', right_index=True, how='left')['tagcount'].fillna(0).astype(int)

ax = quest['questions_tag_count'].plot(kind='box', showfliers=False, grid=True, vert=False, figsize=(18, 5))

ax.set(xlabel='Number of Tags', ylabel='', title='Number of Tags')
#answers['time_delta_answer']

temp1 = quest.loc[:, ['questions_id', 'questions_tag_count', 'questions_answers_count']]

temp1['questions_id']=temp1['questions_id'].astype(str)

temp2 = answers.loc[:, ['answers_question_id', 'time_delta_answer']]

temp2['answers_question_id']=temp2['answers_question_id'].astype(str)

temp3 = pd.merge(temp1, temp2, left_on='questions_id', right_on='answers_question_id', how='left')

temp3.drop('answers_question_id', axis=1, inplace=True)

temp3['Response Time (days)'] = temp3['time_delta_answer']/ timedelta (days=1)

#temp3['time_delta_answer']=temp3['time_delta_answer'].total_seconds() / timedelta (days=1).total_seconds()

ax = sns.regplot(x=temp3['questions_tag_count'], y=temp3['Response Time (days)'])

ax.set(xlabel='Number of Unique Tags', ylabel='Response Time in Days', title='Questions - Number of tags versus response time')
ax = sns.regplot(x=temp3['questions_tag_count'], y=temp3['questions_answers_count'])

ax.set(xlabel='Count of Unique Tags', ylabel='Number of Answers', title='Questions - Number of tags versus number of answers')
# Exploring the variety of tags for the question with the most tags

tenlargest = q_tag.groupby(['questions_id'])['tags_tag_name'].nunique().nlargest(20)

twenlargest_1 = tenlargest.index

for i in twenlargest_1:

    print(q_tag[['questions_title', 'questions_date_added']][q_tag['questions_id'] == i].iloc[1,0])
np.count_nonzero(quest['time_until_1stanswer'].isnull()) 
quest.loc[quest['time_until_1stanswer'].isnull()].head(5)
answers.loc[answers['answers_question_id']=='dab7b240dc394d30a54dd0c5862d5fe3']
tempx = quest.copy()

tempx['ResponseTime'] = (tempx['questions_first_answers'] - tempx['questions_date_added']).dt.days

tempx.loc[tempx['ResponseTime']<0].head(3) 
questions.loc[questions['questions_id']=='2984024edb7f4661bc23b11897dd1a0e']
answers.loc[answers['answers_question_id']=='2984024edb7f4661bc23b11897dd1a0e']