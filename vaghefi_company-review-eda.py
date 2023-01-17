import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import re

import plotly.express as px

import plotly.graph_objects as go

plt.style.use('ggplot')
reviews = pd.read_csv('../input/company-reviews/company_reviews.csv')

reviews.head()
# data type

reviews.dtypes.value_counts().plot(kind='pie');
# percentage of missing values

reviews.isna().mean().sort_values().plot(kind='barh', color='tab:blue');
# Distribution of Companies based on Reveneu

reviews['revenue'].value_counts().plot(kind='barh');
reviews['industry'].value_counts().tail(10)
reviews['industry'] = reviews['industry'].str.replace('\n.*', '', flags=re.DOTALL)

reviews['industry'] = reviews['industry'].replace({'Health Care': 'Healthcare', 'Construction': 'Construction & Facilities Services',

                                                  'Retail': 'Retail & Wholesale', 'Education and Schools': 'Education',

                                                  'Industrial Manufacturing': 'Manufacturing', 'Auto': 'Automotive', 

                                                   'Transport and Freight': 'Transportation & Logistics', 'Organization': 'Nonprofit & NGO',

                                                  'Pharmaceuticals': 'Pharmaceutical & Biotechnology', 'Human Resources and Staffing': 'Human Resources & Staffing',

                                                  'Internet and Software': 'Information Technology', 'Agriculture': 'Agriculture and Extraction'})
# Distribution of Companies based on Industry - Top 20

reviews['industry'].value_counts().plot(kind='bar', figsize=(14,5));
# Clean-up CEO related columns

reviews['ceo_approval'] = reviews['ceo_approval'].str.replace('%', '').astype(float)

reviews['ceo_count'] = reviews['ceo_count'].str.replace(',', '').str.extract('(\d+)').astype(float)
# Distribution of ceo_count and ceo_approval

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# since ceo_count has large variation, we plot it in log-scale

sns.kdeplot(np.log10(reviews['ceo_count']), shade=True)

plt.subplot(1,2,2)

sns.kdeplot(reviews['ceo_approval'], shade=True, color='tab:blue');
# Boxplot of CEO approval vs. Industry

plt.figure(figsize=(16,5))

order = reviews.groupby('industry')['ceo_approval'].median().sort_values()

sns.boxenplot(x='industry', y='ceo_approval', data=reviews, order=order.index)

plt.xticks(rotation=90);
reviews['reviews'] = reviews['reviews'].str.replace(',', '').str.extract('(\d+)').astype(float)
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# since reviews has large variation, we plot it in log-scale

sns.kdeplot(np.log10(reviews['reviews']), shade=True)

plt.subplot(1,2,2)

sns.kdeplot(reviews['rating'], shade=True, color='tab:blue');
sns.scatterplot(x='rating', y='ceo_approval', data=reviews, alpha=0.2)

plt.xlabel('Rating'); plt.ylabel('CEO Approval');
salaries = pd.DataFrame([(i, key, val) for i, x in enumerate(reviews['salary']) for key, val in eval(x).items()], columns=['index', 'title', 'salary']) 

salaries.head()
# separate unit and salary 

salaries[['salary', 'unit']] = salaries['salary'].str.replace(',|\$', '').str.split(' per ', expand=True)

salaries['unit'].value_counts()
# Create a new column and onvert all of them to yearly salary

salaries['yearly_salary'] = salaries['salary'].astype(float) * salaries['unit'].replace({'hour': 52*40, 'year': 1, 'week': 52, 'month': 12, 'day': 262})

salaries.head()
# highest and lowest paid jobs

salary_title = salaries.groupby('title')['yearly_salary'].mean().sort_values()

pd.concat((salary_title.head(10), salary_title.tail(10))).plot(kind='barh', color='tab:orange', figsize=(6, 6));
# average salary per industry

industry_salary = reviews[['rating', 'industry']].join(salaries.set_index('index')).dropna().sort_values('yearly_salary')

industry_salary.groupby('industry')['yearly_salary'].mean().sort_values().plot(kind='pie', figsize=(8,8));
# relationship between salary and company rating

plt.scatter(industry_salary['rating'], industry_salary['yearly_salary'], alpha=0.2);
reviews['website'].str.split('\n').explode().value_counts().head(10).plot(kind='bar',  color='tab:blue');
sns.heatmap(pd.crosstab(reviews['interview_experience'],reviews['interview_difficulty'], normalize='columns')*100, annot=True, fmt='.1f', vmin=0, vmax=100);
# clean up location column

locations = pd.DataFrame([(i, key, val) for i, x in enumerate(reviews['locations']) for key, val in eval(x).items()], columns=['index', 'location', 'rating']) 

locations[['city', 'state']] = locations['location'].str.split(', ', expand=True)

locations['rating'] = locations['rating'].astype(float)

locations.head()
# average company rating per state (looks like US Election Electoral Map :) )

state_rating = locations.groupby('state')['rating'].mean().reset_index()

state_rating['rating'] = np.where(state_rating['rating'] > 4.4, 4.4, state_rating['rating'])

fig = go.Figure(data=go.Choropleth(locations=state_rating['state'], z = state_rating['rating'], locationmode = 'USA-states', 

                                    colorscale = 'RdBu', colorbar_title = "Job Rating"))

fig.update_layout(geo_scope='usa')



fig.show()
# number of job ratings per state

locations.groupby('state').size().sort_values(ascending=False).plot(kind='bar', figsize=(16,4), color='tab:green');
# clean up happiness column

happiness = pd.DataFrame([(i, key, val) for i, x in enumerate(reviews['happiness']) for key, val in eval(x).items()], columns=['index', 'item', 'rating']) 

happiness['rating'] = happiness['rating'].astype(float)

happiness.head()
# happiness score based on industry

industry_happiness = reviews[['industry']].join(happiness.set_index('index')).dropna()

sort_index = industry_happiness.groupby('industry')['rating'].mean().sort_values().index

industry_happiness.groupby(['industry', 'item'])['rating'].mean().sort_values().unstack().loc[sort_index,:].plot(kind='barh', stacked=True, figsize=(12,8));
# company overall rating vs employee happiness

rating_happiness = reviews[['rating']].join(happiness.query('item == "Work Happiness Score"').set_index('index')[['rating']].rename(columns={'rating': 'happiness'})).dropna()

plt.figure(figsize=(8,6))

sns.regplot(x='rating', y='happiness', data=rating_happiness, scatter_kws={'alpha':0.3}, color='tab:green' );
# least and most happiest titles

title_happiness = salaries.set_index('index')[['title']].join(happiness.query('item == "Work Happiness Score"').set_index('index')[['rating']].rename(columns={'rating': 'happiness'})).dropna()

title_happiness.groupby('title').filter(lambda x: x.shape[0] > 20).groupby('title')['happiness'].mean().sort_values().plot(kind='bar', figsize=(26, 8), color='tab:blue');