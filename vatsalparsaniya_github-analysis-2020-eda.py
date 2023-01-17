# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Github_data = pd.read_csv("../input/github-repositories-analysis/Github_data.csv")
Github_data.head()
Github_data = Github_data.rename(columns={'Unnamed: 0': 'index', 'Unnamed: 0.1': 'sub_index'})

Github_data.drop('index', axis=1, inplace=True)

Github_data.reset_index(drop=True, inplace=True)
Github_data.info()
Github_data.at[700, 'issue'] = str(5000)
Numerical_columns = ["star","fork","watch","issue","pull_requests","projects","commits","branches","packages","releases","contributers"]

# Github_data[Numerical_columns] = Github_data[Numerical_columns].apply(lambda x: x.replace(',','').astype(float) if ',' in str(x) else x)

Github_data[Numerical_columns] = Github_data[Numerical_columns].fillna(0)

Github_data["issue"] = Github_data["issue"].apply(lambda x: x.replace(',', '') if ',' in x else x).astype(float)

Github_data["pull_requests"] = Github_data["pull_requests"].apply(lambda x: x.replace(',', '') if ',' in x else x).astype(float)

Github_data["commits"] = Github_data["commits"].apply(lambda x: x.replace(',', '') if ',' in x else x).astype(float)

Github_data["branches"] = Github_data["branches"].apply(lambda x: x.replace(',', '') if ',' in x else x).astype(float)

Github_data["contributers"] = Github_data["contributers"].apply(lambda x: x.replace(',', '') if ',' in x else x).astype(float)
Github_data['star'] = Github_data['star'].apply(lambda x: float(x.replace('k',''))*1000 if 'k' in x else x)

Github_data['fork'] = Github_data['fork'].apply(lambda x: float(x.replace('k',''))*1000 if 'k' in x else x)

Github_data['watch'] = Github_data['watch'].apply(lambda x: float(x.replace('k',''))*1000 if 'k' in x else x)
import ast

unique_tags = []

Github_data['topic_tag'].apply(lambda x: unique_tags.append(ast.literal_eval(x)))

# unique_tags = list(set([item for sublist in unique_tags for item in sublist]))

all_tag = np.array([item for sublist in unique_tags for item in sublist])

unique, counts = np.unique(all_tag, return_counts=True)

print("Total number of tags in 1500 repository : ",len(all_tag))

print("Total number of unique tags in 1500 repository : ",len(unique))

tag_dataframe = pd.DataFrame({"unique":unique,"counts":counts})

tag_dataframe = tag_dataframe.sort_values(['counts'],ascending=[False])
fig = px.bar(tag_dataframe[:20],x="unique",y="counts",color='counts')

fig.show()
Github_data['star'] = Github_data['star'].astype(float)

star_topicwise = Github_data.groupby('topic').sum()['star']

fig = px.bar(star_topicwise,x=star_topicwise.index,y="star",color=star_topicwise.index)

fig.show()
Github_data['fork'] = Github_data['fork'].astype(float)

fork_topicwise = Github_data.groupby('topic').sum()['fork']

fig = px.bar(fork_topicwise,x=fork_topicwise.index,y="fork",color=fork_topicwise.index)

fig.show()
Github_data['watch'] = Github_data['watch'].astype(float)

watch_topicwise = Github_data.groupby('topic').sum()['watch']

fig = px.bar(watch_topicwise,x=watch_topicwise.index,y="watch",color=watch_topicwise.index)

fig.show()
colormap = plt.cm.magma

plt.figure(figsize=(5,5))

plt.title('correlation between star and fork', y=1.05, size=15)

sns.heatmap(Github_data[['star','fork','watch']].corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='white', annot=True)
plt.figure(figsize=(10,10))

plt.title('correlation between star and fork', y=1.05, size=15)

sns.heatmap(Github_data[Numerical_columns].corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='white', annot=True)
commit_topicwise = Github_data.groupby('topic').sum()['commits']

fig = px.pie(commit_topicwise, values='commits', names=commit_topicwise.index, title='Commit Distribution topic wise')

fig.show()
issue_topicwise = Github_data.groupby('topic').sum()['issue']

fig = px.bar(issue_topicwise,x=issue_topicwise.index,y="issue",color=issue_topicwise.index)

fig.show()
contributers_topicwise = Github_data.groupby('topic').sum()['contributers']

fig = go.Figure(data=[go.Pie(labels=contributers_topicwise.index, values=contributers_topicwise.values, hole=.3)])

fig.show()
License_distribution = Github_data["License"].apply(lambda x: x if'\n' not in x else None).value_counts()

fig = px.bar(y=License_distribution.values, x=License_distribution.index)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
github_group = Github_data.groupby('topic')

num_of_top_repository = 10

fig = go.Figure()

for name, group in github_group:

    

    fig.add_trace(go.Bar(

    x=list(range(1,num_of_top_repository+1)),

    y=group["star"].values[:num_of_top_repository+1],

    name=name,

    ))

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
github_group = Github_data.groupby('topic')

num_of_top_repository = 10

fig = go.Figure()

for name, group in github_group:

    

    fig.add_trace(go.Bar(

    x=list(range(1,num_of_top_repository+1)),

    y=group["fork"].values[:num_of_top_repository+1],

    name=name,

    ))

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
github_group = Github_data.groupby('topic')

num_of_top_repository = 10

fig = go.Figure()

for name, group in github_group:

    

    fig.add_trace(go.Bar(

    x=list(range(1,num_of_top_repository+1)),

    y=group["contributers"].values[:num_of_top_repository+1],

    name=name,

    ))

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()