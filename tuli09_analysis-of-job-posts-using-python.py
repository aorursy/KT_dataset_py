import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import spacy

from collections import Counter

import warnings



warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
df = pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")

df.head()
# drop the column 'Unnamed: 0'

del df["Unnamed: 0"]
print("The shape of our dataframe is : {}".format(df.shape))
df.dtypes
df.isnull().sum()
# number of unique elements in each column

df.nunique()
df['Rating'].unique()
df['Size'].unique()
df.replace([-1.,-1, '-1', 'Unknown', 'Unknown / Non-Applicable'], np.nan, inplace=True)
# check our dataframe after the replacement

df.head()
# calculate % of null value in each column and sort them in descending order

null_percentage = df.isnull().sum().sort_values(ascending=False)/len(df)*100



# make a plot

null_percentage[null_percentage>0.1].plot(kind='bar', figsize=(10,8))

plt.xlabel("Features")

plt.ylabel("% of null values");
df.drop(['Easy Apply', 'Competitors'], axis=1, inplace=True)

print("The shape of the dataframe after dropping features : {}".format(df.shape))
# calculate % of null value in each row and sort them in descending order

row_null_percentage = df.isnull().sum(axis=1).sort_values(ascending=False)/len(df)*100

row_null_percentage
# top 30 industries hiring data analysts

df['Industry'].value_counts()[:30].plot(kind='bar', figsize=(14,8))

plt.xlabel("Industry")

plt.ylabel("Number of job posts");
# first 10 entries in 'Salary Estimate'

df['Salary Estimate'][:10]
# the minimum salary offered

df['min_salary'] = df['Salary Estimate'].apply(lambda x:float(x.split()[0].split("-")[0].strip("$,K")) 

                                                          if not pd.isnull(x) else x)



# the maximum salary offered

df['max_salary'] = df['Salary Estimate'].apply(lambda x:float(x.split()[0].split("-")[1].strip("$,K")) 

                                                          if not pd.isnull(x) else x)



# the average salary offered

df['avg_salary'] = (df['min_salary'] + df['max_salary'])/2.



# drop the original column

df.drop('Salary Estimate', axis=1, inplace=True)
# plot top 30 industries with highest offered salaries

df.groupby('Industry')['avg_salary'].mean().sort_values(ascending=False)[:30].plot(kind='bar', figsize=(14,10))

plt.xlabel('Industry')

plt.ylabel('Average salary');
# the 'Location' column

df['Location']
df['Job_state'] = df['Location'].apply(lambda x:x.split(",")[-1].strip())

df['Job_state']
# how many unique values?

df['Job_state'].nunique()
# plot total number of job posting in each state

df['Job_state'].value_counts().plot(kind="bar", figsize=(14,8))

plt.xlabel("Job Location")

plt.ylabel("Number of job posts");
df['Company Name'].unique()[:20]
# extract only the company name

df['Company Name'] = df['Company Name'].apply(lambda x:x.split("\n")[0].strip() if not pd.isnull(x) else x)
# plot top 30 companies with high job postings

df['Company Name'].value_counts()[:30].plot(kind='bar', figsize=(14,10))

plt.xlabel('Company')

plt.ylabel('Number of job posts');
# plot top 30 companies with high average salaries

df.groupby('Company Name')['avg_salary'].mean().sort_values(ascending=False)[:30].plot(kind='bar', figsize=(14,10))

plt.xlabel('Company')

plt.ylabel('Average salary');
# store top 30 companies offering high salaries in a list

top_30_comps = list(df.groupby('Company Name')['avg_salary'].mean().sort_values(ascending=False)[:30].index)
print("The rating is given on a scale {}-{}.".format(df['Rating'].min(), df['Rating'].max()))
# plot ratings of these companies

plt.figure(figsize=(14,8))

sns.barplot(x=df[df['Company Name'].isin(top_30_comps)]['Company Name'], 

            y=df[df['Company Name'].isin(top_30_comps)]['Rating'],

            order = top_30_comps)

plt.xlabel('Company')

plt.xticks(rotation=90);
# unique values in the feature 'Size'

df['Size'].unique()
# Employee size of top 30 high paying companies

plt.figure(figsize=(10,8))

sns.countplot(df[df['Company Name'].isin(top_30_comps)]['Size'])   

plt.xlabel('Employee Size')

plt.xticks(rotation=90);
# unique values in 'Revenue'

df['Revenue'].unique()
# plot revenues of top 30 high paying companies

plt.figure(figsize=(10,8))

sns.countplot(df[df['Company Name'].isin(top_30_comps)]['Revenue'])    

plt.xticks(rotation=90);
# plot locations of these top 30 companies

plt.figure(figsize=(10,8))

sns.countplot(df[df['Company Name'].isin(top_30_comps)]['Location'])    

plt.xlabel("Company Location")

plt.xticks(rotation=90);
desc_len = [len(desc) for desc in df['Job Description']]

plt.figure(figsize=(14,8))

plt.xlabel('Job descripiton length')

plt.hist(desc_len, bins=80, range=(0,4000));
# load the required libraries and create an nlp object

nlp = spacy.load('en_core_web_sm')
# list to store extracted skill keywords

skill_list = []



# feed the entire corpus into batches of 100 samples at a time

for i in range(0,len(df), 100):

    # for the last batch

    if i+np.mod(2253,100)==len(df):

        # combine job descriptions of 100 samples into a single string

        text = " ".join(des for des in df['Job Description'][i:len(df)])

    else :

        text = " ".join(des for des in df['Job Description'][i:i+100])

        

    # process raw text with the nlp object that holds all information about the tokens, their linguistic 

    #features and relationships    

    doc = nlp(text)



    # loop over the named entities

    for entity in set(doc.ents):

        # select entities with label 'ORG'

        if entity.label_ == 'ORG':

            # add to the list

            skill_list.append(entity.text)
# count how many times each entity appears in the list

word_count = Counter(skill_list)

# print the top 100 named entities

word_count.most_common(100)
# make a list of actual skills extracted from the corpus

skill_set = ['SQL', 'Python', 'ETL', 'SAS', 'SAP', 'Oracle', 'PowerPoint', 'AWS', 'Microsoft Office',

             'XML', 'PL/SQL', 'AI', 'Spark', 'MS Office', 'ERP', 'Big Data',  'Tableau', 'Hadoop', 

             'JavaScript', 'Azure', 'Perl']



# loop over top 100 extracted skill keywords/phrases

# select skills present in the above list

# add to a dictionary    

skill_count_dict = {skill:count for skill, count in word_count.most_common(100) if skill in skill_set}        

            

# SQL and SQL server basically point to the same thing. Let's combine them into a single key            

skill_count_dict['SQL'] = skill_count_dict['SQL'] + skill_count_dict['PL/SQL']



# remove the other key

del skill_count_dict['PL/SQL']
# create a dataframe with two columns - skills and corresponding counts

skill_count_df = pd.DataFrame(skill_count_dict.items(), columns=['Skill', 'Total Count'])

skill_count_df
# plot how many times a skill appeared in the corpus

skill_count_df.groupby('Skill')['Total Count'].max().plot(kind='bar', figsize=(14,8))

plt.xlabel("Required skills")

plt.ylabel("Total count");