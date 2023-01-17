# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

otr = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

mcr.drop(index=0, inplace=True)

otr.drop(index=0, inplace=True)
populations = pd.read_csv('/kaggle/input/world-population-2019/world_population_2019.csv')
similar_sized_countries_population = populations[(populations.pop2019 < 6000000) & (populations.pop2019 > 3000000)]

similar_sized_countries_name = similar_sized_countries_population.name

similar_sized_countries = pd.merge(mcr, similar_sized_countries_name, how='inner', left_on='Q3', right_on='name')

similar_sized_countries.drop(columns='name', inplace=True)

print(similar_sized_countries.Q3.unique())

five_countries_pop = populations[(populations.name == 'Ireland') | (populations.name == 'Singapore') | (populations.name == 'Norway') \

     | (populations.name == 'Denmark') | (populations.name == 'New Zealand')]
NZ = mcr[mcr.Q3 == 'New Zealand']

SI = mcr[mcr.Q3 == 'Singapore']

DM = mcr[mcr.Q3 == 'Denmark']

NW = mcr[mcr.Q3 == 'Norway']

IL = mcr[mcr.Q3 == 'Ireland']

five_countries_pop['num_responses'] = [len(SI), len(DM), len(NW), len(IL), len(NZ)]

five_countries_pop['responses_per_million'] = five_countries_pop['num_responses'] / (five_countries_pop['pop2019'] / 1000000.0)

print(five_countries_pop[['name', 'pop2019', 'num_responses', 'responses_per_million']])
sns.set_context('notebook')

fig = plt.figure(figsize=(8, 6))

plt.bar(data=five_countries_pop, x='name', height='responses_per_million', edgecolor='black')

plt.ylabel('Number of responses per million population')

plt.show()
# Company sizes of the NZ respondents

NZ.fillna(value={'Q6':'Not Available'}, inplace=True)

NZ_company_size = NZ.groupby('Q6').Q1.count().reset_index()

NZ_company_size['size'] = [0, 3, 2, 1, 4, 5]

NZ_company_size.sort_values(by=['size'], ascending=False, inplace=True)

print(NZ_company_size)
size = NZ_company_size.Q6.values

y_pos = range(len(size))

fig = plt.figure(figsize=(8, 6))

ax = plt.subplot()

plt.barh(y_pos, NZ_company_size.Q1, edgecolor='black')

plt.yticks(y_pos)

ax.set_yticklabels(size)

ax.set_xlabel('Company counts')

plt.show()
#Q7: Approximately how many individuals are responsible for data science workloads at your place of business?

NZ['Q7_reformatted'] = NZ.Q7.apply(lambda x: '1-2' if x == '2-Jan' else '3-4' if x == '4-Mar' else '5-9' if x == '9-May' \

                                  else '10-14' if x == '14-Oct' else 'Not Available' if pd.isnull(x) else x)

NZ_compsize_numdataind = NZ.groupby(['Q6', 'Q7_reformatted']).Q1.count().reset_index()

NZ_compsize_numdataind.columns = ['Company size', 'Number of individuals', 'Count']
fig = plt.figure(figsize=(15, 6))

ax = plt.subplot()

comp_size_ordered = ['Not Available', '0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']

num_ind_ordered = ['0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+', 'Not Available']

sns.barplot(data=NZ_compsize_numdataind, x='Company size', y='Count', hue='Number of individuals', order=comp_size_ordered, hue_order=num_ind_ordered)

plt.show()
NZ.fillna(value={'Q8':'Not Available'}, inplace=True)

NZ_use_ML = NZ.groupby('Q8').Q1.count().reset_index()



fig = plt.figure(figsize=(8, 6))

plt.barh(NZ_use_ML.Q8, NZ_use_ML.Q1, edgecolor='black')

plt.xlabel('Company counts')

plt.show()
def multiple_choice(df, q, n):

    dataframe = pd.DataFrame(columns=['col1', 'col2'])

    for i in range(n):

        data = pd.DataFrame([[df['Q{}_Part_{}'.format(q, i+1)].dropna().iloc[0], df['Q{}_Part_{}'.format(q, i+1)].count()]], \

                           columns=['col1', 'col2'])

        dataframe = dataframe.append(data, ignore_index=True)

    return dataframe



NZ_Q9_answers_count = multiple_choice(NZ, 9, 8)

NZ_Q9_answers_count.columns=['Activity', 'Count']

NZ_Q9_answers_count.sort_values(by=['Count'], ascending=False, inplace=True)

NZ_Q9_answers_count.index = range(8)

NZ_Q9_answers_count.style

NZ_Q9_answers_count.style
# Job titles of the NZ respondents

NZ_job_titles = NZ.groupby('Q5').Q1.count().reset_index()

print(NZ_job_titles)
fig = plt.figure(figsize=(6, 6))

plt.pie(NZ_job_titles.Q1, labels=NZ.Q5.unique(), autopct='%0.2f%%')

plt.axis('equal')

plt.show()
# What are the job titles in other? None of them is specified.

NZ_Q5_other_index = mcr[(mcr.Q3 == 'New Zealand') & (mcr.Q5 == 'Other')].index.values

NZ_Q5_other_text = otr.loc[NZ_Q5_other_index].Q5_OTHER_TEXT

print(NZ_Q5_other_text)
NZ_salary = NZ.groupby('Q10').Q1.count().reset_index()

NZ_salary.iloc[0,0] = '0-999'



NZ_grouped_salary = pd.DataFrame([['less than 20k', 4], ['20k-40k', 1], ['40k-60k', 9], ['60k-80k', 11], \

                                 ['80k-100k', 6], ['more than 100k', 8]], columns=['Salary range', 'Count'])



fig = plt.figure(figsize=(8, 6))

plt.bar(data=NZ_grouped_salary, x='Salary range', height='Count', edgecolor='black')

plt.xlabel('Salary range in USD')

plt.ylabel('Count')

plt.show()
AUSNZ = mcr[(mcr.Q3 == 'Australia') | (mcr.Q3 == 'New Zealand')]

print(len(AUSNZ))
AUSNZ_Q13_answers_count = multiple_choice(AUSNZ, 13, 12)

AUSNZ_Q13_answers_count.columns = ['Source', 'Count']

AUSNZ_Q13_answers_count.sort_values(by=['Count'], ascending=False, inplace=True)

AUSNZ_Q13_answers_count.index = range(12)

print(AUSNZ_Q13_answers_count)
fig = plt.figure(figsize=(8, 6))

plt.barh(AUSNZ_Q13_answers_count.Source, AUSNZ_Q13_answers_count.Count, edgecolor='black')

plt.show()
AUSNZ_reclan = AUSNZ.groupby('Q19').Q1.count().reset_index()

AUSNZ_reclan.columns = ['Recommendation', 'Count']

AUSNZ_reclan.sort_values(by=['Count'], ascending=False, inplace=True)

AUSNZ_reclan.index = range(10)

AUSNZ_reclan.style
AUSNZ_Q18_answers_count = multiple_choice(AUSNZ, 18, 12)

AUSNZ_Q18_answers_count.columns = ['Language', 'Count']

AUSNZ_Q18_answers_count.sort_values(by=['Count'], ascending=False, inplace=True)

AUSNZ_Q18_answers_count.index = range(12)

AUSNZ_Q18_answers_count.style
AUSNZ_Q18_other_index = mcr[((mcr.Q3 == 'New Zealand') | (mcr.Q3 == 'Australia')) & (mcr.Q18_Part_12 == 'Other')].index.values

AUSNZ_Q18_other_text = otr.loc[AUSNZ_Q18_other_index].Q18_OTHER_TEXT

print(AUSNZ_Q18_other_text.unique())