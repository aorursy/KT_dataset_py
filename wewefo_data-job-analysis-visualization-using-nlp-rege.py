# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pprint
data=pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
print(data.shape)

print(data.columns)
data.head()
data.drop(['Unnamed: 0','Competitors','Easy Apply'],axis=1,inplace=True)
pd.set_option('max_colwidth',200)
data['Company Name']=data['Company Name'].str.split('\n',expand=True)[0]
job_title_str=' '.join([x for x in data['Job Title']])
import nltk

tokens = nltk.word_tokenize(job_title_str)

tokens=[x.lower() for x in tokens]
from nltk.probability import FreqDist

fdist = FreqDist(tokens)

tops=fdist.most_common(100)



# drop single word or symbol

delarr=[]

for key in fdist:

    if len(key)<2:

        delarr.append(key)

for key in delarr:

    del fdist[key]

    

tops=fdist.most_common(100)

pprint.pprint(tops)
import re

pat_lead=re.compile(r'lead|principal|iii|iv',re.I)

pat_sr=re.compile(r'senior|sr.|sr|ii',re.I)

pat_jr=re.compile(r'junior|jr|jr.|entry|i',re.I)
def job_title(row):

    if re.match(pat_lead,row['Job Title']):

        return 'Lead'

    if re.match(pat_sr,row['Job Title']):

        return 'Senior'

    if re.match(pat_jr,row['Job Title']):

        return 'Junior'

    

data['Job Level']=data.apply(job_title,axis=1)
pat=re.compile(r'\d+')

sal_df=(data['Salary Estimate'].str.findall(pat)).apply(pd.Series)

sal_df.columns=['sal_low','sal_up']

data=pd.concat([data,sal_df],axis=1)
data[['sal_low','sal_up']]=data[['sal_low','sal_up']].fillna(0)
data['sal_low']=data['sal_low'].astype('int')

data['sal_up']=data['sal_up'].astype('int')

data['sal_mean']=(data['sal_low']+data['sal_up'])/2
def job_type(row):

    if row['Job Title'].find('Business')!=-1:

        return 'Business'

    if row['Job Title'].find('Healthcare')!=-1:

        return 'healthcare'

    if row['Job Title'].find('quality')!=-1:

        return 'quality'

    if row['Job Title'].find('Reporting')!=-1:

        return 'reporting'

    if row['Job Title'].find('Financial')!=-1:

        return 'financial'

    if row['Job Title'].find('Security')!=-1:

        return 'security'

    if row['Job Title'].find('Product')!=-1:

        return 'product'

    if row['Job Title'].find('Marketing')!=-1:

        return 'marketing'
data['Job Type']=data.apply(job_type,axis=1)
job_des_str=' '.join([x for x in data['Job Description']])

tokens = nltk.word_tokenize(job_des_str)

stopwords = nltk.corpus.stopwords.words('english')





filtered_words_1 = [w.lower() for w in tokens if not w in stopwords]

filtered_words_2 = [w for w in filtered_words_1 if re.match(r'\w',w)]

fdist = FreqDist(filtered_words_2)

tops=fdist.most_common(100)

pprint.pprint(tops)
pat_p_r=re.compile(r'python| R ',re.I)

pat_tab=re.compile(r'powerbi|tableau',re.I)

pat_vis=re.compile(r'visualization',re.I)

pat_c=re.compile(r' C |C#')

pat_exc=re.compile(r'excel',re.I)

pat_sql=re.compile(r'sql|mysql|database',re.I)

pat_had=re.compile(r'hadoop|hive|spark',re.I)

pat_stat=re.compile(r'statistics|statistical',re.I)

pat_code=re.compile(r'coding|programming',re.I)
def job_skill(row):

    if re.search(pat_p_r,row['Job Description']):

        return 'Python/R'

    if re.search(pat_tab,row['Job Description']):

        return 'PowerBi/Tableau'

    if re.search(pat_vis,row['Job Description']):

        return 'Visualization'

    if re.search(pat_c,row['Job Description']):

        return 'C/C#'

    if re.search(pat_sql,row['Job Description']):

        return 'SQL/MySQL'

    if re.search(pat_had,row['Job Description']):

        return 'Hadoop/Hive/Spark'

    if re.search(pat_stat,row['Job Description']):

        return 'Statistics'

    if re.search(pat_code,row['Job Description']):

        return 'Coding'

    return None

    

data['Job Skill']=data.apply(job_skill,axis=1)
# Austin has only 81 observation.

data[data.Location.str.contains("Austin")].shape
plot_data = data.groupby('Location', as_index=False).agg({'sal_mean':'mean', 'Rating':'count'}).sort_values('Rating', 0, False).head(10)



plt.figure(figsize=(12,6))

sns.barplot(x='Location', y='Rating', data=plot_data,color='deepskyblue')

ax=plt.gca()

ax.tick_params(labelsize=10,rotation=45,axis='x') 

ax.spines['top'].set_color('none')

ax.spines['left'].set_color('none')

ax.spines['right'].set_color('none')

ax.set_xlabel('Location',fontsize=20)

ax2 = ax.twinx() 



ax2.spines['top'].set_color('none')

ax2.spines['right'].set_color('none')

ax2.spines['left'].set_color('none')



ax.set_ylabel('Analyst Demand',fontsize=20)

sns.lineplot(data=plot_data, x='Location',y='sal_mean',ax=ax2, sort=False)

ax2.set_ylabel('Mean Salary (k)',fontsize=20)



plt.title('UPDATED: Location - Demand - Salary',fontsize=20)

plt.show()
cities=data.Location.value_counts().head(10).index



plt.figure(figsize=(12,6))

sns.countplot(data[data.Location.isin(cities)]['Location'],order=cities,color='deepskyblue')

ax=plt.gca()

ax.tick_params(labelsize=10,rotation=45,axis='x') 

ax.spines['top'].set_color('none')

ax.spines['left'].set_color('none')

ax.spines['right'].set_color('none')

ax.set_xlabel('Location',fontsize=20)

ax2 = ax.twinx() 



ax2.spines['top'].set_color('none')

ax2.spines['right'].set_color('none')

ax2.spines['left'].set_color('none')



ax.set_ylabel('Analyst Demand',fontsize=20)

sns.lineplot(data=data[data.Location.isin(cities)].groupby('Location', as_index=False).mean(),x='Location',y='sal_mean',ax=ax2, sort=False)

ax2.set_ylabel('Mean Salary (k)',fontsize=20)

plt.title('Location VS Demand VS Salary',fontsize=20)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(data=data,x='Job Level',y='sal_mean',order=['Junior','Senior','Lead'],palette='Blues')

ax=plt.gca()

ax.spines['top'].set_color('none')

ax.spines['left'].set_color('none')

ax.spines['right'].set_color('none')

ax.set_ylabel('Mean Salary (k)',fontsize=25)

ax.set_xlabel('Job Level',fontsize=25)

ax.tick_params(labelsize=15) 

plt.title('Job Level VS Salary',fontsize=30)

plt.show()
data['Company Name'].value_counts().head(10).to_frame().style.set_caption('Most Demand Company').background_gradient(cmap='Blues')
data.groupby('Company Name')['sal_mean'].mean().sort_values(ascending=False).head(20).to_frame().style.set_caption('Highest Mean Salary Company')
plt.figure(figsize=(12,8))

sns.boxplot(data=data,x='Job Type',y='sal_mean',palette='Blues')

ax=plt.gca()

ax.spines['top'].set_color('none')

ax.spines['left'].set_color('none')

ax.spines['right'].set_color('none')

ax.set_ylabel('Mean Salary (k)',fontsize=25)

ax.set_xlabel('Job Type',fontsize=25)

ax.tick_params(labelsize=15) 

plt.title('Job Type VS Salary',fontsize=30)

plt.show()
plt.figure(figsize=(12,8))

sns.heatmap(data[data.Location.isin(cities)].groupby(by=['Location','Job Type'])['sal_mean'].mean().unstack().fillna(0),cmap='Blues',annot=True)

ax=plt.gca()

ax.set_ylabel('Location',fontsize=25)

ax.set_xlabel('Job Type',fontsize=25)

ax.tick_params(labelsize=15,rotation=45) 

plt.title('Location VS Job Type VS Salary',fontsize=30)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(data=data,x='Job Skill',y='sal_mean',palette='Blues')

ax=plt.gca()

ax.spines['top'].set_color('none')

ax.spines['left'].set_color('none')

ax.spines['right'].set_color('none')

ax.set_ylabel('Mean Salary (k)',fontsize=25)

ax.set_xlabel('Job Skill',fontsize=25)

ax.tick_params(labelsize=15,rotation=45) 

plt.title('Job Skill VS Salary',fontsize=30)

plt.show()
plt.figure(figsize=(12,8))

sns.heatmap(data.groupby(by=['Job Skill','Job Type'])['sal_mean'].mean().unstack().fillna(0),cmap='Blues',annot=True)

ax=plt.gca()

ax.set_ylabel('Job Skill',fontsize=25)

ax.set_xlabel('Job Type',fontsize=25)

ax.tick_params(labelsize=15,rotation=45) 

plt.title('Job Skill VS Job Type VS Salary',fontsize=30,pad=20)

plt.show()
lists=['r','python','sql','database','powerbi','tableau','visualization',

       'C','C#','excel','mysql','hadoop','hive','sparl','statistics','etl'

       ,'a/b','sas','bi','algorithm','ai','deep learning','machine learning']
import wordcloud

d_tmp = dict((key, value) for key, value in fdist.items() if key in lists)

w=wordcloud.WordCloud(background_color='White',scale=4)

fig = plt.figure(figsize=(12, 8))

w.generate_from_frequencies(d_tmp)

plt.axis('off')

plt.imshow(w)