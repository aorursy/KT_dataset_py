%matplotlib inline



import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

plotly.tools.set_credentials_file(username='rounakbanik', api_key='xTLaHBy9MVv5szF4Pwan')

import warnings

warnings.filterwarnings('ignore')
mcq = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

mcq.shape
ff = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1", low_memory=False)

ff.shape
sns.countplot(y='GenderSelect', data=mcq)
con_df = pd.DataFrame(mcq['Country'].value_counts())

con_df['country'] = con_df.index

con_df.columns = ['num_resp', 'country']

con_df = con_df.reset_index().drop('index', axis=1)

con_df.head(10)
data = [ dict(

        type = 'choropleth',

        locations = con_df['country'],

        locationmode = 'country names',

        z = con_df['num_resp'],

        text = con_df['country'],

        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],

        autocolorscale = False,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Survey Respondents'),

      ) ]



layout = dict(

    title = 'Survey Respondents by Nationality',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='survey-world-map' )
mcq['Age'].describe()
sns.distplot(mcq[mcq['Age'] > 0]['Age'])
sns.countplot(y='FormalEducation', data=mcq)
sns.countplot(y='EmploymentStatus', data=mcq)
sns.countplot(y='Tenure', data=mcq)
sns.countplot(y='LanguageRecommendationSelect', data=mcq)
mcq[mcq['CurrentJobTitleSelect'].notnull()]['CurrentJobTitleSelect'].shape
data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & ((mcq['LanguageRecommendationSelect'] == 'Python') | (mcq['LanguageRecommendationSelect'] == 'R'))]

plt.figure(figsize=(8, 10))

sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=data)
data = mcq['MLToolNextYearSelect'].value_counts().head(15)

sns.barplot(y=data.index, x=data)
data = mcq['MLMethodNextYearSelect'].value_counts().head(15)

sns.barplot(y=data.index, x=data)
mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str').apply(lambda x: x.split(','))

s = mcq.apply(lambda x: pd.Series(x['LearningPlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'platform'
plt.figure(figsize=(6,8))

data = s[s != 'nan'].value_counts()

sns.barplot(y=data.index, x=data)
use_features = [x for x in mcq.columns if x.find('LearningPlatformUsefulness') != -1]
fdf = {}

for feature in use_features:

    a = mcq[feature].value_counts()

    a = a/a.sum()

    fdf[feature[len('LearningPlatformUsefulness'):]] = a

#fdf = pd.DataFrame(fdf)

fdf = pd.DataFrame(fdf).transpose().sort_values('Very useful', ascending=False)

fdf
cat_features = [x for x in mcq.columns if x.find('LearningCategory') != -1]
cdf = {}

for feature in cat_features:

    cdf[feature[len('LearningCategory'):]] = mcq[feature].mean()



cdf = pd.Series(cdf)



plt.pie(cdf, labels=cdf.index, 

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.title("Contribution of each Platform to Learning")

plt.show()
mcq[mcq['HardwarePersonalProjectsSelect'].notnull()]['HardwarePersonalProjectsSelect'].shape
mcq['HardwarePersonalProjectsSelect'] = mcq['HardwarePersonalProjectsSelect'].astype('str').apply(lambda x: x.split(','))

s = mcq.apply(lambda x: pd.Series(x['HardwarePersonalProjectsSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'hardware'
s[s != 'nan'].value_counts()
plt.figure(figsize=(8,8))

sns.countplot(y='TimeSpentStudying', data=mcq, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))
mcq['BlogsPodcastsNewslettersSelect'] = mcq['BlogsPodcastsNewslettersSelect'].astype('str').apply(lambda x: x.split(','))
s = mcq.apply(lambda x: pd.Series(x['BlogsPodcastsNewslettersSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'platforms'
s = s[s != 'nan'].value_counts()
plt.figure(figsize=(6,8))

plt.title("Most Popular Blogs and Podcasts")

sns.barplot(y=s.index, x=s)
mcq['CoursePlatformSelect'] = mcq['CoursePlatformSelect'].astype('str').apply(lambda x: x.split(','))
t = mcq.apply(lambda x: pd.Series(x['CoursePlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)

t.name = 'courses'
t = t[t != 'nan'].value_counts()
plt.title("Most Popular Course Platforms")

sns.barplot(y=t.index, x=t)
job_features = [x for x in mcq.columns if x.find('JobSkillImportance') != -1 and x.find('JobSkillImportanceOther') == -1]
jdf = {}

for feature in job_features:

    a = mcq[feature].value_counts()

    a = a/a.sum()

    jdf[feature[len('JobSkillImportance'):]] = a

#fdf = pd.DataFrame(fdf)

jdf = pd.DataFrame(jdf).transpose().sort_values('Necessary', ascending=False)

jdf
mcq[mcq['CompensationAmount'].notnull()].shape
def clean_salary(x):

    x = x.replace(',', '')

    try:

        return float(x)

    except:

        return np.nan
def salary_stats(country):

    data = mcq[(mcq['CompensationAmount'].notnull()) & (mcq['Country'] == country) ]

    data['CompensationAmount'] = data['CompensationAmount'].apply(clean_salary)

    print(data[data['CompensationAmount'] < 1e9]['CompensationAmount'].describe())

    sns.distplot(data[data['CompensationAmount'] < 1e9]['CompensationAmount'])
salary_stats('India')
salary_stats('United States')